"""Run the fixed-rule answer-expansion prompt with a local causal LM."""
from __future__ import annotations

import gc
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cse_dpo.candidate_bank_class_judge import candidate_is_extractive
from src.utility.data import clean_text
from src.utility.eval_models import (
    first_model_device,
    load_model_and_tokenizer_for_eval,
    prime_unsloth_runtime,
)
from src.utility.eval_types import ModelSpec

from .answer_variants import read_jsonl, write_json, write_jsonl
from .coverage_comparison import (
    EXPANSION_PROMPT,
    EXTRACTIVE_CANDIDATE_TYPES,
    EXTRACTIVE_EXPANSION_PROMPT,
    model_context,
    official_candidate_matches,
    validate_response,
)


DEFAULT_MODEL = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"


def _json_candidates(text: str):
    """Yield JSON objects embedded in plain text or a fenced response."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_expansion_response(text: str) -> list[dict[str, str]]:
    """Parse and strictly validate the first valid expansion object."""
    errors = []
    for value in _json_candidates(text):
        try:
            answers = validate_response(value, "expansion")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        seen: set[str] = set()
        unique = []
        for answer in answers:
            key = re.sub(r"\s+", " ", answer["answer"].casefold()).strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(answer)
        if not unique or unique[0]["relation_type"] != "original":
            raise ValueError("Expansion lost its required original answer after deduplication")
        return unique
    suffix = f" ({'; '.join(errors)})" if errors else ""
    raise ValueError("No valid expansion JSON object was found" + suffix)


def parse_extractive_response(
    text: str,
    snippets: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Parse extractive candidates and prove literal containment in the cited snippet."""
    snippet_by_id = {str(row["snippet_id"]): str(row.get("text", "")) for row in snippets}
    errors = []
    for value in _json_candidates(text):
        try:
            if set(value) != {"answers"} or not isinstance(value["answers"], list):
                raise ValueError("Expected one answers array")
            if not 1 <= len(value["answers"]) <= 10:
                raise ValueError("Extractive expansion must return 1 to 10 answers")
            seen: set[str] = set()
            answers = []
            for index, row in enumerate(value["answers"], 1):
                if not isinstance(row, dict) or set(row) != {"answer", "snippet_id", "candidate_type"}:
                    raise ValueError(f"Candidate {index} has unexpected fields")
                answer = row["answer"]
                snippet_id = str(row["snippet_id"])
                candidate_type = row["candidate_type"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Candidate {index} has an empty answer")
                if snippet_id not in snippet_by_id:
                    raise ValueError(f"Candidate {index} cites unknown snippet {snippet_id!r}")
                if candidate_type not in EXTRACTIVE_CANDIDATE_TYPES:
                    raise ValueError(f"Candidate {index} has invalid candidate_type {candidate_type!r}")
                if answer not in snippet_by_id[snippet_id]:
                    raise ValueError(
                        f"Candidate {index} is not a literal substring of snippet {snippet_id}"
                    )
                key = re.sub(r"\s+", " ", answer.casefold()).strip()
                if key in seen:
                    continue
                seen.add(key)
                answers.append({
                    "answer": answer,
                    "snippet_id": snippet_id,
                    "candidate_type": candidate_type,
                })
            if not answers:
                raise ValueError("No distinct extractive candidates remain")
            return answers
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    suffix = f" ({'; '.join(errors)})" if errors else ""
    raise ValueError("No valid extractive expansion JSON object was found" + suffix)


def _render(tokenizer: Any, system_prompt: str, example: dict[str, Any], snippets: list[dict[str, Any]]) -> str:
    user = {"question": example["question"],
            "snippets": [{"id": s["snippet_id"], "text": s["text"]} for s in snippets]}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_prompt(
    tokenizer: Any,
    example: dict[str, Any],
    *,
    system_prompt: str = EXPANSION_PROMPT,
    max_prompt_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Keep the fixed instructions/question and pack complete snippets in source order."""
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    selected: list[dict[str, Any]] = []
    prompt = _render(tokenizer, system_prompt, example, selected)
    base_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    if base_tokens > max_prompt_tokens:
        raise ValueError("Expansion instructions and question exceed the prompt token budget")
    for snippet in example.get("snippets", []):
        trial = selected + [snippet]
        trial_prompt = _render(tokenizer, system_prompt, example, trial)
        if len(tokenizer(trial_prompt, add_special_tokens=True)["input_ids"]) > max_prompt_tokens:
            break
        selected = trial
        prompt = trial_prompt
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    return prompt, {
        "prompt_tokens": prompt_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "included_snippets": len(selected),
        "total_snippets": len(example.get("snippets", [])),
        "snippets_truncated": len(selected) < len(example.get("snippets", [])),
    }


def load_local_model(
    model_name: str = DEFAULT_MODEL,
    *,
    max_seq_length: int = 4096,
    local_files_only: bool = True,
):
    prime_unsloth_runtime()
    load_target = model_name
    if local_files_only and not Path(model_name).exists():
        from huggingface_hub import snapshot_download

        load_target = snapshot_download(model_name, local_files_only=True)
    args = SimpleNamespace(
        max_seq_length=max_seq_length,
        dtype=None,
        no_4bit=False,
        local_files_only=local_files_only,
        device_map="auto",
    )
    spec = ModelSpec(
        ref=model_name,
        label=model_name,
        source="local_base",
        load_target=str(load_target),
    )
    return load_model_and_tokenizer_for_eval(spec, args)


def _new_run_dir(output_parent: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(output_parent) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def run_local_expansion(
    examples: list[dict[str, Any]],
    output_parent: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    system_prompt: str = EXPANSION_PROMPT,
    max_seq_length: int = 4096,
    max_new_tokens: int = 512,
    local_files_only: bool = True,
    min_free_cuda_gb: float = 6.0,
    require_all_snippets: bool = False,
    response_mode: str = "equivalent",
    run: bool = False,
) -> dict[str, Any] | Path:
    """Generate once per question. Gold aliases are saved for scoring but never prompted."""
    if not examples or len({e["question_id"] for e in examples}) != len(examples):
        raise ValueError("Provide at least one example with unique question IDs")
    if max_new_tokens <= 0 or max_new_tokens >= max_seq_length:
        raise ValueError("max_new_tokens must be positive and smaller than max_seq_length")
    if response_mode not in {"equivalent", "extractive"}:
        raise ValueError("response_mode must be 'equivalent' or 'extractive'")
    config = {
        "model_name": model_name,
        "question_count": len(examples),
        "max_seq_length": max_seq_length,
        "max_new_tokens": max_new_tokens,
        "max_prompt_tokens": max_seq_length - max_new_tokens,
        "temperature": 0.0,
        "system_prompt": system_prompt,
        "gold_blind_generation": True,
        "local_files_only": local_files_only,
        "require_all_snippets": require_all_snippets,
        "response_mode": response_mode,
    }
    if not run:
        return config

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local 3B expansion experiment")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024 ** 3
    if free_gb < min_free_cuda_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB CUDA memory is free out of {total_bytes / 1024 ** 3:.2f} GiB. "
            "Wait for the active DPO run to finish before loading the 3B base model."
        )

    out = _new_run_dir(Path(output_parent))
    write_json(out / "config.json", config)
    write_jsonl(out / "examples.jsonl", examples)
    status: dict[str, Any] = {"status": "running", "completed_questions": 0}
    write_json(out / "status.json", status)
    generations: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    model = tokenizer = None
    try:
        model, tokenizer = load_local_model(
            model_name,
            max_seq_length=max_seq_length,
            local_files_only=local_files_only,
        )
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        device = first_model_device(model)
        for index, example in enumerate(examples, 1):
            prompt, prompt_info = prepare_prompt(
                tokenizer,
                example,
                system_prompt=system_prompt,
                max_prompt_tokens=max_seq_length - max_new_tokens,
            )
            if require_all_snippets and prompt_info["snippets_truncated"]:
                raise ValueError(
                    f"{example['question_id']}: all snippets do not fit the configured prompt budget"
                )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            if device is not None:
                encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, encoded["input_ids"].shape[-1]:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            try:
                answers = (
                    parse_extractive_response(raw, example["snippets"])
                    if response_mode == "extractive"
                    else parse_expansion_response(raw)
                )
                parse_error = None
            except ValueError as exc:
                answers = []
                parse_error = f"{type(exc).__name__}: {exc}"
            generations.append({
                "question_id": example["question_id"],
                "question": example["question"],
                "raw_response": raw,
                "parse_error": parse_error,
                **prompt_info,
            })
            for position, answer in enumerate(answers, 1):
                candidates.append({
                    "question_id": example["question_id"],
                    "position": position,
                    **answer,
                })
            status["completed_questions"] = index
            status["parse_failures"] = sum(bool(row["parse_error"]) for row in generations)
            write_jsonl(out / "generations.jsonl", generations)
            write_jsonl(out / "candidates.jsonl", candidates)
            write_json(out / "status.json", status)
            print(
                f"{index}/{len(examples)} | candidates: {len(answers)} | "
                f"parse failures: {status['parse_failures']}",
                flush=True,
            )
            del generated_ids, output_ids, encoded
            gc.collect()
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_jsonl(out / "generations.jsonl", generations)
        write_jsonl(out / "candidates.jsonl", candidates)
        write_json(out / "status.json", status)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def analyze_local_expansion(run_dir: str | Path, *, jar_path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"Run is not complete: {status}")
    examples = read_jsonl(run_dir / "examples.jsonl")
    generations = read_jsonl(run_dir / "generations.jsonl")
    candidates = read_jsonl(run_dir / "candidates.jsonl")
    by_id = {e["question_id"]: e for e in examples}
    by_candidates: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_candidates.setdefault(candidate["question_id"], []).append(candidate)
    report = run_dir / ("analysis-" + uuid.uuid4().hex[:8])
    report.mkdir()
    matches = official_candidate_matches(examples, candidates, report, jar_path=jar_path) if candidates else {}
    rows = []
    for generation in generations:
        qid = generation["question_id"]
        example = by_id[qid]
        group = sorted(by_candidates.get(qid, []), key=lambda row: row["position"])
        # Submission ranks are consecutive after filtering/deduplication. Raw
        # response positions can have gaps and are retained only for audit.
        hits = [rank for rank, row in enumerate(group, 1)
                if matches.get((qid, row["answer"]), False)]
        extractive = [candidate_is_extractive(row["answer"], example["snippets"]) for row in group]
        rows.append({
            "question_id": qid,
            "question": example["question"],
            "gold_aliases": example["gold_aliases"],
            "answers": [row["answer"] for row in group],
            "raw_positions": [row["position"] for row in group],
            "candidate_types": [row.get("candidate_type", row.get("relation_type")) for row in group],
            "snippet_ids": [row.get("snippet_id") for row in group],
            "matching_answers": [row["answer"] for row in group if matches.get((qid, row["answer"]), False)],
            "extractive_answers": [row["answer"] for row, flag in zip(group, extractive) if flag],
            "candidate_count": len(group),
            "parse_success": not bool(generation.get("parse_error")),
            "snippets_truncated": bool(generation.get("snippets_truncated")),
            **{f"coverage_at{k}": any(position <= k for position in hits) for k in (1, 5, 10)},
        })
    total = len(rows)
    candidate_type_counts = Counter(c.get("candidate_type", c.get("relation_type")) for c in candidates)
    summary = {
        "question_count": total,
        "parse_success_count": sum(r["parse_success"] for r in rows),
        "parse_success_rate": sum(r["parse_success"] for r in rows) / total,
        "questions_with_truncated_snippets": sum(r["snippets_truncated"] for r in rows),
        "mean_candidates": sum(r["candidate_count"] for r in rows) / total,
        "mean_unique_candidates": sum(len({a.casefold() for a in r["answers"]}) for r in rows) / total,
        "extractive_candidate_rate": (
            sum(len(r["extractive_answers"]) for r in rows) / len(candidates) if candidates else 0.0
        ),
        "candidate_type_counts": dict(candidate_type_counts),
        "response_mode": config.get("response_mode", "equivalent"),
        "model": config["model_name"],
        "scoring": "Official BioASQ Java matcher applied independently to every candidate",
    }
    for k in (1, 5, 10):
        summary[f"covered_at{k}"] = sum(r[f"coverage_at{k}"] for r in rows)
        summary[f"coverage_at{k}"] = summary[f"covered_at{k}"] / total
    write_json(report / "summary.json", summary)
    write_jsonl(report / "per_question.jsonl", rows)
    return report, summary, rows
