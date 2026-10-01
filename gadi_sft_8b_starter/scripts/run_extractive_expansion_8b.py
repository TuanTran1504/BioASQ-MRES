#!/usr/bin/env python3
"""Run a local-model expansion experiment on one Gadi GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/extractive_expansion_8b.json"
VALID_CANDIDATE_TYPES = {
    "minimal_direct",
    "qualified_direct",
    "canonical_surface",
    "abbreviation_surface",
    "numeric_surface",
    "alternative_evidence",
}
VALID_RELATION_TYPES = {
    "original",
    "synonym",
    "abbreviation_expansion",
    "nomenclature_variant",
    "spelling_or_inflection",
    "numerically_equivalent",
    "harmless_formatting",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "run"


def json_objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_extractive_response(
    text: str,
    snippets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], bool]:
    """Salvage literal spans while separately reporting schema compliance."""
    value = next((obj for obj in json_objects(text) if isinstance(obj.get("answers"), list)), None)
    issues: list[str] = []
    if value is None:
        required = {"answer", "snippet_id", "candidate_type"}
        raw_answers = [obj for obj in json_objects(text) if required <= set(obj)]
        if not raw_answers:
            raise ValueError("No JSON answer candidates were found")
        issues.append("incomplete_top_level_json_recovered")
    else:
        if set(value) != {"answers"}:
            issues.append("unexpected_top_level_fields")
        raw_answers = value["answers"]
        if not 1 <= len(raw_answers) <= 10:
            issues.append("answer_count_out_of_range")

    snippet_by_id = {
        str(row["snippet_id"]): str(row.get("text", ""))
        for row in snippets
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_position, row in enumerate(raw_answers, 1):
        if not isinstance(row, dict):
            rejected.append({
                "raw_position": raw_position,
                "reason": "candidate_is_not_an_object",
                "raw_candidate": row,
            })
            issues.append(f"candidate_{raw_position}_not_object")
            continue
        required = {"answer", "snippet_id", "candidate_type"}
        if set(row) != required:
            issues.append(f"candidate_{raw_position}_unexpected_fields")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer:
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "missing_or_empty_answer",
            })
            continue
        candidate_type = str(row.get("candidate_type") or "unrecognized")
        if candidate_type not in VALID_CANDIDATE_TYPES:
            issues.append(f"candidate_{raw_position}_invalid_candidate_type")
        reported_id = str(row.get("snippet_id") or "")
        matching_ids = [
            snippet_id
            for snippet_id, snippet_text in snippet_by_id.items()
            if answer in snippet_text
        ]
        if not matching_ids:
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "answer_not_literal_in_any_supplied_snippet",
            })
            continue
        key = re.sub(r"\s+", " ", answer.casefold()).strip()
        if key in seen:
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "duplicate_answer_surface",
            })
            continue
        seen.add(key)
        if len(accepted) >= 10:
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "unique_candidate_limit_exceeded",
            })
            continue
        actual_id = reported_id if reported_id in matching_ids else matching_ids[0]
        if actual_id != reported_id:
            issues.append(f"candidate_{raw_position}_citation_corrected")
        accepted.append({
            "answer": answer,
            "snippet_id": actual_id,
            "candidate_type": candidate_type,
            "raw_position": raw_position,
            "reported_snippet_id": reported_id,
            "citation_corrected": actual_id != reported_id,
        })
    structural_issues = [
        issue for issue in issues
        if "citation_corrected" not in issue
    ]
    return accepted, rejected, issues, not structural_issues


def parse_equivalent_response(
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], bool]:
    """Recover and deduplicate equivalent-expression candidates without using gold."""
    value = next((obj for obj in json_objects(text) if isinstance(obj.get("answers"), list)), None)
    issues: list[str] = []
    if value is None:
        required = {"answer", "relation_type"}
        raw_answers = [obj for obj in json_objects(text) if required <= set(obj)]
        if not raw_answers:
            raise ValueError("No JSON answer candidates were found")
        issues.append("incomplete_top_level_json_recovered")
    else:
        if set(value) != {"answers"}:
            issues.append("unexpected_top_level_fields")
        raw_answers = value["answers"]
        if not 1 <= len(raw_answers) <= 10:
            issues.append("answer_count_out_of_range")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_position, row in enumerate(raw_answers, 1):
        if not isinstance(row, dict):
            rejected.append({
                "raw_position": raw_position,
                "reason": "candidate_is_not_an_object",
                "raw_candidate": row,
            })
            issues.append(f"candidate_{raw_position}_not_object")
            continue
        required = {"answer", "relation_type"}
        if set(row) != required:
            issues.append(f"candidate_{raw_position}_unexpected_fields")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "missing_or_empty_answer",
            })
            continue
        relation_type = str(row.get("relation_type") or "unrecognized")
        if relation_type not in VALID_RELATION_TYPES:
            issues.append(f"candidate_{raw_position}_invalid_relation_type")
        if raw_position == 1 and relation_type != "original":
            issues.append("first_candidate_not_original")
        if raw_position > 1 and relation_type == "original":
            issues.append(f"candidate_{raw_position}_unexpected_original")
        key = re.sub(r"\s+", " ", answer.casefold()).strip()
        if key in seen:
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "duplicate_answer_surface",
            })
            continue
        seen.add(key)
        if len(accepted) >= 10:
            rejected.append({
                **row,
                "raw_position": raw_position,
                "reason": "unique_candidate_limit_exceeded",
            })
            continue
        accepted.append({
            "answer": answer,
            "relation_type": relation_type,
            "raw_position": raw_position,
        })
    return accepted, rejected, issues, not issues


def render_prompt(
    tokenizer: Any,
    system_prompt: str,
    example: dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    user = {
        "question": example["question"],
        "snippets": [
            {"id": row["snippet_id"], "text": row["text"]}
            for row in example["snippets"]
        ],
    }
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )


def tokenize_text(tokenizer: Any, prompt: str, **kwargs: Any) -> Any:
    """Tokenize text without letting multimodal processors treat it as an image."""
    return tokenizer(text=prompt, **kwargs)


def load_model(
    model_name: str,
    max_seq_length: int,
    allow_download: bool,
    model_loader: str = "fast_language_model",
):
    from src.utility.eval_models import load_model_and_tokenizer_for_eval, prime_unsloth_runtime
    from src.utility.eval_types import ModelSpec

    prime_unsloth_runtime()
    load_target = model_name
    if not allow_download and not Path(model_name).exists():
        from huggingface_hub import snapshot_download

        load_target = snapshot_download(model_name, local_files_only=True)
    if model_loader not in {"fast_language_model", "fast_model"}:
        raise ValueError("model_loader must be 'fast_language_model' or 'fast_model'")

    args = SimpleNamespace(
        max_seq_length=max_seq_length,
        dtype=None,
        no_4bit=False,
        local_files_only=not allow_download,
        device_map="auto",
    )
    spec = ModelSpec(
        ref=model_name,
        label=model_name,
        source="gadi_base",
        load_target=str(load_target),
    )
    if model_loader == "fast_model":
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=str(load_target),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            local_files_only=not allow_download,
        )
        class_handler = getattr(FastModel, "for_inference", None)
        if callable(class_handler):
            class_handler(model)
        elif hasattr(model, "for_inference"):
            model.for_inference()
        if (
            getattr(tokenizer, "pad_token_id", None) is None
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token
        return model.eval(), tokenizer
    return load_model_and_tokenizer_for_eval(spec, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-parent", type=Path, default=ROOT / "outputs/expansion")
    parser.add_argument("--model-name", default=os.environ.get("BIOASQ_MODEL_NAME"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume-run", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--min-free-cuda-gb", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_template = read_json(args.config)
    response_mode = str(config_template.get("response_mode", "extractive"))
    if response_mode not in {"extractive", "equivalent"}:
        raise ValueError("response_mode must be 'extractive' or 'equivalent'")
    model_loader = str(config_template.get("model_loader", "fast_language_model"))
    if model_loader not in {"fast_language_model", "fast_model"}:
        raise ValueError("model_loader must be 'fast_language_model' or 'fast_model'")
    chat_template_kwargs = config_template.get("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    input_path = ROOT / config_template["input"]
    prompt_path = ROOT / config_template["prompt"]
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing expansion input: {input_path}. Run scripts/export_gadi_expansion_dev.py "
            "in the full repository before transferring this bundle."
        )
    examples = read_jsonl(input_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        examples = examples[:args.limit]
    if len({row["question_id"] for row in examples}) != len(examples):
        raise ValueError("Expansion examples contain duplicate question IDs")

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    model_name = args.model_name or config_template["model_name"]
    resolved_config = {
        **config_template,
        "model_name": model_name,
        "question_count": len(examples),
        "input_sha256": file_sha256(input_path),
        "prompt_sha256": file_sha256(prompt_path),
        "gold_blind_generation": True,
        "local_files_only": not args.allow_download,
        "response_mode": response_mode,
    }

    if args.resume_run:
        run_dir = args.resume_run.resolve()
        if read_json(run_dir / "config.json") != resolved_config:
            raise ValueError("Resume configuration does not match the saved run")
        saved_examples = read_jsonl(run_dir / "examples.jsonl")
        if saved_examples != examples:
            raise ValueError("Resume examples do not match the saved run")
        generations = read_jsonl(run_dir / "generations.jsonl")
        candidates = read_jsonl(run_dir / "candidates.jsonl")
        invalid_candidates = read_jsonl(run_dir / "invalid_candidates.jsonl")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        label = slug(args.run_name or f"llama31-8b-extractive-{timestamp}")
        run_dir = (args.output_parent / f"{timestamp}-{label}").resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        generations, candidates, invalid_candidates = [], [], []
        write_json(run_dir / "config.json", resolved_config)
        write_jsonl(run_dir / "examples.jsonl", examples)

    completed_ids = {row["question_id"] for row in generations}
    status = {
        "status": "running",
        "expected_questions": len(examples),
        "completed_questions": len(completed_ids),
    }
    write_json(run_dir / "status.json", status)
    print("Run directory:", run_dir, flush=True)
    print("Completed before this process:", len(completed_ids), flush=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this script through a Gadi GPU queue")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024 ** 3
    if free_gb < args.min_free_cuda_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB CUDA memory is free out of "
            f"{total_bytes / 1024 ** 3:.2f} GiB"
        )

    model = tokenizer = None
    active_question = None
    try:
        model, tokenizer = load_model(
            model_name,
            max_seq_length=int(config_template["max_seq_length"]),
            allow_download=args.allow_download,
            model_loader=model_loader,
        )
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        try:
            device = model.device
        except Exception:
            device = next(model.parameters()).device

        max_prompt_tokens = int(config_template["max_seq_length"]) - int(config_template["max_new_tokens"])
        prepared_prompts: dict[str, tuple[str, int]] = {}
        for example in examples:
            qid = example["question_id"]
            prompt = render_prompt(
                tokenizer,
                system_prompt,
                example,
                chat_template_kwargs=chat_template_kwargs,
            )
            prompt_tokens = len(
                tokenize_text(tokenizer, prompt, add_special_tokens=True)["input_ids"]
            )
            if prompt_tokens > max_prompt_tokens:
                raise ValueError(
                    f"{qid}: all snippets require {prompt_tokens} prompt tokens, exceeding "
                    f"the {max_prompt_tokens}-token budget"
                )
            prepared_prompts[qid] = (prompt, prompt_tokens)
        print(
            "Prompt preflight: all snippets fit; maximum prompt tokens:",
            max(tokens for _, tokens in prepared_prompts.values()),
            flush=True,
        )

        for example in examples:
            qid = example["question_id"]
            if qid in completed_ids:
                continue
            active_question = qid
            prompt, prompt_tokens = prepared_prompts[qid]
            encoded = tokenize_text(
                tokenizer,
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=int(config_template["max_new_tokens"]),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, encoded["input_ids"].shape[-1]:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            parse_error = None
            try:
                if response_mode == "extractive":
                    accepted, rejected, issues, schema_compliant = parse_extractive_response(
                        raw, example["snippets"]
                    )
                else:
                    accepted, rejected, issues, schema_compliant = parse_equivalent_response(raw)
            except ValueError as exc:
                accepted, rejected, issues, schema_compliant = [], [], [], False
                parse_error = f"{type(exc).__name__}: {exc}"

            generation = {
                "question_id": qid,
                "question": example["question"],
                "raw_response": raw,
                "parse_error": parse_error,
                "schema_compliant": schema_compliant,
                "response_recovered": "incomplete_top_level_json_recovered" in issues,
                "validation_issues": issues,
                "invalid_candidate_count": len(rejected),
                "prompt_tokens": prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "included_snippets": len(example["snippets"]),
                "total_snippets": len(example["snippets"]),
                "snippets_truncated": False,
            }
            generations.append(generation)
            candidates.extend({
                "question_id": qid,
                "position": row["raw_position"],
                **row,
            } for row in accepted)
            invalid_candidates.extend({
                "question_id": qid,
                **row,
            } for row in rejected)
            completed_ids.add(qid)
            status.update(
                completed_questions=len(completed_ids),
                parse_failures=sum(bool(row.get("parse_error")) for row in generations),
                schema_compliant_responses=sum(bool(row.get("schema_compliant")) for row in generations),
                recovered_responses=sum(bool(row.get("response_recovered")) for row in generations),
                accepted_candidates=len(candidates),
                invalid_candidates=len(invalid_candidates),
                corrected_citations=sum(bool(row.get("citation_corrected")) for row in candidates),
            )
            write_jsonl(run_dir / "generations.jsonl", generations)
            write_jsonl(run_dir / "candidates.jsonl", candidates)
            write_jsonl(run_dir / "invalid_candidates.jsonl", invalid_candidates)
            write_json(run_dir / "status.json", status)
            print(
                f"{len(completed_ids)}/{len(examples)} | accepted candidates: {len(accepted)} | "
                f"rejected: {len(rejected)} | parse failures: {status['parse_failures']}",
                flush=True,
            )
            del generated_ids, output_ids, encoded
            gc.collect()
        status["status"] = "complete"
    except BaseException as exc:
        status.update(
            status="incomplete",
            error=f"{type(exc).__name__}: {exc}",
            failed_question_id=active_question,
        )
        raise
    finally:
        write_jsonl(run_dir / "generations.jsonl", generations)
        write_jsonl(run_dir / "candidates.jsonl", candidates)
        write_jsonl(run_dir / "invalid_candidates.jsonl", invalid_candidates)
        write_json(run_dir / "status.json", status)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
