"""Resumable, training-only generation audit used by the Stage 1 audit notebook.

Textual snippet occurrence is an eligibility check, not a semantic judgment.
No generated pair is automatically declared a valid preference-training pair.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["question_id", "status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    temporary.replace(path)


def weak_normalize(value: str) -> str:
    # Preserve punctuation, hyphens and Greek characters.
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def flatten_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return list(dict.fromkeys(a for item in value for a in flatten_aliases(item)))
    return []


def contains_span(text: str, answer: str) -> bool:
    text, answer = weak_normalize(text), weak_normalize(answer)
    return bool(answer) and re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text) is not None


def marked_snippets(resources: list[str]) -> list[dict]:
    snippets = []
    for resource_index, resource in enumerate(resources, 1):
        for snippet_index, match in enumerate(re.finditer(r"\[BS\](.*?)\[ES\]", resource, re.S), 1):
            snippets.append({"resource_index": resource_index, "snippet_index": snippet_index,
                             "text": match.group(1).strip()})
    return snippets


def resolve_stage1(project_root: Path, stage_dir: Path, checkpoint_key: str = "best_mrr_adapter") -> dict:
    if checkpoint_key not in {"best_mrr_adapter", "best_selected_adapter"}:
        raise ValueError("Choose best_mrr_adapter or best_selected_adapter; no silent last-checkpoint fallback.")
    manifest_path = stage_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("stage") != "concept_learning":
        raise ValueError(f"Not a completed Stage 1 manifest: {manifest_path}")
    if not manifest.get(checkpoint_key):
        raise ValueError(f"Manifest has no {checkpoint_key}: {manifest_path}")
    config = dict(manifest["config"])
    config.update(manifest.get("stage_config", config.get("stage_settings", {}).get("concept_learning", {})))
    adapter = Path(manifest[checkpoint_key])
    if not adapter.is_absolute():
        adapter = project_root / adapter
    base = Path(config["base_model"])
    if not base.is_absolute():
        base = project_root / base
    for path in (adapter / "adapter_config.json", base / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    weights = sorted(adapter.glob("adapter_model*.safetensors")) + sorted(adapter.glob("adapter_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No adapter weights in {adapter}")
    return {"stage_dir": str(stage_dir.resolve()), "checkpoint_key": checkpoint_key,
            "adapter": str(adapter.resolve()), "base_model": str(base.resolve()), "config": config,
            "checkpoint_sha256": {p.name: file_digest(p) for p in [adapter / "adapter_config.json", *weights]},
            "best_dev_mrr": manifest.get("best_dev_mrr")}


def _question_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload["questions"]
    return {str(row["id"]) for row in rows}


def prepare_questions(project_root: Path, stage: dict, train_file: Path, gold_file: Path,
                      exclude_curriculum_holdout: bool = True) -> tuple[list[dict], list[dict], dict]:
    from src.utility.data import list_record_resources

    train = json.loads(train_file.read_text())
    if not isinstance(train, list):
        raise ValueError("train_file must be the prepared training JSON list.")
    gold = {str(q["id"]): q for q in json.loads(gold_file.read_text())["questions"]}
    cfg = stage["config"]
    dev_path = Path(cfg["dev_source_input"])
    if not dev_path.is_absolute():
        dev_path = project_root / dev_path
    protected_files = [dev_path, *sorted((project_root / "data/Task13BTest").glob("*_golden.json"))]
    protected = set().union(*(_question_ids(p) for p in protected_files))
    holdout = set()
    if exclude_curriculum_holdout:
        run_config_path = Path(stage["stage_dir"]).parent / "config.json"
        run_config = json.loads(run_config_path.read_text())
        if "global_eval_question_ids" not in run_config:
            raise ValueError("Run config lacks global_eval_question_ids. Set exclude_curriculum_holdout=False only for an audit that will not supply training data.")
        holdout = set(run_config["global_eval_question_ids"])
    eligible, excluded, seen = [], [], set()
    for row in train:
        if row.get("type") != "factoid":
            continue
        qid = str(row["id"])
        if qid in seen:
            raise ValueError(f"Duplicate training question ID: {qid}")
        seen.add(qid)
        raw = gold.get(qid)
        aliases = flatten_aliases(raw.get("exact_answer", [])) if raw else []
        resources = list_record_resources(row)
        snippets = marked_snippets(resources)
        evidence = [{"alias": a, **s} for a in aliases for s in snippets if contains_span(s["text"], a)]
        supported = list(dict.fromkeys(e["alias"] for e in evidence))
        reason = ("official_dev_or_test" if qid in protected else
                  "curriculum_heldout" if qid in holdout else
                  "missing_gold_question" if raw is None else
                  "no_gold_aliases" if not aliases else
                  "no_gold_alias_in_marked_snippet" if not supported else None)
        if reason:
            excluded.append({"question_id": qid, "question": row.get("input_1", ""), "reason": reason})
            continue
        eligible.append({"question_id": qid, "question": row["input_1"], "gold_aliases": aliases,
                         "supported_gold_aliases": supported, "gold_evidence": evidence,
                         "resources": resources, "snippets": snippets, "raw_question": raw,
                         "source_path": str(train_file.resolve())})
    metadata = {"training_factoid_questions": len(seen), "eligible_questions": len(eligible),
                "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
                "eligibility_rule": "Gold alias occurs within one marked snippet; lowercase/whitespace only, word boundaries; not semantic verification.",
                "exclude_curriculum_holdout": exclude_curriculum_holdout,
                "protected_files": {str(p.resolve()): file_digest(p) for p in protected_files},
                "curriculum_holdout_ids_sha256": digest(sorted(holdout)),
                "train_file_sha256": file_digest(train_file), "gold_file_sha256": file_digest(gold_file)}
    return eligible, excluded, metadata


def load_tokenizer_and_prompts(project_root: Path, stage: dict, questions: list[dict]):
    from transformers import AutoTokenizer
    from src.prompt_registry import resolve_prompt_bundle
    from src.utility.config import QUESTION_INSTRUCTIONS
    from src.utility.eval_dataset import render_prompt
    from src.utility.eval_types import EvalExample

    cfg = stage["config"]
    tokenizer_path = Path(stage["adapter"])
    if not (tokenizer_path / "tokenizer_config.json").exists():
        tokenizer_path = Path(cfg["initial_adapter"])
        if not tokenizer_path.is_absolute():
            tokenizer_path = project_root / tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    registry = Path(cfg["prompt_registry"])
    if not registry.is_absolute():
        registry = project_root / registry
    bundle = resolve_prompt_bundle(registry, cfg["prompt_ref"], QUESTION_INSTRUCTIONS)
    prepared = []
    for q in questions:
        example = EvalExample(q["question_id"], "factoid", q["question"], bundle["instructions"]["factoid"],
                              tuple(q["resources"]), "", q["source_path"], q["raw_question"])
        prompt = render_prompt(tokenizer, example, chat_template=bundle.get("chat_template", "qwen-2.5"), prompt_format="chat")
        prepared.append({**q, "prompt": prompt, "prompt_sha256": digest(prompt),
                         "prompt_tokens": len(tokenizer(prompt)["input_ids"])})
    return tokenizer, prepared, bundle


def initialize_audit(output_dir: Path, stage: dict, questions: list[dict], settings: dict, eligibility: dict) -> dict:
    # Limit/smoke settings are intentionally excluded: a small run can resume to all questions.
    import importlib.metadata
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {"audit_version": VERSION, "code_sha256": file_digest(Path(__file__)),
                "adapter": stage["adapter"], "base_model": stage["base_model"],
                "checkpoint_sha256": stage["checkpoint_sha256"], "settings": settings, "versions": versions,
                "base_config_sha256": file_digest(Path(stage["base_model"]) / "config.json"),
                "inputs_sha256": digest([(q["question_id"], q["prompt_sha256"], q["gold_aliases"]) for q in questions]),
                "eligibility": eligibility}
    manifest = {"fingerprint": digest(identity), **identity}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "audit_config.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("This audit directory belongs to different model/data/settings. Choose another OUTPUT_DIR.")
    elif any(output_dir.iterdir()):
        raise ValueError("Nonempty audit directory has no audit_config.json; choose another OUTPUT_DIR.")
    else:
        write_json(path, manifest)
    return manifest


def read_journal(path: Path) -> dict[str, dict]:
    """Recover a torn last write; refuse corruption earlier in the journal."""
    records = {}
    if not path.exists():
        return records
    with path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.truncate(offset)
                break
            row = json.loads(line)
            records[row["question_id"]] = row
    return records


def append_result(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def classify_prediction(prediction: str, question: dict) -> dict:
    from src.utility.bioasq_format import parse_prediction_items, normalize_for_bioasq_exact_match

    items = parse_prediction_items(prediction, "factoid")
    candidate = items[0] if items else ""
    aliases = question["gold_aliases"]
    exact = bool(candidate) and any(normalize_for_bioasq_exact_match(candidate) == normalize_for_bioasq_exact_match(a) for a in aliases)
    format_ok = re.fullmatch(r"\s*\[BE\](?:(?!\[BE\]|\[EE\]).)+\[EE\]\s*", prediction, re.S) is not None
    matches = [s for s in question["snippets"] if contains_span(s["text"], candidate)]
    relations = []
    if not exact and candidate:
        for alias in question["supported_gold_aliases"]:
            if contains_span(alias, candidate):
                relations.append({"alias": alias, "relation": "too_short_candidate"})
            elif contains_span(candidate, alias):
                relations.append({"alias": alias, "relation": "too_long_candidate"})
    directions = {r["relation"] for r in relations}
    if not candidate or not format_ok or len(items) != 1:
        category = "format_violation"
    elif exact:
        category = "correct"
    elif len(directions) > 1:
        category = "ambiguous_boundary_candidate"
    elif directions:
        category = next(iter(directions))
    elif matches:
        category = "other_extractive_mismatch_review"
    else:
        category = "non_extractive_mismatch_review"
    return {"prediction": prediction, "candidate": candidate, "parsed_answer_count": len(items),
            "format_ok": format_ok, "top1_alias_match_audit": exact, "category": category,
            "candidate_extractive": bool(matches), "candidate_evidence": matches,
            "boundary_relations": relations,
            "needs_semantic_review": not (exact and format_ok and len(items) == 1)}


def export_audit(output_dir: Path, questions: list[dict], records: dict[str, dict]) -> dict:
    rows = []
    for q in questions:
        result = records.get(q["question_id"], {"status": "pending"})
        rows.append({"question_id": q["question_id"], "question": q["question"], **result,
                     "gold_aliases": q["gold_aliases"], "supported_gold_aliases": q["supported_gold_aliases"],
                     "gold_evidence": q["gold_evidence"], "snippets": q["snippets"],
                     "prompt_tokens": q["prompt_tokens"]})
    write_csv(output_dir / "all_questions_audit.csv", rows)
    boundary = [r for r in rows if r.get("boundary_relations")]
    write_csv(output_dir / "boundary_candidates.csv", boundary)
    write_csv(output_dir / "skipped_or_failed.csv", [r for r in rows if r["status"] not in {"complete", "pending"}])
    # Preserve manual annotations on reruns, and append newly completed questions.
    review_path = output_dir / "manual_review.csv"
    previous = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            previous = {r["question_id"]: r for r in csv.DictReader(handle)}
    review = []
    for row in rows:
        if row.get("needs_semantic_review"):
            old = previous.get(row["question_id"], {})
            review.append({**row, **{key: old.get(key, "") for key in
                           ("review_label", "preferred_alias", "approved_for_training", "review_notes")}})
    write_csv(review_path, review)
    summary = {"eligible_questions": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
               "category_counts": dict(Counter(r["category"] for r in rows if r.get("category"))),
               "top1_alias_matches_audit": sum(bool(r.get("top1_alias_match_audit")) for r in rows),
               "boundary_candidates": len(boundary), "needs_review": len(review),
               "note": "Audit labels are surface heuristics, not semantic C1/C2 judgments or official metrics. No pairs are automatically approved."}
    write_json(output_dir / "audit_summary.json", summary)
    return summary


def load_stage1_model(stage: dict, load_in_4bit: bool = True):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA notebook kernel for generation. Preparation and CSV inspection work without a GPU.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(local_files_only=True, device_map={"": torch.cuda.current_device()},
                  torch_dtype=dtype, attn_implementation=stage["config"].get("attn_implementation", "sdpa"))
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(stage["base_model"], **kwargs)
    if load_in_4bit:
        # Match the trainer's upcasting of non-quantized layers, without training hooks.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, stage["adapter"], is_trainable=False, local_files_only=True)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.eval()
    return model


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    import torch
    from src.utility.data import clean_text

    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    encoded = {k: v.to(next(model.parameters()).device) for k, v in encoded.items()}
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                num_return_sequences=1, use_cache=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = output[0, encoded["input_ids"].shape[-1]:]
    text = clean_text(tokenizer.decode(generated, skip_special_tokens=True))
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    return {"raw_output": text, "generated_tokens": len(generated),
            "hit_generation_limit": len(generated) >= max_new_tokens and int(generated[-1]) != tokenizer.eos_token_id}


def run_audit(model, tokenizer, questions: list[dict], output_dir: Path, *, max_prompt_tokens: int,
              max_new_tokens: int, limit_questions: int | None = None, retry_failed: bool = True) -> dict:
    import torch
    from tqdm.auto import tqdm

    if max_prompt_tokens < 0 or max_new_tokens <= 0 or (limit_questions is not None and limit_questions <= 0):
        raise ValueError("Use max_prompt_tokens >= 0, max_new_tokens > 0, and a positive question limit or None.")
    journal = output_dir / "generations.jsonl"
    records = read_journal(journal)
    selected = questions if limit_questions is None else questions[:limit_questions]
    model_limit = getattr(model.config, "max_position_embeddings", None)
    try:
        for q in tqdm(selected, desc="Stage 1 greedy train audit"):
            qid = q["question_id"]
            old = records.get(qid)
            if old and (old["status"] != "oom" or not retry_failed):
                continue
            result = {"question_id": qid, "prompt_sha256": q["prompt_sha256"]}
            if max_prompt_tokens and q["prompt_tokens"] > max_prompt_tokens:
                result.update(status="skipped_context_budget", reason="Full prompt exceeds audit budget; no truncation.")
            elif model_limit and q["prompt_tokens"] + max_new_tokens > model_limit:
                result.update(status="skipped_model_context", reason="Prompt plus generation allowance exceeds model context.")
            else:
                try:
                    generated = generate_one(model, tokenizer, q["prompt"], max_new_tokens)
                    result.update(status="complete", **generated, **classify_prediction(generated["raw_output"], q))
                    if generated["hit_generation_limit"]:
                        result["needs_semantic_review"] = True
                except torch.OutOfMemoryError as exc:
                    result.update(status="oom", reason=str(exc)[:1500])
                # Outside the exception handler: release the failed forward's traceback first.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            append_result(journal, result)
            records[qid] = result
            if len(records) % 25 == 0:
                export_audit(output_dir, questions, records)
    finally:
        summary = export_audit(output_dir, questions, records)
    return summary


def score_completed_questions(project_root: Path, output_dir: Path, questions: list[dict]) -> dict:
    """Use the project's official Java evaluator; score completed generations only."""
    from argparse import Namespace
    from src.utility.bioasq_official import evaluate_with_bioasq_java
    from src.utility.eval_types import EvalExample

    records = read_journal(output_dir / "generations.jsonl")
    prediction_rows, examples = [], {}
    for q in questions:
        result = records.get(q["question_id"], {})
        if result.get("status") != "complete":
            continue
        prediction_rows.append({"question_id": q["question_id"], "question_type": "factoid",
                                "body": q["question"], "prediction": result["prediction"]})
        # Older training questions store flat alias lists; Java v5 expects nested lists.
        # For factoids these are alternative aliases for the single answer.
        scorer_gold = {**q["raw_question"], "exact_answer": [q["gold_aliases"]]}
        examples[(q["question_id"], "factoid")] = EvalExample(
            q["question_id"], "factoid", q["question"], "", tuple(q["resources"]), "",
            q["source_path"], scorer_gold)
    if not prediction_rows:
        raise ValueError("No completed generations to score.")
    return evaluate_with_bioasq_java(
        prediction_rows=prediction_rows, examples_by_key=examples, model_label="stage1-training-audit",
        model_dir=output_dir, include_per_question=True,
        args=Namespace(bioasq_java_jar=str(project_root / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"),
                       bioasq_java_heap="512m", bioasq_java_version=5))
