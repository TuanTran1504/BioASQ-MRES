#!/usr/bin/env python3
"""Audited whole-response factoid expansion preferences; no CUDA imports."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_matched_8b_sft import ROOT, digest, read, records, validate, write
from run_extractive_expansion_8b import VALID_RELATION_TYPES, write_jsonl

DEFAULT_CONFIG = ROOT / "configs/expansion_dpo_8b.json"
MODELS = ("llama31", "qwen3", "ministral3")


def strict_answers(raw):
    """Primary semantic pairs require complete, unrepaired, duplicate-free JSON."""
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"answers"}:
        raise ValueError("Expected only the answers field")
    answers = value["answers"]
    if not isinstance(answers, list) or not 1 <= len(answers) <= 10:
        raise ValueError("Expected one to ten candidates")
    seen = set()
    for i, row in enumerate(answers):
        if not isinstance(row, dict) or set(row) != {"answer", "relation_type"}:
            raise ValueError("Unexpected candidate schema")
        if not isinstance(row["answer"], str) or not row["answer"].strip():
            raise ValueError("Empty candidate")
        if row["relation_type"] not in VALID_RELATION_TYPES:
            raise ValueError("Unknown relation type")
        if (row["relation_type"] == "original") != (i == 0):
            raise ValueError("Exactly the first candidate must be original")
        key = re.sub(r"\s+", " ", row["answer"].casefold()).strip()
        if key in seen:
            raise ValueError("Duplicate candidate")
        seen.add(key)
    return answers


def response_id(model, qid, draw, raw):
    return hashlib.sha256(json.dumps([model, qid, draw, raw], ensure_ascii=False).encode()).hexdigest()


def split_rows(config, full=False):
    evaluation = read(ROOT / config["evaluation_config"])
    matrix = read(ROOT / evaluation["training_matrix"])
    validate(matrix)
    source = ROOT / matrix["output_dataset"] / "expansion"
    result = {}
    for split in ("train", "validation"):
        rows = sorted(records(source / (split + ".jsonl")), key=lambda r: r["question_id"])
        random.Random(config["seed"]).shuffle(rows)
        result[split] = rows if full else rows[:config["pilot_questions"][split]]
    return result


def validate_bank(bank, config):
    """Require the exact same completed question panel and source files across models."""
    manifest = read(bank / "manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("smoke_test")
            or manifest.get("dpo_config_sha256") != digest(DEFAULT_CONFIG)):
        raise ValueError("Expected a complete non-smoke bank with the current DPO config")
    source = split_rows(config, manifest["full_dataset"])
    expected = {(s, r["question_id"]): r["messages"][:2]
                for s, rows in source.items() for r in rows}
    if manifest["question_ids"] != {s: [r["question_id"] for r in rows] for s, rows in source.items()}:
        raise ValueError("Bank question panel differs from the reproducible partition")
    values = records(bank / "responses.jsonl")
    if digest(bank / "responses.jsonl") != manifest["responses_sha256"]:
        raise ValueError("Response bank hash differs")
    keys = set()
    for row in values:
        key = (row["split"], row["question_id"])
        unique = (*key, row["draw"])
        if (key not in expected or row["prompt"] != expected[key]
                or row["model_key"] != manifest["model_key"] or unique in keys
                or row["draw"] not in range(config["sampled_responses"] + 1)
                or row["response_id"] != response_id(row["model_key"], row["question_id"], row["draw"], row["raw_response"])):
            raise ValueError("Bank contains changed, duplicate or out-of-partition responses")
        keys.add(unique)
    wanted = {(*key, draw) for key in expected for draw in range(config["sampled_responses"] + 1)}
    if keys != wanted:
        raise ValueError("Missing response draws")
    if manifest["response_count"] != len(values):
        raise ValueError("Response count differs from bank manifest")
    return manifest, values


def annotation_template(row):
    result = {**row, "reviewer": "", "review_notes": "", "candidates": []}
    try:
        answers = strict_answers(row["raw_response"])
    except (ValueError, TypeError):
        result["excluded_reason"] = "invalid primary response schema"
        return result
    result["excluded_reason"] = ""
    result["candidates"] = [{**r, "official_accepted": None, "class": None, "supported": None,
                              "equivalent_to_original": None, "evidence": ""}
                             for r in answers]
    return result


def quality(row):
    if row.get("excluded_reason"):
        return None
    if not row.get("reviewer", "").strip() or not row.get("review_notes", "").strip():
        return None
    answers = strict_answers(row["raw_response"])
    labels = row["candidates"]
    if len(labels) != len(answers):
        raise ValueError("Annotation count differs from the unmodified response")
    counts = Counter()
    for i, (answer, label) in enumerate(zip(answers, labels)):
        if any(label.get(k) != v for k, v in answer.items()):
            raise ValueError("Annotated candidate differs from raw response")
        if label.get("class") not in ("C1", "C2", "C3") or any(
            type(label.get(k)) is not bool for k in ("official_accepted", "supported", "equivalent_to_original")):
            return None  # unresolved biomedical labels never become negative examples
        if (label["class"] == "C3") != label["official_accepted"]:
            raise ValueError("C3 must agree with the official candidate match; unmatched does not mean C1")
        if not isinstance(label.get("evidence"), str) or not label["evidence"].strip():
            return None
        if i == 0 and not label["equivalent_to_original"]:
            raise ValueError("Original must be equivalent to itself")
        counts["accepted"] += label["class"] == "C3"
        counts["correct"] += label["class"] in ("C2", "C3")
        counts["incorrect"] += label["class"] == "C1"
        counts["unsupported"] += not label["supported"]
        counts["non_equivalent"] += not label["equivalent_to_original"]
    counts["original_correct"] = int(labels[0]["class"] in ("C2", "C3"))
    return dict(counts)


def preferred(a, b):
    """Conservative Pareto dominance; C2 is correct and length earns no reward."""
    if not a["original_correct"] or any(a[k] for k in ("incorrect", "unsupported", "non_equivalent")):
        return False
    larger = ("accepted", "correct", "original_correct")
    smaller = ("incorrect", "unsupported", "non_equivalent")
    return (all(a[k] >= b[k] for k in larger) and all(a[k] <= b[k] for k in smaller)
            and (any(a[k] > b[k] for k in larger) or any(a[k] < b[k] for k in smaller)))


def build_pairs(rows, cap=2):
    groups, exclusions = defaultdict(list), Counter()
    for row in rows:
        q = quality(row)
        if q is None:
            exclusions["excluded_or_unresolved_responses"] += 1
            continue
        groups[(row["split"], row["question_id"])].append((row, q))
    pairs = {"train": [], "validation": []}
    for (split, qid), group in sorted(groups.items()):
        candidates, seen = [], set()
        for (a, aq), (b, bq) in itertools.combinations(sorted(group, key=lambda x: x[0]["response_id"]), 2):
            if preferred(bq, aq):
                a, b, aq, bq = b, a, bq, aq
            if not preferred(aq, bq):
                continue
            key = (json.dumps(json.loads(a["raw_response"]), sort_keys=True),
                   json.dumps(json.loads(b["raw_response"]), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            if a["prompt"] != b["prompt"]:
                raise ValueError("Pair prompts differ")
            candidates.append({"question_id": qid, "prompt": a["prompt"],
                               "chosen": a["raw_response"], "rejected": b["raw_response"],
                               "chosen_id": a["response_id"], "rejected_id": b["response_id"],
                               "chosen_model": a["model_key"], "rejected_model": b["model_key"],
                               "chosen_quality": aq, "rejected_quality": bq})
        # Deterministic cap keeps questions with many responses from dominating.
        pairs[split].extend(candidates[:cap])
    return pairs, dict(exclusions)


def validate_pairs(directory, config):
    manifest = read(directory / "manifest.json")
    if manifest.get("status") != "ready" or manifest["dpo_config_sha256"] != digest(DEFAULT_CONFIG):
        raise ValueError("Preferences are not ready for this DPO configuration")
    banks = manifest["banks"]
    if set(banks) != set(MODELS):
        raise ValueError("A shared bank from all three models is required")
    original = {}
    for model, bank in banks.items():
        saved, rows = validate_bank(Path(bank["path"]), config)
        if saved != bank["manifest"] or saved["model_key"] != model:
            raise ValueError("Bank provenance changed")
        original.update({r["response_id"]: r for r in rows})
    annotations = records(directory / "annotations.jsonl")
    if digest(directory / "annotations.jsonl") != manifest["annotations_sha256"]:
        raise ValueError("Preference annotations changed")
    if len(annotations) != len(original) or {r["response_id"] for r in annotations} != set(original):
        raise ValueError("Annotations must retain every bank response exactly once")
    for row in annotations:
        raw = original[row["response_id"]]
        if any(row.get(k) != v for k, v in raw.items()):
            raise ValueError("Annotation changed generation data")
    recomputed, _ = build_pairs(annotations, config["max_pairs_per_question"])
    for split in ("train", "validation"):
        path = directory / (split + ".jsonl")
        if digest(path) != manifest["output_sha256"][split] or records(path) != recomputed[split]:
            raise ValueError("Preference pairs differ from audited annotation decisions")
        if not recomputed[split]:
            raise ValueError(f"No reliable {split} pairs; inspect pair yield before training")
    return manifest, recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("annotate", "pairs", "validate"), required=True)
    parser.add_argument("--banks", nargs=3, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(DEFAULT_CONFIG)
    if args.mode == "validate":
        manifest, _ = validate_pairs(args.output.resolve(), config)
        print(json.dumps(manifest["pair_counts"], indent=2))
        return
    if not args.banks:
        parser.error("--banks requires one completed bank from each model")
    banks, source = {}, {}
    for path in args.banks:
        saved, rows = validate_bank(path.resolve(), config)
        model = saved["model_key"]
        if model in banks or model not in MODELS:
            raise ValueError("Expected three distinct model banks")
        banks[model] = {"path": str(path.resolve()), "manifest": saved}
        source.update({r["response_id"]: r for r in rows})
    panels = {json.dumps(b["manifest"]["question_ids"], sort_keys=True) for b in banks.values()}
    if len(panels) != 1:
        raise ValueError("Model banks must use the same question panel")
    if args.mode == "annotate":
        if args.output.exists():
            raise FileExistsError(args.output)
        write_jsonl(args.output, [annotation_template(r) for r in source.values()])
        print(f"Review template: {args.output}; no biomedical labels inferred from exact mismatch")
        return
    if not args.annotations:
        parser.error("--annotations is required for pair construction")
    annotations = records(args.annotations)
    ids = [r["response_id"] for r in annotations]
    if len(set(ids)) != len(ids) or set(ids) != set(source):
        raise ValueError("Review must retain every bank response exactly once; exclude explicitly")
    for row in annotations:
        if any(row.get(k) != v for k, v in source[row["response_id"]].items()):
            raise ValueError("Annotation modified a saved generation")
    pairs, exclusions = build_pairs(annotations, config["max_pairs_per_question"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "annotations.jsonl", annotations)
    for split, values in pairs.items():
        write_jsonl(args.output / (split + ".jsonl"), values)
    write(args.output / "manifest.json", {
        "status": "ready" if all(pairs.values()) else "insufficient_pairs",
        "banks": banks, "dpo_config_sha256": digest(DEFAULT_CONFIG),
        "annotations_sha256": digest(args.output / "annotations.jsonl"),
        "pair_counts": {s: len(v) for s, v in pairs.items()}, "exclusions": exclusions,
        "output_sha256": {s: digest(args.output / (s + ".jsonl")) for s in pairs},
        "preference_rule": "supported correct concept; Pareto accepted/correct coverage and error removal; no length reward",
        "label_source": "explicit reviewed C1/C2/C3, evidence and equivalence annotations",
        "selection": "internal validation preference loss; no outer development training labels"})
    print(json.dumps({"pair_counts": {s: len(v) for s, v in pairs.items()}, "exclusions": exclusions}, indent=2))


if __name__ == "__main__":
    main()
