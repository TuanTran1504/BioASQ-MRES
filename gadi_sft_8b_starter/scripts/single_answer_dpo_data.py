#!/usr/bin/env python3
"""Original-SFT single-answer banks and independently audited C3>C1/C2 preferences."""
import argparse
from collections import defaultdict, Counter
import itertools
import json
from pathlib import Path
import sys

from expansion_dpo_data import MODELS, response_id, split_rows
from prepare_matched_8b_sft import ROOT, digest, read, records, write, load_source
from run_single_answer_sampling import parse_single_answer, sample_seed
from run_extractive_expansion_8b import write_jsonl

DEFAULT_CONFIG = ROOT / "configs/original_sft_dpo_8b.json"
STRATEGIES = ("concept", "form", "mixed")


def validate_responses(manifest, rows, config):
    if (manifest.get("status") != "complete" or manifest.get("smoke_test") or
            manifest.get("formulation") != "original" or manifest.get("dpo_config_sha256") != digest(DEFAULT_CONFIG)):
        raise ValueError("Expected a complete original-SFT bank with the pinned configuration")
    panels = split_rows(config, manifest["full_dataset"])
    prompts = {(s, r["question_id"]): r["messages"][:2] for s, values in panels.items() for r in values}
    if manifest["question_ids"] != {s: [r["question_id"] for r in values] for s, values in panels.items()}:
        raise ValueError("Bank question partition changed")
    seen = set()
    for row in rows:
        key = (row["split"], row["question_id"], row["draw"])
        if (key in seen or key[:2] not in prompts or row["prompt"] != prompts[key[:2]] or
                row["model_key"] != manifest["model_key"] or
                row["response_id"] != response_id(row["model_key"], row["question_id"], row["draw"], row["raw_response"]) or
                row["seed"] != sample_seed(row["question_id"], row["draw"] + 1, config["seed"]) or
                row["decoding"] != ("sampling" if row["draw"] else "greedy")):
            raise ValueError("Changed, duplicate or out-of-partition bank response")
        seen.add(key)
    expected = {(*key, draw) for key in prompts for draw in range(config["sampled_responses"] + 1)}
    if seen != expected or manifest["response_count"] != len(rows):
        raise ValueError("Missing bank questions or draws")


def validate_bank(directory, config):
    manifest = read(directory / "manifest.json")
    if digest(directory / "responses.jsonl") != manifest["responses_sha256"]:
        raise ValueError("Bank response hash changed")
    rows = records(directory / "responses.jsonl")
    validate_responses(manifest, rows, config)
    return manifest, rows


def template(row):
    output = {**row, "answer": None, "official_accepted": None, "class": None,
        "supported": None, "confidence": None, "reviewer": "", "evidence": [], "review_notes": "", "excluded_reason": ""}
    try:
        output["answer"] = parse_single_answer(row["raw_response"])
    except ValueError:
        output["excluded_reason"] = "invalid_single_answer_schema"
    return output


def reliable(row):
    if row.get("excluded_reason"):
        return False
    if parse_single_answer(row["raw_response"]) != row.get("answer"):
        raise ValueError("Review changed the parsed answer")
    if (row.get("class") not in ("C1", "C2", "C3") or type(row.get("official_accepted")) is not bool or
            type(row.get("supported")) is not bool or row.get("confidence") != "high" or
            not row.get("reviewer") or not row.get("review_notes")):
        return False
    if (row["class"] == "C3") != row["official_accepted"]:
        raise ValueError("C3 must agree with official matching; an exact mismatch is not C1")
    if row["class"] in ("C2", "C3") and not row["supported"]:
        return False
    context = json.loads(row["prompt"][1]["content"])
    snippets = {str(s["id"]): s["text"] for s in context["snippets"]}
    if row["supported"] and not row["evidence"]:
        return False
    if any(e.get("snippet_id") not in snippets or not isinstance(e.get("quote"), str)
           or len(e["quote"].strip()) < 8 or e["quote"] not in snippets[e["snippet_id"]] for e in row["evidence"]):
        return False
    return True


def build_pairs(annotations, cap=2):
    groups, exclusions = defaultdict(list), Counter()
    for row in annotations:
        if not reliable(row):
            exclusions["unresolved_or_excluded"] += 1
            continue
        groups[(row["split"], row["question_id"])].append(row)
    output = {strategy: {s: [] for s in ("train", "validation")} for strategy in STRATEGIES}
    for (split, qid), group in sorted(groups.items()):
        unique = {}
        for row in sorted(group, key=lambda r: r["response_id"]):
            key = row["answer"].casefold()
            if key in unique and (unique[key]["class"], unique[key]["supported"]) != (row["class"], row["supported"]):
                raise ValueError("Conflicting labels for an identical candidate")
            unique.setdefault(key, row)
        per_type = {"concept": [], "form": []}
        for chosen, rejected in itertools.product(unique.values(), repeat=2):
            if chosen["class"] != "C3" or rejected["class"] not in ("C1", "C2"):
                continue
            if chosen["prompt"] != rejected["prompt"]:
                raise ValueError("Pair prompts differ")
            strategy = "concept" if rejected["class"] == "C1" else "form"
            per_type[strategy].append({"question_id": qid, "prompt": chosen["prompt"],
                "chosen": chosen["raw_response"], "rejected": rejected["raw_response"],
                "chosen_id": chosen["response_id"], "rejected_id": rejected["response_id"],
                "chosen_model": chosen["model_key"], "rejected_model": rejected["model_key"],
                "chosen_class": "C3", "rejected_class": rejected["class"], "contrast": strategy})
        for strategy in ("concept", "form"):
            per_type[strategy] = per_type[strategy][:cap]
            output[strategy][split].extend(per_type[strategy])
        mixed = [pair for rows in itertools.zip_longest(per_type["concept"], per_type["form"])
                 for pair in rows if pair is not None][:cap]
        output["mixed"][split].extend(mixed)
    return output, dict(exclusions)


def audit_aliases(config):
    evaluation = read(ROOT / config["evaluation_config"])
    matrix = read(ROOT / evaluation["training_matrix"])
    source, _, _ = load_source(matrix)
    path = source / "audit.jsonl"
    if digest(path) != read(source / "manifest.json")["output_sha256"]["audit.jsonl"]:
        raise ValueError("Official aliases audit changed")
    return {r["question_id"]: r["gold_aliases"] for r in records(path)}


def pack_banks(paths, config):
    banks, values = {}, []
    for path in paths:
        manifest, rows = validate_bank(path, config)
        model = manifest["model_key"]
        if model in banks or model not in MODELS:
            raise ValueError("Expected one bank per original model")
        banks[model] = {"manifest": manifest}
        values.extend(rows)
    if set(banks) != set(MODELS) or len({json.dumps(b["manifest"]["question_ids"], sort_keys=True) for b in banks.values()}) != 1:
        raise ValueError("All three banks must use the same question partitions")
    return banks, values


def validate_pairs(directory, config, strategy="mixed"):
    manifest = read(directory / "manifest.json")
    if manifest["status"] != "ready" or manifest["dpo_config_sha256"] != digest(DEFAULT_CONFIG):
        raise ValueError("Preference package not ready for this original-DPO configuration")
    if set(manifest["banks"]) != set(MODELS):
        raise ValueError("Shared preferences require all three original model banks")
    raw = records(directory / "responses.jsonl")
    annotations = records(directory / "annotations.jsonl")
    for name in ("responses.jsonl", "annotations.jsonl"):
        if digest(directory / name) != manifest["files_sha256"][name]:
            raise ValueError("Preference response/review hash changed")
    for model, bank in manifest["banks"].items():
        selected = [r for r in raw if r["model_key"] == model]
        validate_responses(bank["manifest"], selected, config)
    by_id = {r["response_id"]: r for r in raw}
    if len(by_id) != len(raw) or len(annotations) != len(raw) or {r["response_id"] for r in annotations} != set(by_id):
        raise ValueError("Review must retain all raw responses exactly once")
    for row in annotations:
        if any(row.get(k) != v for k, v in by_id[row["response_id"]].items()):
            raise ValueError("Review changed raw bank data")
    pairs, _ = build_pairs(annotations, config["max_pairs_per_question"])
    for s in ("train", "validation"):
        filename = f"{strategy}-{s}.jsonl"
        if digest(directory / filename) != manifest["files_sha256"][filename] or records(directory / filename) != pairs[strategy][s]:
            raise ValueError("Pairs differ from audited classification decisions")
        if not pairs[strategy][s]:
            raise ValueError(f"No reliable {strategy} {s} pairs")
    return manifest, pairs[strategy]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("review", "pairs", "validate"), required=True)
    parser.add_argument("--banks", nargs=3, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="mixed")
    args = parser.parse_args()
    config = read(DEFAULT_CONFIG)
    if args.mode == "validate":
        _, pairs = validate_pairs(args.output, config, args.strategy)
        print({s: len(v) for s, v in pairs.items()})
        return
    if not args.banks:
        parser.error("--banks requires all three completed banks")
    banks, raw = pack_banks(args.banks, config)
    if args.mode == "review":
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        rows = [template(r) for r in raw]
        aliases = audit_aliases(config)
        contexts = {r["question_id"]: json.loads(r["prompt"][1]["content"]) for r in raw}
        examples = [{"question_id": qid, "question": contexts[qid]["question"], "gold_aliases": aliases[qid]}
                    for qid in sorted(contexts)]
        candidates = [{"question_id": r["question_id"], "answer": r["answer"]} for r in rows if r["answer"]]
        sys.path.insert(0, str(ROOT.parent))
        from src.notebook_workflows.coverage_comparison import official_candidate_matches
        report = args.output / "official-audit"
        report.mkdir()
        jar = ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
        matches = official_candidate_matches(examples, candidates, report, jar_path=jar) if candidates else {}
        for row in rows:
            if row["answer"]:
                row["official_accepted"] = matches[(row["question_id"], row["answer"])]
                row["class"] = "C3" if row["official_accepted"] else None
        write_jsonl(args.output / "responses.jsonl", raw)
        write_jsonl(args.output / "review.jsonl", rows)
        write(args.output / "manifest.json", {"status": "review", "banks": banks,
            "dpo_config_sha256": digest(DEFAULT_CONFIG), "files_sha256": {
                name: digest(args.output / name) for name in ("responses.jsonl", "review.jsonl")}})
        print(f"Official matches saved: {args.output}; unmatched C1/C2 and evidence support remain unresolved")
        return
    if not args.annotations:
        parser.error("--annotations is required")
    rows = records(args.annotations)
    originals = {r["response_id"]: r for r in raw}
    if len(rows) != len(raw) or {r["response_id"] for r in rows} != set(originals):
        raise ValueError("Annotation IDs must match every bank response")
    for row in rows:
        if any(row.get(k) != v for k, v in originals[row["response_id"]].items()):
            raise ValueError("Annotation changed raw response")
    pairs, excluded = build_pairs(rows, config["max_pairs_per_question"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "responses.jsonl", raw)
    write_jsonl(args.output / "annotations.jsonl", rows)
    names = ["responses.jsonl", "annotations.jsonl"]
    for strategy in STRATEGIES:
        for split, values in pairs[strategy].items():
            name = f"{strategy}-{split}.jsonl"
            write_jsonl(args.output / name, values)
            names.append(name)
    write(args.output / "manifest.json", {"status": "ready", "banks": banks,
        "dpo_config_sha256": digest(DEFAULT_CONFIG), "files_sha256": {n: digest(args.output / n) for n in names},
        "pair_counts": {k: {s: len(v) for s, v in p.items()} for k, p in pairs.items()}, "exclusions": excluded,
        "label_source": "explicit semantic/evidence review; C2 is scientifically correct, unaccepted wording",
        "portability": "Raw banks embedded; adapter provenance preserved for Gadi training"})
    print(json.dumps({k: {s: len(v) for s, v in p.items()} for k, p in pairs.items()}, indent=2))


if __name__ == "__main__":
    main()
