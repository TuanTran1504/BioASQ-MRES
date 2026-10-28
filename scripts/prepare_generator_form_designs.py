#!/usr/bin/env python3
"""Freeze size-matched model-specific/shared DPO designs from inspected pairs."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_generator_form_pilot as reviewed
from scripts import prepare_answer_selector_data as data

SCHEMA = "generator-form-design-comparison-v1"
DESIGNS = ("model_specific", "shared")


def key(pair):
    return reviewed.identity([pair["question_id"], pair["prompt"], pair["chosen"], pair["rejected"]])


def pooled(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result = []
    for identity, group in sorted(groups.items()):
        group.sort(key=lambda p: (data.MODELS.index(p["model_key"]), p["pair_id"]))
        first = group[0]
        result.append({k: first[k] for k in ("question_id", "prompt", "chosen", "rejected", "chosen_id", "rejected_id")} | {
            "preference_key": identity,
            "origins": [{k: p[k] for k in ("model_key", "pair_id", "chosen_id", "rejected_id", "source_pair_ids", "source_objectives")} for p in group]})
    return result


def question_balanced(rows, count, seed, namespace):
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_id"]].append(row)
    for q, group in groups.items():
        group.sort(key=lambda p: reviewed.identity([seed, namespace, q, p["preference_key"]]))
    questions = sorted(groups, key=lambda q: reviewed.identity([seed, namespace, q]))
    ordered = [groups[q][i] for i in range(max(map(len, groups.values()), default=0))
               for q in questions if i < len(groups[q])]
    if count > len(ordered) or count < 1:
        raise ValueError("Insufficient unique reviewed training preferences")
    return ordered[:count]


def shared_balanced(rows, count, seed):
    if count % len(data.MODELS):
        raise ValueError("Shared training size must permit equal author-source sampling slots")
    queues = {m: question_balanced([r for r in rows if any(o["model_key"] == m for o in r["origins"])],
               sum(any(o["model_key"] == m for o in r["origins"]) for r in rows), seed, "shared/" + m)
              for m in data.MODELS}
    result, used = [], set()
    for _ in range(count // len(data.MODELS)):
        for model in data.MODELS:
            while queues[model] and queues[model][0]["preference_key"] in used:
                queues[model].pop(0)
            if not queues[model]:
                raise ValueError("Cannot fill equal source slots without duplicate preferences")
            row = queues[model].pop(0)
            result.append({**row, "sampling_origin": model})
            used.add(row["preference_key"])
    return result


def construct(rows, split, count=132, seed=3407):
    training = [p for p in rows if split[p["question_id"]] == "train"]
    holdout = pooled([p for p in rows if split[p["question_id"]] == "inner_validation"])
    specific = {m: question_balanced(pooled([p for p in training if p["model_key"] == m]), count, seed, m)
                for m in data.MODELS}
    shared = shared_balanced(pooled(training), count, seed)
    if {r["question_id"] for r in shared} & {r["question_id"] for r in holdout}:
        raise ValueError("Question leakage into common preference holdout")
    return specific, shared, holdout


def build(inspection, selector, output, archive=None, count=132):
    if output.exists() or (archive and archive.exists()):
        raise FileExistsError("Choose new output/archive names")
    state = reviewed.validate(inspection)
    if state["inspection_scope"] != "all_pairs" or state["uninspected_pair_records"] != 0:
        raise ValueError("Need the full inspected form-preference export")
    original = data.read(selector / "manifest.json")
    if data.digest(selector / "manifest.json") != state["selector_manifest_sha256"]:
        raise ValueError("Wrong source selector audit")
    data.verify_files(selector, original["output_sha256"])
    rows = data.records(inspection / "selected_pairs.jsonl")
    split = data.read(inspection / "question_split.json")
    specific, shared, holdout = construct(rows, split, count)
    fitting = sorted({r["question_id"] for r in data.records(selector / "train_inputs.jsonl")})
    protected = {"validation144": sorted({r["question_id"] for r in data.records(selector / "validation_inputs.jsonl")}),
                 **data.read(selector / "heldout_ids.json")}
    if not set(split) <= set(fitting) or any(set(split) & set(ids) for ids in protected.values()):
        raise ValueError("Preferences overlap a protected panel")
    output.mkdir(parents=True)
    shutil.copytree(inspection, output / "review")
    shutil.copy2(selector / "source_provenance.json", output / "source_provenance.json")
    shutil.copy2(selector / "manifest.json", output / "selector_audit_manifest.json")
    data.write(output / "protected_ids.json", protected)
    data.write(output / "fitting_ids.json", fitting)
    data.jsonl(output / "inner_validation.jsonl", holdout)
    for m in data.MODELS:
        data.jsonl(output / "model_specific" / m / "train.jsonl", specific[m])
    data.jsonl(output / "shared/train.jsonl", shared)
    protocol = {"seed": 3407, "formulation": "expansion", "output_root": "outputs/generator_form_designs",
        "max_seq_length": 8704, "beta": .1, "learning_rate": 1e-6, "num_train_epochs": 3,
        "gradient_accumulation_steps": 4, "warmup_ratio": .1, "minimum_optimizer_steps": 12,
        "minimum_train_questions": 64, "minimum_validation_questions": 12,
        "retain_epoch_checkpoints": True, "epoch_validation_mrr": False,
        "review_scope": "official_form", "training_pairs": count,
        "checkpoint_selection": "validation144 sampling4: ensemble full-pool coverage, focal full-pool coverage, MRR@5, earliest epoch; epoch zero included",
        "objective": "official answer-form acceptance coverage; not semantic-error correction or compactness"}
    data.write(output / "training_protocol.json", protocol)
    questions = {m: {p["question_id"] for p in specific[m]} for m in data.MODELS}
    summary = {"training_pairs_per_run": count, "shared_train_questions": len({p["question_id"] for p in shared}),
        "model_specific_train_questions": {m: len(q) for m, q in questions.items()},
        "common_preference_holdout_pairs": len(holdout), "common_preference_holdout_questions": len({p["question_id"] for p in holdout}),
        "shared_sampling_slots": dict(Counter(p["sampling_origin"] for p in shared)),
        "common_model_specific_training_questions": len(set.intersection(*questions.values())),
        "optimizer_steps_per_run": ((count + 3) // 4) * 3,
        "question_distribution_matched": False, "training_launched": False,
        "interpretation": "Size/update-matched practical pooling comparison; source diversity and question coverage vary together. This does not isolate response authorship alone."}
    data.write(output / "summary.json", summary)
    (output / "report.md").write_text("# Two generator DPO data designs\n\n" +
        json.dumps(summary, indent=2) + "\n\nSix DPO runs: three original max-five SFT sources, two designs each. "
        "All use the same number of training records/updates and identical pooled fitting-only preference holdout. "
        "Shared training JSONL is literally the same file for all three targets. Each target uses its own tokenizer and frozen starting SFT reference. "
        f"Question distributions differ; only {summary['common_model_specific_training_questions']} fitting questions have selected own-model pairs for all three. "
        "No validation144/dev160/test95 preferences. Retained raw responses/prompts are unchanged. "
        "GPU preflight and native-source verification are required on Gadi before training.\n", encoding="utf-8")
    data.write(output / "manifest.json", {"schema": SCHEMA, "status": "prepared", "training_ready": False,
        "inspection_manifest_sha256": data.digest(inspection / "manifest.json"),
        "selector_manifest_sha256": state["selector_manifest_sha256"],
        "files_sha256": {p.relative_to(output).as_posix(): data.digest(p) for p in output.rglob("*") if p.is_file()}})
    validate(output)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            for p in sorted(output.rglob("*")):
                if p.is_file(): handle.add(p, arcname=p.relative_to(output).as_posix())
    return summary


def validate(directory):
    state = data.read(directory / "manifest.json")
    if state.get("schema") != SCHEMA or state.get("status") != "prepared":
        raise ValueError("Wrong design-comparison package")
    data.verify_files(directory, state["files_sha256"])
    if data.digest(directory / "selector_audit_manifest.json") != state["selector_manifest_sha256"]:
        raise ValueError("Source selector manifest changed")
    reviewed.validate(directory / "review")
    if data.digest(directory / "review/manifest.json") != state["inspection_manifest_sha256"]:
        raise ValueError("Inspected preference source changed")
    split = data.read(directory / "review/question_split.json")
    protocol = data.read(directory / "training_protocol.json")
    specific, shared, holdout = construct(data.records(directory / "review/selected_pairs.jsonl"), split, protocol["training_pairs"], protocol["seed"])
    if shared != data.records(directory / "shared/train.jsonl") or holdout != data.records(directory / "inner_validation.jsonl"):
        raise ValueError("Shared preferences no longer reconstruct")
    for m in data.MODELS:
        if specific[m] != data.records(directory / "model_specific" / m / "train.jsonl"):
            raise ValueError("Own-model preferences no longer reconstruct")
        if any(o["model_key"] != m for p in specific[m] for o in p["origins"]):
            raise ValueError("Foreign response in model-specific arm")
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspection", type=Path, required=True)
    p.add_argument("--selector-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.inspection.resolve(), a.selector_audit.resolve(), a.output.resolve(), a.archive.resolve() if a.archive else None), indent=2))
