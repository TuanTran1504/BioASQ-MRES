"""Reconstruct and validate a reviewed compactness DPO pilot; no CUDA imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from prepare_matched_8b_sft import read, records, digest, write
import max5_dpo_data as data
from scripts.judge_generator_feedback import load_shortlist

DEFAULT_CONFIG = ROOT / "configs/generator_feedback_dpo.json"
VETOES = {
    "66099a03fdcbea915f00001d": "Preferred response treats screening start age 45 and age intervals 45-49/45-74 as equivalent; supplied snippets distinguish them.",
    "58cc5bae02b8c60953000037": "Supplied ALD estimate has the lower bound 'at least'; preferred responses omit it. Hold out rather than approve the bound change.",
    "5ace17c30340b9f058000009": "Question asks androsterone, but supplied OR7D4 evidence concerns androstenone. Do not approve the judge's unsupported terminology substitution.",
    "601db4ac1cb411341a000048": "Preferred protein expression 'lamin A (LMNA)' declares gene name LMNA an abbreviation of a protein. Hold the gene/protein and precursor distinctions out of this clean pilot."
}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def focal_model(config):
    model = config.get("model_key", "llama31")
    if model not in data.MODELS:
        raise ValueError("Unknown compactness DPO model")
    return model


def check_pair(pair, originals, allow_pool_ties=False, model="llama31"):
    a, b = (originals[pair[k]] for k in ("chosen_id", "rejected_id"))
    if (pair["split"] not in ("train", "validation") or pair["model_key"] != model
            or a["model_key"] != model or b["model_key"] != model
            or a["split"] != pair["split"] or b["split"] != pair["split"]
            or a["question_id"] != pair["question_id"] or b["question_id"] != pair["question_id"]
            or pair["chosen"] != a["raw_response"] or pair["rejected"] != b["raw_response"]
            or a["prompt"] != b["prompt"] or pair["prompt"] != a["prompt"]):
        raise ValueError("Preference raw strings, prompt or partition differs")
    norm = lambda c: (" ".join(c["answer"].split()).casefold(), c["relation_type"])
    ac, bc = tuple(map(norm, data.answers(pair["chosen"]))), tuple(map(norm, data.answers(pair["rejected"])))
    if ac[0] != bc[0] or len(ac) == len(bc):
        raise ValueError("Expected unchanged initial answer and a strict nested list")
    small, large = (ac, bc) if len(ac) < len(bc) else (bc, ac)
    cursor = iter(large)
    if not all(any(x == y for y in cursor) for x in small):
        raise ValueError("Preference lists are not ordered subsets with retained tags")
    accepted_a = {norm(c)[0] for c in a["candidates"] if c["official_accepted"]}
    accepted_b = {norm(c)[0] for c in b["candidates"] if c["official_accepted"]}
    ao, bo = pair["chosen_outcome"], pair["rejected_outcome"]
    if (not accepted_a or not accepted_b <= accepted_a or ao["mrr_at5"] < bo["mrr_at5"] - 1e-12
            or ao["coverage_full_pool"] < bo["coverage_full_pool"]
            or ao["fixed_response_ids"] != bo["fixed_response_ids"]
            or ao["fixed_response_ids"] != pair["fixed_response_ids"] or len(pair["fixed_response_ids"]) != 11):
        raise ValueError("Accepted-form, ranking or companion preservation failed")
    if pair["objective"] == "compactness":
        if (len(ac) >= len(bc) or not a["candidates"][0]["official_accepted"]
                or ao["pool_candidates"] > bo["pool_candidates"]
                or (not allow_pool_ties and ao["pool_candidates"] == bo["pool_candidates"])):
            raise ValueError("Compactness must shrink the pool with accepted initial retained")
    elif pair["objective"] == "retain_needed_expansion":
        if len(ac) <= len(bc) or accepted_b:
            raise ValueError("Retention must preserve an accepted focal form missing from the shorter response")
    else:
        raise ValueError("Unsupported generator preference objective")


def choose_pairs(source, annotations, config):
    scope = config.get("review_scope", "pool_reducing")
    model = focal_model(config)
    _, candidates, original = load_shortlist(source, scope, model)
    annotations = [r for r in annotations if r["model_key"] == model]
    expected = {r["response_id"]: r for r in original}
    reviewed = {r["response_id"]: r for r in annotations}
    if len(reviewed) != len(annotations) or set(reviewed) != set(expected):
        raise ValueError("Judgments must retain every shortlisted response exactly once")
    for identity, row in reviewed.items():
        old = expected[identity]
        if any(row.get(k) != v for k, v in old.items() if k not in ("candidates", "reviewer", "review_notes", "excluded_reason")):
            raise ValueError("Review changed raw generation data")
        if len(row["candidates"]) != len(old["candidates"]):
            raise ValueError("Review changed candidate count")
        for a, b in zip(row["candidates"], old["candidates"]):
            if any(a.get(k) != b.get(k) for k in ("answer", "relation_type", "official_accepted")):
                raise ValueError("Review changed candidate text, relation or acceptance")
    eligible = {s: {"compactness": [], "retain_needed_expansion": []} for s in ("train", "validation")}
    vetoes = {**VETOES, **config.get("assistant_vetoes", {})}
    exclusions, review = Counter(), []
    for p in candidates:
        check_pair(p, expected, allow_pool_ties=scope == "all_compactness", model=model)
        a, b = reviewed[p["chosen_id"]], reviewed[p["rejected_id"]]
        reason = None
        if p["question_id"] in vetoes:
            reason = "assistant_numeric_or_scope_veto"
        elif not data.reliable(a) or not data.reliable(b):
            reason = "unresolved_or_conflicting_response_review"
        elif not all(data.valid(c) for c in a["candidates"]):
            reason = "chosen_not_fully_supported_correct_equivalent_and_valid"
        elif not data.valid(b["candidates"][0]):
            reason = "rejected_initial_not_semantically_correct_and_supported"
        elif p["objective"] == "retain_needed_expansion" and not all(data.valid(c) for c in b["candidates"]):
            reason = "retention_short_response_not_clean"
        if reason:
            exclusions[reason] += 1
            review.append({"question_id": p["question_id"], "objective": p["objective"],
                           "chosen_id": p["chosen_id"], "rejected_id": p["rejected_id"], "excluded_reason": reason})
            continue
        # Compactness deliberately allows deletion of valid optional expressions.
        # Never describe every removed candidate as a factual error.
        removed = [c for c in b["candidates"] if c["answer"].casefold() not in
                   {x["answer"].casefold() for x in a["candidates"]}]
        accepted = {**p, "semantic_review_status": "LLM_high_confidence_with_assistant_vetoes",
                    "training_ready": True, "deletion_assessment": {
                        "removed_candidates": len(removed), "judge_assessed_valid": sum(data.valid(c) for c in removed),
                        "judge_assessed_harmful": sum(data.harmful(c) for c in removed)}}
        if scope == "all_compactness":
            accepted["pool_effect"] = ("shrinks_pool" if p["chosen_outcome"]["pool_candidates"] < p["rejected_outcome"]["pool_candidates"]
                                       else "unchanged_pool" if p["chosen_outcome"]["pool_candidates"] == p["rejected_outcome"]["pool_candidates"] else "larger_pool_retention")
        eligible[p["split"]][p["objective"]].append(accepted)
    selected, counts = {}, {}
    for split, arms in eligible.items():
        def order(p):
            return hashlib.sha256(f"{config['seed']}/{split}/{p['question_id']}/{p['chosen_id']}/{p['rejected_id']}".encode()).hexdigest()
        capped = {}
        for arm, rows in arms.items():
            unique = {}
            for p in sorted(rows, key=order):
                unique.setdefault(p["question_id"], p)
            capped[arm] = list(unique.values())
        controls = capped["retain_needed_expansion"]
        limit = (len(capped["compactness"]) if split == "validation" and config.get("validation_use_all_reviewed_questions")
                 else len(controls) * config["compactness_to_retention_ratio"])
        compact = capped["compactness"][:limit]
        selected[split] = sorted(controls + compact, key=order)
        counts[split] = {"eligible_pairs": {arm: len(rows) for arm, rows in arms.items()},
                         "distinct_eligible_questions": {arm: len(rows) for arm, rows in capped.items()},
                         "selected_pairs": len(selected[split]),
                         "selected_questions": len({p["question_id"] for p in selected[split]}),
                         "selected_objectives": dict(Counter(p["objective"] for p in selected[split]))}
        if scope == "all_compactness":
            counts[split]["pool_effects"] = dict(Counter(p["pool_effect"] for p in selected[split]))
    return selected, {"counts": counts, "excluded": dict(exclusions), "excluded_pairs": review,
                      "assistant_vetoes": vetoes}


def build(source, judge, output, config_path=DEFAULT_CONFIG):
    if output.exists():
        raise FileExistsError(output)
    config = read(config_path)
    reviewed = read(judge / "manifest.json")
    if (reviewed["status"] != "complete" or reviewed["source_manifest_sha256"] != digest(source / "manifest.json")
            or reviewed.get("review_scope", "pool_reducing") != config.get("review_scope", "pool_reducing")):
        raise ValueError("Need complete, matching targeted semantic judgments")
    if digest(judge / "reviewed.jsonl") != reviewed["reviewed_sha256"]:
        raise ValueError("Semantic judgments changed")
    splits, summary = choose_pairs(source, records(judge / "reviewed.jsonl"), config)
    for split, rows in splits.items():
        minimum = config[f"minimum_{split}_questions"]
        if summary["counts"][split]["selected_questions"] < minimum:
            raise ValueError(f"Insufficient reviewed {split} questions: {summary['counts'][split]}")
        if len(summary["counts"][split]["selected_objectives"]) != 2:
            raise ValueError("Both compactness and retention controls are required in each split")
    output.mkdir(parents=True)
    manifest = read(source / "manifest.json")
    (output / "source").mkdir()
    shutil.copy2(source / "manifest.json", output / "source/manifest.json")
    for name in manifest["output_sha256"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(judge / "reviewed.jsonl", output / "annotations.jsonl")
    shutil.copy2(judge / "manifest.json", output / "judging_manifest.json")
    shutil.copy2(config_path, output / "training_config.json")
    for split, rows in splits.items():
        write_rows(output / f"{split}.jsonl", rows)
    write(output / "review_summary.json", summary)
    hashes = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()}
    write(output / "manifest.json", {"status": "ready", "schema": "generator-feedback-compactness-v1",
        "model_key": focal_model(config), "configuration": config, "pair_counts": {s: len(r) for s, r in splits.items()},
        "source_manifest_sha256": digest(source / "manifest.json"), "files_sha256": hashes,
        "annotation_source": "GPT-4.1-mini automated review plus explicit assistant vetoes; not expert-certified",
        "scope": f"Question-capped {config['compactness_to_retention_ratio']}:1 compactness/retention; {config.get('review_scope', 'pool_reducing')}; no ranking-gain arm or test preferences"})
    validate(output, config)
    print(json.dumps(summary["counts"], indent=2))


def validate(directory, config):
    manifest = read(directory / "manifest.json")
    if (manifest.get("status") != "ready" or manifest.get("schema") != "generator-feedback-compactness-v1"
            or manifest.get("model_key") != focal_model(config)
            or manifest["configuration"] != config):
        raise ValueError("Wrong or incomplete reviewed generator-feedback package")
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("Reviewed training artifact changed: " + name)
    if digest(directory / "source/manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Feedback provenance changed")
    judging = read(directory / "judging_manifest.json")
    if (judging["status"] != "complete" or judging["source_manifest_sha256"] != manifest["source_manifest_sha256"]
            or judging["reviewed_sha256"] != digest(directory / "annotations.jsonl")
            or judging.get("review_scope", "pool_reducing") != config.get("review_scope", "pool_reducing")):
        raise ValueError("Annotation provenance differs from the completed judging run")
    splits, summary = choose_pairs(directory / "source", records(directory / "annotations.jsonl"), config)
    for split, rows in splits.items():
        if rows != records(directory / f"{split}.jsonl") or len(rows) != manifest["pair_counts"][split]:
            raise ValueError("Training pairs do not reconstruct from semantic decisions")
        if summary["counts"][split]["selected_questions"] < config[f"minimum_{split}_questions"]:
            raise ValueError("Too few distinct reviewed questions")
        if len(summary["counts"][split]["selected_objectives"]) != 2:
            raise ValueError("Missing retention or compactness arm")
    if {p["question_id"] for p in splits["train"]} & {p["question_id"] for p in splits["validation"]}:
        raise ValueError("Fitting/validation question overlap")
    return manifest, splits


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--judge", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    build(args.source.resolve(), args.judge.resolve(), args.output.resolve(), args.config.resolve())
