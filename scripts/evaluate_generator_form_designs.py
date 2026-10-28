#!/usr/bin/env python3
"""Coverage-primary validation and comparison for matched generator DPO designs."""
from collections import Counter, defaultdict
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))
from scripts import prepare_answer_selector_data as data
from scripts.prepare_generator_form_designs import DESIGNS
from run_generator_form_designs import native_inputs, validate_child


def exact_pools(rows, examples):
    """Verbatim pool construction for either all three models or a focal model."""
    pools, lengths, diagnostics = {}, Counter(), Counter()
    expected_models = {r["model_key"] for r in rows}
    if not expected_models or not expected_models <= set(data.MODELS):
        raise ValueError("Unknown generation source")
    expected = {(s, q, m, d) for s, panel in examples.items() for q in panel for m in expected_models for d in range(5)}
    seen = set()
    for r in sorted(rows, key=lambda r: (r["split"], r["question_id"], data.MODELS.index(r["model_key"]), r["draw"])):
        identity = (r["split"], r["question_id"], r["model_key"], r["draw"])
        if identity in seen or identity not in expected:
            raise ValueError("Duplicate/out-of-panel validation draw")
        seen.add(identity)
        split, q, m, draw = identity
        mode = "max5_greedy" if draw == 0 else "max5_sampling4"
        pp = pools.setdefault(split, {}).setdefault(mode, {})
        candidates, warnings, _ = data.parse_surfaces(r["raw_response"])
        diagnostics.update(warnings)
        emitted = set()
        for c in candidates:
            answer = c["answer"]
            target = pp.setdefault((q, answer), {"question_id": q, "answer": answer, "sources": [],
                "source_ranks": {}, "relation_types": [], "sample_frequencies": {},
                "surface_operations": [], "is_format_variant": False})
            if m not in target["sources"]:
                target["sources"].append(m)
                lengths[split, mode, q, m] += 1
                target["source_ranks"][m] = lengths[split, mode, q, m]
            if c["relation_type"] not in target["relation_types"]:
                target["relation_types"].append(c["relation_type"])
            if answer not in emitted:
                target["sample_frequencies"][m] = target["sample_frequencies"].get(m, 0) + (1 if draw == 0 else .25)
                emitted.add(answer)
    if seen != expected:
        raise ValueError("Missing validation draws; retain failures in the denominator")
    # Empty/failed pools must still exist for every mode/panel.
    for split in examples:
        for mode in ("max5_greedy", "max5_sampling4"):
            pools.setdefault(split, {}).setdefault(mode, {})
    return {s: {m: list(pp.values()) for m, pp in modes.items()} for s, modes in pools.items()}, dict(diagnostics)


def choose_coverage(rows, model="llama31", require_individual_coverage=True):
    """Coverage objective does not reward shortening or require zero coverage losses."""
    base = next(r for r in rows if r["epoch"] == 0)
    for r in rows:
        r["eligible"] = all(r["generation_quality"][k] <= base["generation_quality"][k]
                            for k in ("empty_responses", "ceiling_responses", "surplus_responses"))
    return max((r for r in rows if r["eligible"]), key=lambda r:
        (r["ensemble_full_pool_coverage"], r["focal_full_pool_coverage"], r["mrr_at5"], -r["epoch"]))


def selector_native(directory, config, repository):
    return native_inputs(repository, directory, config["model_key"], config["design"])


def validate(submission, training, output):
    validate_child(submission)
    from select_generator_feedback_dpo import select
    select(submission, training, output, native_validator=selector_native,
           checkpoint_chooser=choose_coverage, pool_builder=exact_pools, preserve_surfaces=True,
           jar_path_override=submission / "scorer.jar")
    validate_child(submission)


def joint_scores(submission, paths, summaries, output):
    """Combine saved focal outputs; no further model calls or fitting."""
    import joblib
    from scripts.prepare_reranker_guided_dpo import configure_data_root
    from scripts.evaluate_sft_ensemble_rerankers import rank_frozen
    from src.notebook_workflows.coverage_comparison import official_candidate_matches
    from select_generator_feedback_dpo import native_examples, projected_rows
    parent = data.read(submission / "prepared.json")
    first = Path(parent["children"][0])
    p, _, _, _ = validate_child(first)
    native = configure_data_root(Path(p["repository"]))
    generation = data.read(native.DEFAULT_CONFIG)
    feedback = p["feedback"]
    _, raw = native.pack_banks([Path(feedback["banks"][m]["path"]) for m in data.MODELS], generation)
    examples = native_examples(native, generation, raw)
    context = {e["question_id"]: e for e in examples}
    baseline = [r for r in raw if r["split"] == "validation"]
    alternatives = {"joint_baseline": baseline}
    for design in DESIGNS:
        for which in ("selected", "final_epoch"):
            combined = []
            for model in data.MODELS:
                epoch = summaries[model, design]["selected_epoch"] if which == "selected" else 3
                if epoch == 0:
                    combined.extend(r for r in baseline if r["model_key"] == model)
                else:
                    root = paths[model, design] / f"epoch-{epoch}"
                    generated = {mode: data.records(root / mode / "generations.jsonl") for mode in ("greedy", "sampling4")}
                    combined.extend(projected_rows(generated, model))
            alternatives[f"joint_{design}_{which}"] = combined
    pools = {}
    for name, rows in alternatives.items():
        pp, _ = exact_pools(rows, {"validation": context})
        pools.update({name + "_" + mode.removeprefix("max5_"): rr for mode, rr in pp["validation"].items()})
    distinct = {(r["question_id"], r["answer"]): r for rr in pools.values() for r in rr}
    official = output / "official_joint"
    official.mkdir(parents=True)
    jar = submission / "scorer.jar"
    if data.digest(jar) != parent["scorer_sha256"]:
        raise ValueError("Frozen comparison scorer changed")
    matches = official_candidate_matches(examples, list(distinct.values()), official, jar_path=jar) if distinct else {}
    for rr in pools.values():
        for r in rr:
            r["label"] = int(matches[r["question_id"], r["answer"]])
    rerankers = Path(feedback["rerankers"])
    meta = data.read(rerankers / "manifest.json")["selectors"]["max5"]
    payload = joblib.load(rerankers / meta["model_path"])
    conditions, scores, rankings = rank_frozen(payload, pools, context)
    data.jsonl(output / "joint_rankings.jsonl", rankings)
    return conditions, scores


def contrast(before, after, perq):
    from scripts.analyze_matched_qwen_expansion import paired_contrast
    from scripts.analyze_generator_feedback_evaluation import coverage_transitions
    result = paired_contrast(perq[before], perq[after], metrics=("mrr_at5", "coverage_at5", "coverage_full_pool"))
    return {"before": before, "after": after,
        "metrics": {k: {"difference": v["sft_minus_base"], **{x:y for x,y in v.items() if x != "sft_minus_base"}}
                    for k,v in result.items()},
        "coverage_transitions": {k: coverage_transitions(perq[before],perq[after],k)
                                 for k in ("coverage_at5", "coverage_full_pool")}}


def _compare(submission, output):
    if output.exists():
        raise FileExistsError(output)
    p = data.read(submission / "prepared.json")
    data.verify_files(submission / "code", p["code_hashes"])
    if data.digest(submission / "compare.pbs") != p["comparison_job_sha256"]:
        raise ValueError("Comparison job changed")
    with (submission / "jobs.tsv").open() as f:
        jobs = list(csv.DictReader(f, delimiter="\t"))
    paths = {(r["model"], r["design"]): Path(r["output"]) for r in jobs if r["stage"] == "validate"}
    expected = {(m, d) for m in data.MODELS for d in DESIGNS}
    if set(paths) != expected:
        raise ValueError("Need all six validation runs")
    summaries, scores, baseline = {}, {}, None
    conditions, perq, contrasts = {}, {}, {}
    for (m, d), path in paths.items():
        child = submission / m / d
        if data.digest(child / "prepared.json") != p["child_prepared_sha256"][str(child)]:
            raise ValueError("Child preparation changed")
        state = data.read(path / "manifest.json")
        if state["status"] != "complete" or state["smoke_test"] or len(state["question_ids"]) != 144:
            raise ValueError("Need all144 completed validation results")
        if state["submission"] != str(child) or state["prepared_sha256"] != data.digest(child / "prepared.json"):
            raise ValueError("Validation belongs to another prepared experiment")
        data.verify_files(path, state["output_sha256"])
        s, q = data.read(path / "comparison_summary.json"), data.read(path / "comparison_per_question.json")
        if s["pool_policy"] != "verbatim surfaces":
            raise ValueError("All designs must use the same verbatim pool policy")
        current_base = {mode: q[f"epoch0_{mode}_max5_reranker"] for mode in ("greedy", "sampling4")}
        if baseline is not None and current_base != baseline:
            raise ValueError("Baseline pools/scoring differ across experiments")
        baseline = current_base
        summaries[m, d], scores[m, d] = s, q
        for which, epoch in (("selected", s["selected_epoch"]), ("final_epoch", 3)):
            for mode in ("greedy", "sampling4"):
                name = f"{m}_{d}_{which}_{mode}"
                key = f"epoch{epoch}_{mode}_max5_reranker"
                conditions[name] = {**s["conditions"][key], "epoch": epoch}
                perq[name] = q[key]
    for mode in ("greedy", "sampling4"):
        perq["baseline_" + mode] = baseline[mode]
        conditions["baseline_" + mode] = {"metrics": {k: sum(r[k] for r in baseline[mode])/144 for k in baseline[mode][0] if k != "question_id"}}
        for m in data.MODELS:
            for which in ("selected", "final_epoch"):
                a, b = (f"{m}_{d}_{which}_{mode}" for d in DESIGNS)
                contrasts[f"{m}_{which}_{mode}_shared_minus_specific"] = contrast(a,b,perq)
                for name in (a,b):
                    contrasts[name+"_minus_baseline"] = contrast("baseline_"+mode,name,perq)
    output.mkdir(parents=True)
    joint_conditions, joint_perq = joint_scores(submission, paths, summaries, output)
    conditions.update(joint_conditions)
    perq.update(joint_perq)
    for mode in ("greedy", "sampling4"):
        if perq[f"joint_baseline_{mode}_max5_reranker"] != baseline[mode]:
            raise ValueError("Joint and focal baseline scoring differ")
        for which in ("selected", "final_epoch"):
            a, b = (f"joint_{d}_{which}_{mode}_max5_reranker" for d in DESIGNS)
            contrasts[f"joint_{which}_{mode}_shared_minus_specific"] = contrast(a,b,perq)
            for name in (a,b):
                contrasts[name+"_minus_baseline"] = contrast(f"joint_baseline_{mode}_max5_reranker",name,perq)
    data.write(output / "comparison_summary.json", {"question_count": 144, "conditions": conditions,
        "paired_contrasts": contrasts, "models": {m: {d: {k:summaries[m,d][k] for k in ("selected_epoch","checkpoint_scores")} for d in DESIGNS} for m in data.MODELS},
        "limitations": ["Internal validation selects checkpoints and designs; intervals are descriptive, not independent held-out evidence.",
            "Training pair counts/updates are matched, question distributions are not.",
            "Focal comparisons keep two SFT companions fixed. Joint comparisons combine all three saved DPO outputs; no extra generation or joint checkpoint search.",
            "Verbatim pools are rebuilt consistently; historical normalized-pool scores are not directly comparable."]})
    data.write(output / "comparison_per_question.json", perq)
    lines = ["# Model-specific versus shared generator DPO: validation144", "",
        "132 training pairs and 99 optimizer updates/run; common preference holdout. Fixed SFT companions and logistic reranker; verbatim candidate pools.", "",
        f"Baseline ensemble: greedy MRR {conditions['baseline_greedy']['metrics']['mrr_at5']:.6f}; sampling4 MRR {conditions['baseline_sampling4']['metrics']['mrr_at5']:.6f}, full-pool coverage {conditions['baseline_sampling4']['metrics']['coverage_full_pool']:.4f}.", "",
        "| Model | Design | Epoch | Greedy MRR | Sampling4 MRR | Ensemble full-pool coverage | Model full-pool coverage | Model tokens/response |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in data.MODELS:
        for d in DESIGNS:
            a = conditions[f"{m}_{d}_selected_greedy"]["metrics"]
            b = conditions[f"{m}_{d}_selected_sampling4"]["metrics"]
            selected = next(r for r in summaries[m,d]["checkpoint_scores"] if r["epoch"] == summaries[m,d]["selected_epoch"])
            lines.append(f"| {m} | {d} | {selected['epoch']} | {a['mrr_at5']:.6f} | {b['mrr_at5']:.6f} | {b['coverage_full_pool']:.4f} | {selected['focal_full_pool_coverage']:.4f} | {selected['mean_output_tokens']:.2f} |")
    lines += ["", "## All three generators combined", "", "| Setting | Greedy MRR | Sampling4 MRR | Sampling4 full-pool coverage |", "|---|---:|---:|---:|"]
    for name in ("baseline", "model_specific_selected", "shared_selected", "model_specific_final_epoch", "shared_final_epoch"):
        a, b = (conditions[f"joint_{name}_{mode}_max5_reranker"]["metrics"] for mode in ("greedy", "sampling4"))
        lines.append(f"| {name} | {a['mrr_at5']:.6f} | {b['mrr_at5']:.6f} | {b['coverage_full_pool']:.4f} |")
    lines += ["", "Epoch 0 keeps the starting SFT. Fixed final-epoch results and paired comparisons are also in the JSON, so selected-checkpoint effects remain visible.",
        "These are repeated internal-validation/selection diagnostics. They do not establish held-out superiority or isolate response authorship from question diversity."]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    data.write(output / "manifest.json", {"status": "complete", "question_count": 144,
        "submission": str(submission), "outputs_sha256": {x.relative_to(output).as_posix(): data.digest(x) for x in output.rglob("*") if x.is_file()}})
    print("Comparison report:", output / "report.md", flush=True)


def compare(submission, output):
    if output.exists():
        raise FileExistsError(output)
    try:
        _compare(submission, output)
    except BaseException as exc:
        data.write(output / "manifest.json", {"status":"failed", "submission":str(submission),
            "error":f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "compare"), required=True)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--training", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "validate":
        if not a.training: p.error("--training is required")
        validate(a.submission.resolve(), a.training.resolve(), a.output.resolve())
    else:
        compare(a.submission.resolve(), a.output.resolve())
