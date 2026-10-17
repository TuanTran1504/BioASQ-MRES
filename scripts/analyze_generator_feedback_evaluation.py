#!/usr/bin/env python3
"""Score compactness DPO on dev160 with the unchanged max-five logistic selector."""
import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))

import joblib
import sklearn
from prepare_matched_8b_sft import read, records, write, digest
from run_single_answer_sampling import sample_seed, expansion_candidates
from run_extractive_expansion_8b import parse_equivalent_response, write_jsonl
from scripts.train_candidate_reranker_pilot import key
from scripts.analyze_sft_ensemble_test import MODELS, merge_pool, summarize_pools
from scripts.evaluate_sft_ensemble_rerankers import rank_frozen
from scripts.compare_expansion_pool_rerankers import evaluate_rankings
from scripts.analyze_matched_qwen_expansion import paired_contrast
from scripts.ablate_reranker_relation_labels import verify_files
from src.notebook_workflows.coverage_comparison import official_candidate_matches

MODES = ("greedy", "sampling10")
METRICS = ("mrr_at5", "coverage_at1", "coverage_at5", "coverage_at10", "coverage_full_pool")
SIZE_METRICS = ("mean_candidates_per_response", "mean_output_tokens_per_response", "unique_candidates")


def input_view(rows):
    return [{k: r[k] for k in ("question_id", "question", "snippets")} for r in rows]


def validate_membership(examples, membership, preferences):
    ids = {r["question_id"] for r in examples}
    if len(examples) != 160 or len(ids) != 160:
        raise ValueError("Need exactly 160 distinct dev questions")
    split_ids = {s: {r["question_id"] for r in rr} for s, rr in membership.items()}
    if ({s: len(v) for s, v in split_ids.items()} != {"train": 1296, "validation": 144}
            or any(len(membership[s]) != len(v) for s, v in split_ids.items())
            or split_ids["train"] & split_ids["validation"]):
        raise ValueError("Invalid reranker fitting/validation membership")
    if any(ids & qq for qq in split_ids.values()):
        raise ValueError("Dev questions overlap reranker fitting or validation")
    for split, rows in preferences.items():
        if any(r["question_id"] not in split_ids[split] for r in rows):
            raise ValueError("DPO preferences outside original fitting/validation partition")


def load_selector(directory, hashes, examples, sources):
    verify_files(directory, hashes)
    state = read(directory / "manifest.json")
    if (state.get("status") != "complete" or state.get("sklearn_version") != sklearn.__version__
            or state.get("question_counts") != {"train": 1296, "validation": 144}):
        raise ValueError("Need completed frozen reranker with matching scikit-learn")
    if hashes.get("manifest.json") != digest(directory / "manifest.json"):
        raise ValueError("Unpinned reranker manifest")
    for name, value in state["artifacts_sha256"].items():
        if hashes.get(name) != value:
            raise ValueError("Missing pinned reranker artifact: " + name)
    prepared_path = Path(state["submission"]) / "prepared.json"
    if digest(prepared_path) != state["prepared_sha256"]:
        raise ValueError("Frozen reranker preparation changed")
    prepared = read(prepared_path)
    for name in ("scripts/train_candidate_reranker_pilot.py", "scripts/train_sft_ensemble_reranker.py"):
        if digest(REPOSITORY / name) != prepared["code_hashes"][name]:
            raise ValueError("Feature/pooling code differs from frozen reranker")
    from matched_8b_provenance import validate_provenance
    for model in MODELS:
        bank = read(Path(prepared["max5_banks"][model]) / "manifest.json")
        validate_provenance({"max5": bank["source"]}, {"max5": sources[model]})
    membership = {s: records(directory / f"{s}_examples.jsonl") for s in ("train", "validation")}
    validate_membership(examples, membership, {"train": [], "validation": []})
    metadata = state["selectors"]["max5"]
    path = directory / metadata["model_path"]
    if metadata["model_sha256"] != hashes.get(metadata["model_path"]):
        raise ValueError("Unpinned max-five selector weights")
    payload = joblib.load(path)
    if (payload.get("schema") != "sft-ensemble-logistic-v1" or payload.get("selector") != "max5"
            or tuple(payload.get("models", [])) != MODELS or payload.get("training_arms") != ["max5"]
            or payload.get("selection_conditions") != ["max5_greedy", "max5_sampling4"]
            or payload.get("selected_C") != state["selected_C"]["max5"]):
        raise ValueError("Frozen selector schema differs")
    return payload, state


def load_generation(directory, source, examples, mode):
    """Reparse every draw; invalid draws and empty pools stay in denominators."""
    cfg = read(directory / "config.json")
    count = 1 if mode == "greedy" else 10
    expected = {"model_name": source["adapter"], "max_candidates": 5, "max_new_tokens": 512,
        "seed": 3407, "response_mode": "equivalent_greedy" if count == 1 else "equivalent_sampling",
        "num_generations": count, "temperature": 0 if count == 1 else .8,
        "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
        "mark_snippets": False, "require_all_snippets": True, "question_count": len(examples),
        "gold_blind_generation": True, "local_files_only": True}
    if count == 10:
        expected.update(top_p=.95, top_k=0)
    if (any(cfg.get(k) != v for k, v in expected.items())
            or cfg.get("execution_mode", "default") != source.get("execution_mode", "default")):
        raise ValueError("Wrong source, decoding budget or evidence policy")
    prompt = Path(cfg["prompt"])
    if (digest(prompt) != cfg["prompt_sha256"]
            or prompt.read_text(encoding="utf-8").strip() != source["system_prompt"].strip()):
        raise ValueError("Prompt changed")
    if input_view(records(directory / "examples.jsonl")) != input_view(examples):
        raise ValueError("Generation input/evidence differs")
    rows = records(directory / "generations.jsonl")
    wanted = {r["question_id"]: r for r in examples}
    if (read(directory / "status.json")["status"] != "complete" or len(rows) != len(examples)
            or len({r["question_id"] for r in rows}) != len(rows)
            or {r["question_id"] for r in rows} != set(wanted)):
        raise ValueError("Incomplete or duplicate generation questions")
    candidates, invalid = [], []
    for row in rows:
        qid = row["question_id"]
        if (row["snippets_truncated"] or row["included_snippets"] != len(wanted[qid]["snippets"])
                or row["total_snippets"] != row["included_snippets"] or row["request_count"] != count
                or len(row["samples"]) != count
                or row["prompt_tokens"] != cfg["expected_prompt_tokens"][qid]
                or row["input_tokens"] != count * row["prompt_tokens"]):
            raise ValueError("Incomplete evidence, token plan or draws")
        for draw, sample in enumerate(row["samples"], 1):
            if sample["draw"] != draw or sample["seed"] != sample_seed(qid, draw, 3407):
                raise ValueError("Draw order or seed differs")
            try:
                aa, bad, _, _ = parse_equivalent_response(sample["raw_response"], 5)
                success = bool(aa)
            except ValueError:
                aa, bad, success = [], [], False
            if sample["answers"] != aa or bool(sample.get("parse_error")) == success:
                raise ValueError("Saved parsing differs from raw completion")
            invalid.extend({"question_id": qid, "draw": draw, **r} for r in bad)
            if sample["output_tokens"] < 0 or sample["generation_seconds"] < 0:
                raise ValueError("Negative generation cost")
        if (row["output_tokens"] != sum(s["output_tokens"] for s in row["samples"])
                or abs(row["generation_seconds"] - sum(s["generation_seconds"] for s in row["samples"])) > 1e-6):
            raise ValueError("Generation cost counters differ")
        candidates.extend({"question_id": qid, **c} for c in expansion_candidates(row["samples"]))
    if (records(directory / "candidates.jsonl") != candidates
            or records(directory / "invalid_candidates.jsonl") != invalid):
        raise ValueError("Saved candidate membership/order differs from raw responses")
    return rows


def make_pools(baseline, dpo):
    pools = {}
    for mode in MODES:
        original = {m: {"max5": baseline[f"{m}_{mode}"]} for m in MODELS}
        replacement = {**original, "llama31": {"max5": dpo[mode]}}
        for policy, responses in (("sft", original), ("dpo", replacement)):
            pools[f"{policy}_{mode}"] = merge_pool(responses, "max5")
            solo = {m: responses[m] if m == "llama31" else {"max5": []} for m in MODELS}
            pools[f"llama_{policy}_{mode}"] = merge_pool(solo, "max5")
    return pools


def coverage_transitions(before, after, metric):
    a, b = ({r["question_id"]: bool(r[metric]) for r in rows} for rows in (before, after))
    if set(a) != set(b):
        raise ValueError("Transition question membership differs")
    gained = sorted(q for q in a if not a[q] and b[q])
    lost = sorted(q for q in a if a[q] and not b[q])
    return {"sft_covered": sum(a.values()), "dpo_covered": sum(b.values()),
            "gained": len(gained), "lost": len(lost), "net": len(gained) - len(lost),
            "gained_question_ids": gained, "lost_question_ids": lost}


def compactness(rows):
    result = []
    for row in rows:
        ss = row["samples"]
        result.append({"question_id": row["question_id"],
            "mean_candidates_per_response": sum(len(s["answers"]) for s in ss) / len(ss),
            "mean_output_tokens_per_response": sum(s["output_tokens"] for s in ss) / len(ss),
            "unique_candidates": len(expansion_candidates(ss))})
    return {k: sum(r[k] for r in result) / len(result) for k in SIZE_METRICS}, result


def contrast(before, after, metrics, resamples):
    return {k: {"difference": v["sft_minus_base"], **{f: x for f, x in v.items() if f != "sft_minus_base"}}
            for k, v in paired_contrast(before, after, resamples=resamples, metrics=metrics).items()}


def score_pools(pools, examples, payload, output, jar, matcher=official_candidate_matches):
    distinct = {(r["question_id"], r["answer"]): r for rows in pools.values() for r in rows}
    matches = matcher(list(examples.values()), list(distinct.values()), output / "official", jar_path=jar) if distinct else {}
    normalized = {}
    for (qid, answer), label in matches.items():
        identity = (qid, key(answer))
        if identity in normalized and normalized[identity] != label:
            raise ValueError("Deduplication changes official acceptance")
        normalized[identity] = label
    for rows in pools.values():
        for row in rows:
            row["label"] = int(matches[(row["question_id"], row["answer"])])
    gold = {q: {**e, "batch": "dev160"} for q, e in examples.items()}
    ensembles = {k: v for k, v in pools.items() if not k.startswith("llama_")}
    conditions, scores, rankings = summarize_pools(ensembles, examples, gold, methods=("sample_frequency",))
    cc, ss, rr = rank_frozen(payload, ensembles, examples)
    conditions.update(cc); scores.update(ss); rankings.extend(rr)
    for name, rows in pools.items():
        if not name.startswith("llama_"):
            continue
        groups = {q: [] for q in examples}
        for row in rows:
            groups[row["question_id"]].append(row)
        # Llama-only baseline uses unchanged draw/within-response occurrence order.
        metrics, scored = evaluate_rankings(groups)
        conditions[name + "_generated_order"] = {"metrics": metrics, "mean_unique_candidates": len(rows) / len(examples)}
        scores[name + "_generated_order"] = scored
        rankings.extend({"condition": name + "_generated_order", "rank": i, **r}
                        for rr in groups.values() for i, r in enumerate(rr, 1))
    return conditions, scores, rankings


def analyze(submission, generation, output, resamples=10000):
    from run_generator_feedback_evaluation import validate_prepared
    if output.exists() or resamples < 2:
        raise ValueError("Need a new output directory and at least two bootstrap resamples")
    prepared = validate_prepared(submission)
    state = read(generation / "manifest.json")
    if (state.get("status") != "complete" or state.get("smoke_test") is not False
            or state.get("expected_questions") != 160 or set(state["runs"]) != set(MODES)
            or Path(state["submission"]).resolve() != submission
            or state["prepared_sha256"] != digest(submission / "prepared.json")
            or state["source"] != prepared["dpo_source"]):
        raise ValueError("Need a complete full DPO generation from this submission")
    generation_hash = digest(generation / "manifest.json")
    panel = records(submission / "dev_gold.jsonl")
    examples = {r["question_id"]: r for r in panel}
    baseline, dpo = {}, {}
    for model in MODELS:
        for mode in MODES:
            baseline[f"{model}_{mode}"] = load_generation(Path(prepared["baseline_runs"][f"{model}_{mode}"]), prepared["sources"][model], panel, mode)
    for mode in MODES:
        path = (generation / state["runs"][mode]).resolve()
        if not path.is_relative_to(generation):
            raise ValueError("DPO inference escaped generation directory")
        verify_files(path, state["arm_hashes"][mode])
        cfg = read(path / "config.json")
        if any(cfg[k] != prepared["input_plan"][k] for k in (
                "max_seq_length", "expected_prompt_tokens", "expected_prompt_ids_sha256")):
            raise ValueError("DPO input allocation differs from starting SFT")
        dpo[mode] = load_generation(path, prepared["dpo_source"], panel, mode)
    payload, _ = load_selector(Path(prepared["rerankers"]), prepared["reranker_hashes"], panel, prepared["sources"])
    pools = make_pools(baseline, dpo)
    output.mkdir(parents=True, exist_ok=False)
    result_state = {"status": "running", "submission": str(submission),
        "prepared_sha256": digest(submission / "prepared.json"), "generation": str(generation),
        "generation_manifest_sha256": generation_hash, "training": False, "paid_judging": False}
    write(output / "manifest.json", result_state)
    try:
        jar = Path(prepared["repository"]) / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
        conditions, scores, rankings = score_pools(pools, examples, payload, output, jar)
        comparisons, transitions = {}, {}
        for mode in MODES:
            for prefix, method in (("", "max5_reranker"), ("", "sample_frequency"), ("llama_", "generated_order")):
                a, b = (f"{prefix}{p}_{mode}_{method}" for p in ("sft", "dpo"))
                name = f"{prefix}dpo_minus_sft_{mode}_{method}"
                comparisons[name] = {"before": a, "after": b, "metrics": contrast(scores[a], scores[b], METRICS, resamples)}
                transitions[name] = {k: coverage_transitions(scores[a], scores[b], k) for k in METRICS if k != "mrr_at5"}
        size, size_rows, warnings = {}, {}, []
        for mode in MODES:
            for policy, rows in (("sft", baseline[f"llama31_{mode}"]), ("dpo", dpo[mode])):
                name = f"{policy}_{mode}"
                size[name], size_rows[name] = compactness(rows)
                samples = [s for r in rows for s in r["samples"]]
                size[name].update(failed_draws=sum(bool(s["parse_error"]) for s in samples),
                    output_token_ceiling_count=sum(s["output_tokens"] >= 512 for s in samples),
                    candidate_cap_exceeded_count=sum(bool(s.get("candidate_limit_applied")) for s in samples),
                    generation_seconds=sum(r["generation_seconds"] for r in rows),
                    ensemble_mean_unique_candidates=len(pools[name]) / 160)
                if any(size[name][k] for k in ("failed_draws", "output_token_ceiling_count", "candidate_cap_exceeded_count")):
                    warnings.append(name + ": parse failures/capped/truncated responses retained; inspect compactness counters")
            comparisons[f"compactness_dpo_minus_sft_{mode}"] = {"metrics": contrast(size_rows[f"sft_{mode}"], size_rows[f"dpo_{mode}"], SIZE_METRICS, resamples)}
        result = {"question_count": 160, "conditions": conditions, "paired_contrasts": comparisons,
            "coverage_transitions": transitions, "llama_generation": size, "quality_warnings": warnings,
            "primary_selector": "frozen max5 logistic with relation labels", "reranker_selected_C": payload["selected_C"],
            "limitations": ["Exploratory, historically inspected dev160; no checkpoint or reranker selection on these scores.",
                "Only Llama changes. Qwen3/Ministral response archives are identical in both conditions.",
                "Training used four-sample feedback; evaluation uses ten samples. One training/decoding seed.",
                "DPO checkpoint selected by preference loss on four validation pairs; weak pilot selection.",
                "Smaller candidate lists are not evidence of semantic correctness or higher MRR.",
                "Generation times come from different archived GPU runs and are descriptive only.",
                "Coverage@10 is a candidate cutoff, not ten generation draws. Full-pool coverage uses all unique candidates."]}
        validate_prepared(submission)
        if digest(generation / "manifest.json") != generation_hash:
            raise ValueError("Generation metadata changed while scoring")
        for mode in MODES:
            verify_files(generation / state["runs"][mode], state["arm_hashes"][mode])
        write(output / "comparison_summary.json", result)
        write(output / "comparison_per_question.json", scores)
        write(output / "compactness_per_question.json", size_rows)
        write_jsonl(output / "rankings.jsonl", rankings)
        for name, rows in pools.items():
            write_jsonl(output / f"{name}_candidates.jsonl", rows)
        for mode in MODES:
            print("\n" + mode, flush=True)
            for policy in ("sft", "dpo"):
                m = conditions[f"{policy}_{mode}_max5_reranker"]["metrics"]
                z = size[f"{policy}_{mode}"]
                print(f"  {policy}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, full pool={m['coverage_full_pool']:.4f}; "
                      f"Llama candidates/response={z['mean_candidates_per_response']:.3f}, tokens/response={z['mean_output_tokens_per_response']:.2f}, ensemble pool={z['ensemble_mean_unique_candidates']:.2f}", flush=True)
            name = f"dpo_minus_sft_{mode}_max5_reranker"
            c = comparisons[name]["metrics"]["mrr_at5"]
            t = transitions[name]["coverage_full_pool"]
            print(f"  MRR delta={c['difference']:+.6f}, CI={c['paired_bootstrap_95ci']}; full-pool gained={t['gained']}, lost={t['lost']}", flush=True)
        result_state["status"] = "complete"
        result_state["output_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    except BaseException as exc:
        result_state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", result_state)
    print(f"Saved: {output / 'comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--generation", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resamples", type=int, default=10000)
    args = p.parse_args()
    analyze(args.submission.resolve(), args.generation.resolve(), args.output.resolve(), args.resamples)
