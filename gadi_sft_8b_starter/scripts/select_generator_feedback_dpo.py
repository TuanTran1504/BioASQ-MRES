#!/usr/bin/env python3
"""Select focal-model DPO epochs on full internal validation with frozen companions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))
import src
import joblib
from prepare_matched_8b_sft import read, records, digest, write
from run_generator_feedback_dpo import validate_native
from generator_feedback_dpo_data import focal_model
from scripts.prepare_reranker_guided_dpo import load_inputs, configure_data_root
from scripts.train_sft_ensemble_reranker import build_pools
from scripts.ablate_reranker_relation_labels import verify_files
from scripts.evaluate_sft_ensemble_rerankers import rank_frozen
from scripts.compare_expansion_pool_rerankers import evaluate_rankings
from scripts.analyze_generator_feedback_evaluation import coverage_transitions
from src.notebook_workflows.coverage_comparison import official_candidate_matches
from run_base_original_prompt import full_input_plan, inference_worker
from run_extractive_expansion_8b import write_jsonl, parse_equivalent_response
from run_single_answer_sampling import sample_seed
from select_sft_epoch_checkpoints import epoch_checkpoints, stage_adapter


def choose(rows, model="llama31", require_individual_coverage=True):
    """Predeclared conservative rule; epoch zero is an eligible fallback."""
    base = next(r for r in rows if r["epoch"] == 0)
    prefix = "llama" if model == "llama31" else model
    eligible = []
    for r in rows:
        r["eligible"] = (r["ensemble_full_pool_lost"] == 0
                         and (not require_individual_coverage or r[f"{prefix}_full_pool_lost"] == 0)
                         and r["mrr_at5"] >= base["mrr_at5"] - 1e-12
                         and r["mean_output_tokens"] <= base["mean_output_tokens"] + 1e-12
                         and all(r["generation_quality"][k] <= base["generation_quality"][k]
                                 for k in ("empty_responses", "ceiling_responses", "surplus_responses")))
        if r["eligible"]:
            eligible.append(r)
    if not eligible:
        raise ValueError("Starting SFT must remain an eligible fallback")
    return max(eligible, key=lambda r: (r["mrr_at5"], -r["mean_output_tokens"], -r["epoch"]))


def generation_quality(rows):
    counts = dict(empty_responses=0, ceiling_responses=0, surplus_responses=0)
    candidate_count = 0
    for row in rows:
        try:
            answers, rejected, _, _ = parse_equivalent_response(row["raw_response"], 5)
        except ValueError:
            answers, rejected = [], []
        candidate_count += len(answers)
        counts["empty_responses"] += int(not answers)
        counts["ceiling_responses"] += int(row["output_tokens"] >= 512)
        counts["surplus_responses"] += int(any(r.get("reason") == "unique_candidate_limit_exceeded" for r in rejected))
    counts["mean_candidates_per_response"] = candidate_count / len(rows)
    return counts


def native_examples(data, generation, bank_rows):
    panels, aliases = data.panels(generation)
    examples = []
    for row in panels["validation"]:
        context = json.loads(row["messages"][1]["content"])
        examples.append({"question_id": row["question_id"], "question": context["question"],
            "snippets": [{"snippet_id": s["id"], "text": s["text"]} for s in context["snippets"]],
            "gold_aliases": aliases[row["question_id"]]})
    wanted = {e["question_id"] for e in examples}
    if len(examples) != 144 or len(wanted) != 144 or wanted & {r["question_id"] for r in panels["train"]}:
        raise ValueError("Expected 144 disjoint internal-validation questions")
    if any(r["question_id"] not in wanted for r in bank_rows if r["split"] == "validation"):
        raise ValueError("Companion validation membership differs")
    return examples


def projected_rows(rows, model="llama31"):
    """Convert saved worker draws back to native bank indices (greedy 0, samples 1..4)."""
    result = []
    for mode, generated in rows.items():
        for row in generated:
            for s in row["samples"]:
                result.append({"question_id": row["question_id"], "model_key": model,
                    "split": "validation", "arm": "max5", "draw": 0 if mode == "greedy" else s["draw"],
                    "raw_response": s["raw_response"], "output_tokens": s["output_tokens"]})
    return result


def verify_generated(path, examples, source, mode, plan):
    cfg = read(path / "config.json")
    count = 1 if mode == "greedy" else 4
    expected = {"model_name": source["adapter"], "num_generations": count, "max_candidates": 5,
        "temperature": 0 if count == 1 else .8, "seed": 3407, "sample_seed_offset": 0 if count == 1 else 1,
        "max_new_tokens": 512, "require_all_snippets": True, "mark_snippets": False,
        "response_mode": "equivalent_greedy" if count == 1 else "equivalent_sampling",
        "top_p": .95, "top_k": 0, "model_loader": source["model_loader"],
        "chat_template_kwargs": source["chat_template_kwargs"], "execution_mode": source.get("execution_mode", "default"),
        "max_seq_length": plan["max_seq_length"], "expected_prompt_ids_sha256": plan["expected_prompt_ids_sha256"],
        "expected_prompt_tokens": plan["expected_prompt_tokens"]}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Validation generation source, seeds or input plan differs")
    rows = records(path / "generations.jsonl")
    wanted = {e["question_id"]: e for e in examples}
    if (read(path / "status.json")["status"] != "complete" or len(rows) != len(wanted)
            or {r["question_id"] for r in rows} != set(wanted)
            or records(path / "examples.jsonl") != [{k: e[k] for k in ("question_id", "question", "snippets")} for e in examples]):
        raise ValueError("Missing, changed or duplicate validation generation")
    for row in rows:
        qid = row["question_id"]
        if (len(row["samples"]) != count or row["snippets_truncated"]
                or row["included_snippets"] != len(wanted[qid]["snippets"])
                or row["included_snippets"] != row["total_snippets"]):
            raise ValueError("Validation dropped evidence or sampled draws")
        for i, s in enumerate(row["samples"], 1):
            expected_seed = sample_seed(qid, i + (0 if mode == "greedy" else 1), 3407)
            if s["draw"] != i or s["seed"] != expected_seed:
                raise ValueError("Validation sample seed differs from baseline bank")
            try:
                aa = parse_equivalent_response(s["raw_response"], 5)[0]
            except ValueError:
                aa = []
            if aa != s["answers"]:
                raise ValueError("Validation parsed answers differ from raw responses")
    return rows


def select(submission, training, output, smoke=False, *, native_validator=None,
           checkpoint_chooser=None, pool_builder=None, preserve_surfaces=False, jar_path_override=None):
    if output.exists():
        raise FileExistsError(output)
    prepared = read(submission / "prepared.json")
    verify_files(submission / "code", prepared["code_hashes"])
    repository = Path(prepared["repository"])
    config = prepared["configuration"]
    focal = focal_model(config)
    coverage_protocol = native_validator is not None
    validator = native_validator or validate_native
    chooser = checkpoint_chooser or choose
    builder = pool_builder or build_pools
    expected_scope = "official_form" if coverage_protocol else "all_compactness"
    if config.get("review_scope") != expected_scope or not config.get("retain_epoch_checkpoints"):
        raise ValueError("Full validation selection requires the expanded retained-epoch protocol")
    manifest, splits, source, feedback = validator(Path(prepared["preferences"]), config, repository)
    state = read(training / "manifest.json")
    expected_counts = {s: min(4, len(rr)) if smoke else len(rr) for s, rr in splits.items()}
    if (state["status"] != "complete" or state["smoke_test"] != smoke or state["source"] != source
            or state["configuration"] != config or state["prepared_sha256"] != digest(submission / "prepared.json")
            or feedback["banks"] != prepared["native_banks"] or state["phase"] != "save_adapter"
            or state["trained_pair_counts"] != expected_counts or state["global_steps"] <= 0):
        raise ValueError("Training/selection source differs from prepared expanded preferences")
    verify_files(training / "adapter", state["adapter_files_sha256"])
    data = configure_data_root(repository)
    generation = read(data.DEFAULT_CONFIG)
    banks, raw = data.pack_banks([Path(feedback["banks"][m]["path"]) for m in data.MODELS], generation)
    examples = native_examples(data, generation, raw)
    rerankers = Path(feedback["rerankers"])
    audit_config = read(repository / "gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json")
    _, _, _, _, hashes, bank_hashes, _ = load_inputs(repository, rerankers, audit_config)
    if hashes != feedback["reranker_hashes"] or bank_hashes != feedback["banks"]:
        raise ValueError("Selection reranker differs from frozen feedback model")
    reranker_state = read(rerankers / "manifest.json")
    metadata = reranker_state["selectors"]["max5"]
    payload = joblib.load(rerankers / metadata["model_path"])
    if payload["selector"] != "max5" or payload["schema"] != "sft-ensemble-logistic-v1":
        raise ValueError("Unexpected selector")
    plan = full_input_plan(source, examples, 512)
    if plan["max_seq_length"] > generation["max_seq_length"]:
        raise ValueError("Complete validation prompts exceed the original bank allocation")
    plan["max_seq_length"] = generation["max_seq_length"]
    if smoke:
        examples = sorted(examples, key=lambda e: (-plan["expected_prompt_tokens"][e["question_id"]], e["question_id"]))[:4]
    ids = {e["question_id"] for e in examples}
    baseline = [{**r, "arm": "max5"} for r in raw if r["split"] == "validation" and r["question_id"] in ids]
    expected_messages = {e["question_id"]: [{"role": "system", "content": source["system_prompt"]},
        {"role": "user", "content": json.dumps({"question": e["question"], "snippets": [
            {"id": s["snippet_id"], "text": s["text"]} for s in e["snippets"]]}, ensure_ascii=False)}] for e in examples}
    for row in baseline:
        if row["model_key"] == focal and (row["prompt"] != expected_messages[row["question_id"]]
                or row["prompt_tokens"] != plan["expected_prompt_tokens"][row["question_id"]]):
            raise ValueError("Starting-SFT bank prompt or token count differs from epoch validation input")
    output.mkdir(parents=True, exist_ok=False)
    selected_state = {"status": "running", "smoke_test": smoke, "validation_question_count": len(examples),
        "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json"),
        "training": str(training), "training_manifest_sha256": digest(training / "manifest.json"),
        "reranker_hashes": hashes, "question_ids": [e["question_id"] for e in examples],
        "focal_model": focal,
        "selection_rule": (config["checkpoint_selection"] if coverage_protocol else
            f"sampling4: no ensemble/{focal} full-pool losses; MRR >= SFT; mean {focal} tokens <= SFT; no additional empty, ceiling or surplus responses; maximize MRR then minimize tokens then earliest epoch; epoch zero fallback"),
        "pool_policy": "verbatim surfaces" if preserve_surfaces else "historical normalized surfaces"}
    write(output / "manifest.json", selected_state)
    try:
        write_jsonl(output / "validation_inputs.jsonl", [{k: e[k] for k in ("question_id", "question", "snippets")} for e in examples])
        (output / "prompt.txt").write_text(source["system_prompt"], encoding="utf-8")
        checkpoints = ([{"epoch": 1, "path": training / "adapter", "global_step": state["global_steps"], "eval_loss": state["best_validation_loss"]}]
                       if smoke else epoch_checkpoints(training, int(config["num_train_epochs"])))
        for c in checkpoints:
            if not smoke:
                c["path"] = Path(c["path"])
        alternatives = {0: baseline}
        adapter_paths = {0: Path(source["adapter"])}
        generation_hashes = {}
        for checkpoint in checkpoints:
            epoch = checkpoint["epoch"]
            print(f"Validation generation: {focal}, epoch {epoch}, greedy plus four samples/question", flush=True)
            root = output / f"epoch-{epoch}"
            root.mkdir()
            adapter = root / "adapter"
            stage_adapter(checkpoint["path"], training / "adapter", adapter)
            adapter_paths[epoch] = adapter
            current = {**source, "adapter": str(adapter)}
            candidate_plan = full_input_plan(current, examples, 512)
            for field in ("expected_prompt_tokens", "expected_prompt_ids_sha256"):
                if any(candidate_plan[field][q] != plan[field][q] for q in ids):
                    raise ValueError("Epoch tokenizer differs from initial SFT")
            rows = {}
            for mode in ("greedy", "sampling4"):
                settings = {"input": str(output / "validation_inputs.jsonl"), "prompt": str(output / "prompt.txt"),
                    "response_mode": "equivalent_greedy" if mode == "greedy" else "equivalent_sampling",
                    "num_generations": 1 if mode == "greedy" else 4, "temperature": 0 if mode == "greedy" else .8,
                    "sample_seed_offset": 0 if mode == "greedy" else 1, "top_p": .95, "top_k": 0,
                    "seed": 3407, "max_new_tokens": 512, "max_candidates": 5, "require_all_snippets": True,
                    "mark_snippets": False, "max_seq_length": plan["max_seq_length"],
                    "expected_prompt_tokens": plan["expected_prompt_tokens"], "expected_prompt_ids_sha256": plan["expected_prompt_ids_sha256"],
                    "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
                    "execution_mode": source.get("execution_mode", "default"), "prompt_version": "expanded-dpo-validation144-v1"}
                cfg = root / f"{mode}_config.json"; write(cfg, settings)
                subprocess.run([sys.executable, str(inference_worker(submission, prepared)), "--config", str(cfg),
                    "--model-name", str(adapter), "--output-dir", str(root / mode), "--limit", str(len(examples)), "--preflight-all"], check=True)
                rows[mode] = verify_generated(root / mode, examples, current, mode, plan)
                generation_hashes[f"epoch-{epoch}/{mode}"] = {p.name: digest(p) for p in (root / mode).iterdir() if p.is_file()}
            alternatives[epoch] = [r for r in baseline if r["model_key"] != focal] + projected_rows(rows, focal)
        contexts = {e["question_id"]: e for e in examples}
        solo_prefix = "llama" if focal == "llama31" else focal
        pools = {}
        for epoch, rows in alternatives.items():
            pp, _ = builder(rows, {"validation": contexts})
            for mode in ("greedy", "sampling4"):
                pools[f"epoch{epoch}_{mode}"] = pp["validation"][f"max5_{mode}"]
                solo, _ = builder([r for r in rows if r["model_key"] == focal], {"validation": contexts})
                pools[f"{solo_prefix}_epoch{epoch}_{mode}"] = solo["validation"][f"max5_{mode}"]
        distinct = {(r["question_id"], r["answer"]): r for rr in pools.values() for r in rr}
        (output / "official").mkdir()
        jar = jar_path_override or repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
        matches = official_candidate_matches(examples, list(distinct.values()), output / "official", jar_path=jar) if distinct else {}
        normalized = {}
        from scripts.train_candidate_reranker_pilot import key
        for (qid, answer), accepted in matches.items():
            identity = (qid, key(answer))
            if not preserve_surfaces and identity in normalized and normalized[identity] != accepted:
                raise ValueError("Pool deduplication changes official acceptance")
            normalized[identity] = accepted
        for rr in pools.values():
            for r in rr:
                r["label"] = int(matches[(r["question_id"], r["answer"])])
        ensemble_pools = {n: rr for n, rr in pools.items() if not n.startswith(solo_prefix+"_")}
        conditions, scores, rankings = rank_frozen(payload, ensemble_pools, contexts)
        for n, rr in pools.items():
            if n.startswith(solo_prefix+"_"):
                groups = {q: [] for q in ids}
                for r in rr: groups[r["question_id"]].append(r)
                metrics, perq = evaluate_rankings(groups)
                conditions[n] = {"metrics": metrics}; scores[n] = perq
        selection_rows = []
        epoch_metadata = {c["epoch"]: {"validation_preference_loss": c["eval_loss"], "global_step": c["global_step"]} for c in checkpoints}
        base = scores["epoch0_sampling4_max5_reranker"]
        solo_base = scores[f"{solo_prefix}_epoch0_sampling4"]
        for epoch, rows in alternatives.items():
            current = scores[f"epoch{epoch}_sampling4_max5_reranker"]
            solo = scores[f"{solo_prefix}_epoch{epoch}_sampling4"]
            calls = [r for r in rows if r["model_key"] == focal and r["draw"] > 0]
            if len(calls) != len(examples)*4:
                raise ValueError("Incomplete validation sampling budget")
            selection_rows.append({"epoch": epoch, "mrr_at5": conditions[f"epoch{epoch}_sampling4_max5_reranker"]["metrics"]["mrr_at5"],
                **epoch_metadata.get(epoch, {"validation_preference_loss": None, "global_step": 0}),
                "mean_output_tokens": sum(r["output_tokens"] for r in calls)/len(calls),
                "generation_quality": generation_quality(calls),
                "ensemble_full_pool_lost": coverage_transitions(base, current, "coverage_full_pool")["lost"],
                "ensemble_full_pool_coverage": conditions[f"epoch{epoch}_sampling4_max5_reranker"]["metrics"]["coverage_full_pool"],
                "focal_full_pool_coverage": conditions[f"{solo_prefix}_epoch{epoch}_sampling4"]["metrics"]["coverage_full_pool"],
                f"{solo_prefix}_full_pool_lost": coverage_transitions(solo_base, solo, "coverage_full_pool")["lost"],
                "coverage_transitions": {k: coverage_transitions(base, current, k) for k in ("coverage_at5", "coverage_full_pool")}})
        winner = chooser(selection_rows, solo_prefix)
        # An explicitly separate diagnostic does not replace the strict winner.
        relaxed_rows = [{**r, "generation_quality": dict(r["generation_quality"])} for r in selection_rows]
        ensemble_candidate = chooser(relaxed_rows, solo_prefix, require_individual_coverage=False)
        shutil.copytree(adapter_paths[winner["epoch"]], output / "selected_adapter")
        write(output / "comparison_summary.json", {"question_count": len(examples), "conditions": conditions,
            "checkpoint_scores": selection_rows, "selected_epoch": winner["epoch"], "selection_rule": selected_state["selection_rule"],
            "ensemble_only_candidate_epoch": ensemble_candidate["epoch"],
            "ensemble_only_candidate_is_selected": ensemble_candidate["epoch"] == winner["epoch"],
            "ensemble_only_candidate_rule": ("Not applicable to coverage-primary selection; same winner." if coverage_protocol else
                "Diagnostic only: same guards except individual-model full-pool loss; selected_adapter remains strict winner."),
            "pool_policy": selected_state["pool_policy"],
            "limitations": ["Internal validation already selected the frozen reranker C; these are selection diagnostics, not held-out performance.",
                "Four samples per model, matching the original feedback bank seed indices. One seed; no semantic judging of new generated answers."]})
        write(output / "comparison_per_question.json", scores)
        write_jsonl(output / "rankings.jsonl", rankings)
        verify_files(submission / "code", prepared["code_hashes"])
        verify_files(rerankers, hashes)
        validator(Path(prepared["preferences"]), config, repository)
        if digest(training / "manifest.json") != selected_state["training_manifest_sha256"]:
            raise ValueError("Training metadata changed during validation selection")
        verify_files(training / "adapter", state["adapter_files_sha256"])
        if not smoke:
            for checkpoint in checkpoints:
                verify_files(checkpoint["path"], checkpoint["adapter_sha256"])
        for path, expected in generation_hashes.items():
            verify_files(output / path, expected)
        selected_state.update(status="complete", selected_epoch=winner["epoch"],
            selected_adapter_sha256={p.name: digest(p) for p in (output / "selected_adapter").iterdir() if p.is_file()},
            generation_hashes=generation_hashes,
            output_sha256={p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"})
    except BaseException as exc:
        selected_state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", selected_state)
    print(f"Validation selected epoch {selected_state['selected_epoch']} (0 means keep starting SFT). Results: {output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--training", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    select(a.submission.resolve(), a.training.resolve(), a.output.resolve(), a.smoke_test)
