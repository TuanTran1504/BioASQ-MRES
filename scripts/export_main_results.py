"""Export aggregate results from local artifacts without datasets or credentials."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SFT = "Artifacts/notebook_runs/sft/answer_sft"
EVAL = "Artifacts/notebook_runs/evaluation/local_evaluation/20260929-133024-48a9b350/evaluation"
DPO = "Artifacts/notebook_runs/dpo/staged_dpo"
COVERAGE = "Artifacts/notebook_runs/coverage_comparison"


def main():
    sources = []

    def read(relative):
        payload = (ROOT / relative).read_bytes()
        sources.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest()})
        return json.loads(payload)

    results = {"schema_version": 1, "snapshot_date": "2026-10-06", "factoid_dev_questions": 160}
    results["sft"] = []
    for model, run in [("Qwen2.5-0.5B", "20260927-143545-67f7a338"),
                       ("Qwen2.5-3B", "20260927-143545-3fca1c3d")]:
        evaluated = read(f"{EVAL}/{run}/scores.json")
        selected = read(f"{SFT}/{run}/generated_dev_selection/best_summary.json")
        results["sft"].append({
            "model": model, "run_id": run,
            "standalone_evaluation": {
                "created_at": evaluated["created_at"],
                "metrics": evaluated["aggregate"]["by_type"]["factoid"],
                "semantic": evaluated["grounded_semantic"],
                "generation": evaluated["generation"],
                "evidence_settings": {k: evaluated["dataset"][k] for k in
                                      ("max_resources", "max_resource_chars", "resource_selection")},
                "prompt_id": evaluated["prompt"]["prompt_id"],
                "scoring_backend": evaluated["scoring"]["selected_backend"],
            },
            "training_checkpoint_selection": {
                k: selected[k] for k in ("step", "epoch", "metric_value", "question_count", "generation_protocol")
            },
        })

    run = "20260929-140224-fa5a3f25"
    history = read(f"{DPO}/{run}/training/stage_1_concept_learning/eval_history.json")
    manifest = read(f"{DPO}/{run}/training/stage_1_concept_learning/manifest.json")
    config = read(f"{DPO}/{run}/training/config.json")["config"]
    fields = ("epoch", "step", "dev_eval_question_count", "dev_mrr", "dev_strict_accuracy",
              "dev_lenient_accuracy", "baseline_correct_count", "retained_correct_count",
              "newly_correct_count", "lost_correct_count", "generation_protocol", "scoring_backend")
    results["factoid_dpo"] = {
        "model": "Qwen2.5-0.5B", "run_id": run,
        "status": read(f"{DPO}/{run}/status.json")["status"],
        "best_dev_mrr": manifest["best_dev_mrr"],
        "selected_step": int(Path(manifest["best_selected_adapter"].replace("\\", "/")).name.split("_")[-1]),
        "configuration": {k: config[k] for k in ("seed", "beta", "objective", "learning_rate",
                              "optimizer", "epochs", "batch_size", "gradient_accumulation",
                              "max_length", "selection_metric", "stop_after_stage")},
        "history": [{k: row[k] for k in fields} for row in history],
        "matched_baseline_mrr": history[0]["dev_mrr"],
        "best_minus_matched_baseline_mrr": round(manifest["best_dev_mrr"] - history[0]["dev_mrr"], 8),
    }
    run3 = "20260929-140224-46d6d62c"
    results["factoid_dpo_3b"] = {
        "model": "Qwen2.5-3B", "run_id": run3,
        "saved_status": read(f"{DPO}/{run3}/status.json")["status"],
        "evaluation_available": False,
        "note": "No saved evaluation history or completed summary was found at snapshot time; saved status is not live process status.",
    }
    if (ROOT / DPO / run3 / "training/stage_1_concept_learning/eval_history.json").exists():
        raise RuntimeError("3B evaluation is now available; update the exporter before publishing.")

    results["coverage_pilots"] = {}
    for label, relative in {
        "gpt41mini_equivalent_vs_sampling": f"{COVERAGE}/20260929-070458-a6267ca4/analysis-59ac5450/summary.json",
        "gpt41mini_extractive_v1_vs_sampling": f"{COVERAGE}/20261001-010109-2d745f42/analysis-21e98a02/summary.json",
        "gpt41mini_extractive_v2_vs_sampling": f"{COVERAGE}/20261001-081220-082b3f96/analysis-c1d65d52/summary.json",
        "llama31_8b_equivalent": "Artifacts/gadi_runs/llama31_8b_equivalent_dev160/analysis-c672a1c9/summary.json",
        "qwen3_8b_equivalent": "Artifacts/gadi_runs/qwen3_8b_equivalent_dev160/analysis-795d7fb5/summary.json",
    }.items():
        results["coverage_pilots"][label] = read(relative)
    reranker = read("Artifacts/reranker_pilot/20261002-023517-tfidf-logistic/summary.json")
    results["reranker_pilot"] = {k: reranker[k] for k in
        ("status", "method", "evaluation", "warning", "question_count", "candidate_count",
         "positive_candidate_count", "questions_with_positive_candidate", "format_variants", "folds", "metrics")}
    results["historical_list_dpo"] = read("reproducibility/best_list_system/expected_result.json")
    results["interpretation_notes"] = [
        "Factoid results are development pilots, not unseen-test confirmation.",
        "SFT checkpoint-selection and standalone evaluations differ; do not substitute one for the other.",
        "Use DPO step zero as the matched baseline; its 0.40000 differs from standalone SFT 0.40625.",
        "Candidate coverage at ten is an offline diagnostic, not an official ten-answer submission or final MRR.",
        "Reranker scores use five-fold question-grouped development cross-validation; the final fitted model has no independent score.",
        "LLM semantic labels require auditing. No semantic score is available here for factoid DPO.",
    ]
    results["sources"] = sources
    output = ROOT / "results/main_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(sources)} source records to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
