"""Small, editable experiment variants; algorithms remain in shared modules."""
import copy
from .runner import defaults

# (workflow method, overrides). Model size is a parameter, never a copied notebook.
PRESETS = {
    "data": {
        "snippet_supported": ("snippet_split", {}),
        "supported_dev_legacy": ("snippet_split", {"split_order": "filter_first"}),
        "literal_supported": ("snippet_split", {"filter_mode": "literal"}),
        "full_resources": ("snippet_split", {"filter_mode": "none"}),
        "one_supported_alias": ("snippet_split", {"alias_mode": "first"}),
        "prepared_train_dev": ("prepared_split", {}),
        "reserve_dpo_questions": ("sft_dpo_split", {}),
        "occurrence_evidence": ("occurrence_export", {}),
        "judge_evidence": ("evidence_annotation", {}),
        "export_judged_evidence": ("evidence_annotation", {"phase": "export"}),
        "dev_occurrence": ("dev_evidence", {}),
        "dev_judge": ("dev_evidence", {"phase": "annotate"}),
        "dev_export": ("dev_evidence", {"phase": "export"}),
    },
    "generation": {
        "sample20": ("sample_candidates", {}),
        "sample10": ("sample_candidates", {"samples_per_question_total": 10}),
        "sample5": ("sample_candidates", {"samples_per_question_total": 5}),
        "literal_copy": ("sample_candidates", {"prompt": "factoid-single-answer-literal-copy-v1"}),
        "score_logprobs": ("score_candidates", {}),
        "filter": ("filter_candidates", {}),
        "merge_models": ("merge_candidates", {}),
    },
    "pairs": {
        "judge_sample20": ("judge_candidates", {}),
        "judge_sample10": ("judge_candidates", {"expected_samples": 10}),
        "standard_factoid": ("wrong_entity_pairs", {}),
        "all_factoid_negatives": ("wrong_entity_pairs", {"wrongs_per_question": 1000000}),
        "extractive_spans": ("span_pairs", {}),
        "three_stage_curriculum": ("curriculum_split", {}),
        "stage2_ties": ("tie_dataset", {}),
        "stage2_alignment": ("alignment_filter", {}),
        "whole_response_list": ("whole_response_pairs", {}),
        "gold_response_list": ("gold_response_pairs", {}),
        "filter": ("filter_pairs", {}),
    },
    "sft": {
        "answer_only": ("answer_sft", {}),
        "evidence_plus_answer": ("evidence_sft", {"arm": "evidence"}),
        "evidence_answer_control": ("evidence_sft", {"arm": "answer"}),
        "rationale_from_base": ("rationale_sft", {"from_base": True}),
        "rationale_continued": ("rationale_sft", {"from_base": False, "learning_rate": 2e-5}),
    },
    "dpo": {
        "standard": ("standard_dpo", {}),
        "stage1": ("staged_dpo", {"stop_after_stage": "concept_learning"}),
        "two_stage": ("staged_dpo", {}),
        "three_stage": ("staged_dpo", {"stop_after_stage": "hierarchical_ranking"}),
        "stage2_retention": ("staged_dpo", {
            "objective": "dpo_d", "skip_stages": ["concept_learning"],
            "stop_after_stage": "format_alignment", "selection_metric": "retention_score",
            "learning_rate": 2e-7, "beta": .05, "optimizer": "AdamW", "weight_decay": 0.,
            "retention_anchor_weight": 1., "retention_anchor_max_examples": 128,
            "l2sp_weight": .1, "retention_min_rate": .98, "epochs": 4,
        }),
        "cal_dpo": ("staged_dpo", {"objective": "cal_dpo"}),
        "apo_zero": ("staged_dpo", {"objective": "apo_zero"}),
        "adaptive_nll": ("staged_dpo", {"objective": "dpo_adaptive_nll"}),
        "softmax": ("orbit_dpo", {"method": "softmax_dpo"}),
        "representative": ("orbit_dpo", {}),
        "error_aware": ("error_aware_dpo", {}),
    },
    "evaluation": {
        "greedy": ("local_evaluation", {}),
        "sample10_frequency": ("local_evaluation", {
            "num_generations": 10, "aggregation_strategy": "frequency",
            "aggregation_min_frequency": 1, "do_sample": True, "temperature": .7, "top_p": .9,
        }),
        "sample5_union": ("local_evaluation", {
            "num_generations": 5, "aggregation_strategy": "union",
            "do_sample": True, "temperature": .7, "top_p": .9,
        }),
        "direct_top5": ("local_evaluation", {"prompt": "factoid-top-five-eval-v1"}),
        "compare_sampling": ("sampling_comparison", {}),
        "history_conditioned": ("conditioned_sampling", {}),
        "gpt_direct_vs_reasoning": ("gpt_reasoning", {}),
        "openai_direct": ("openai_direct", {}),
    },
    "analysis": {
        "compare_banks": ("compare_banks", {}),
        "test_evidence_coverage": ("evidence_coverage", {}),
        "dpo_diagnostics": ("dpo_diagnostics", {}),
        "stage1_training_audit": ("stage1_audit", {}),
    },
    "synthetic": {
        "prepare": ("synthetic_qa", {"phase": "prepare"}),
        "generate": ("synthetic_qa", {"phase": "generate"}),
        "verify": ("synthetic_qa", {"phase": "verify"}),
        "finalize": ("synthetic_qa", {"phase": "finalize"}),
    },
}


def configuration(category, preset, overrides=None):
    method, changes = PRESETS[category][preset]
    return method, {**defaults(method), **copy.deepcopy(changes), **copy.deepcopy(overrides or {})}
