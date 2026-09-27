"""Workflow methods reuse the repository's trainers, scorers and data builders.

Output arguments are owned by the runner. Input arguments are checked before
execution. CLI schemas are read from source without importing GPU libraries.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    category: str
    description: str
    module: str = ""
    schema_module: str = ""
    operation: str = ""
    parameters: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    inputs: tuple = ()
    required: tuple = ()
    gpu: bool = False
    api: bool = False


PROMPT = {
    "chat_template": "qwen-2.5", "prompt_format": "chat",
    "prompt_file": "prompts/factoid_single_answer_aligned.json",
    "prompt": "factoid-single-answer-extractive-v1",
    "max_resources": 0, "max_resource_chars": 0, "max_seq_length": 4096,
}
TEST = [f"data/Task13BTest/13B{i}_golden.json" for i in range(1, 5)]

METHODS = {}


def add(name, category, description, **kwargs):
    METHODS[name] = Method(category, description, **kwargs)


add("snippet_split", "data", "Split all factoids, keep dev unfiltered, filter training and expand its aliases.",
    module="scripts.prepare_factoid_snippet_sft",
    parameters={"input": "data/training13b.json", "test_input": TEST, "dev_ratio": .1,
                "seed": 3407, "filter_mode": "normalized", "alias_mode": "per_alias", "split_order": "split_first"},
    outputs={"output_dir": "data"}, inputs=("input", "test_input"))
add("prepared_split", "data", "Split an existing prepared dataset by question ID.",
    module="src.utility.split_prepared_dataset",
    parameters={"input": None, "validation_ratio": .1, "seed": 3407, "group_by": "id"},
    outputs={"output_dir": "data"}, inputs=("input", "reference_dev_input"), required=("input",))
add("sft_dpo_split", "data", "Reserve disjoint SFT/DPO questions, stratified by answer/context features.",
    module="cse_dpo.split_gold_supported_sft_dpo",
    parameters={"question_source": None, "expanded_alias_source": None, "sft_fraction": .8, "seed": 3407},
    outputs={"output_dir": "data"}, inputs=("question_source", "expanded_alias_source"),
    required=("question_source", "expanded_alias_source"))
add("occurrence_export", "data", "Build evidence-plus-answer and answer-only records from prepared questions.",
    operation="occurrence_export", parameters={"source": None, "match_mode": "normalized", "all_aliases": True},
    inputs=("source",), required=("source",))
add("evidence_annotation", "data", "Judge alias evidence, or export existing judgments; explicit API budget.",
    operation="evidence_annotation",
    parameters={"source": None, "phase": "annotate", "annotation_dir": None,
                "api_key_file": "open_ai_api.txt", "model": "gpt-4.1-mini-2025-04-14",
                "max_new_calls": 10, "seed": 3407},
    inputs=("source", "annotation_dir"), required=("source",), api=True)
add("dev_evidence", "data", "Create separate v2 dev occurrence/semantic references for evidence SFT evaluation.",
    operation="dev_evidence",
    parameters={"source": None, "train_source": None, "phase": "occurrence", "annotation_dir": None,
                "api_key_file": "open_ai_api.txt", "model": "gpt-4.1-mini-2025-04-14", "max_new_calls": 10},
    inputs=("source", "train_source", "annotation_dir"), required=("source", "train_source"), api=True)

add("sample_candidates", "generation", "Sample one or more models using a shared prompt and source split.",
    module="cse_dpo.generate_candidate_bank",
    parameters={**PROMPT, "eval_input": None, "model_ref": None, "question_types": ["factoid"],
                "samples_per_question_total": 20, "temperature": .7, "top_p": .9,
                "max_new_tokens": 64, "seed": 3407, "batch_size": 1},
    outputs={"output_dir": "candidates"}, inputs=("eval_input", "prompt_file"),
    required=("eval_input", "model_ref"), gpu=True)
add("score_candidates", "generation", "Score candidate completion log probabilities, with optional reference.",
    module="cse_dpo.score_candidate_bank",
    parameters={"candidate_input": None, "model_ref": None, "max_seq_length": 4096},
    outputs={"output_jsonl": "scored_candidates.jsonl", "summary_json": "summary.json"},
    inputs=("candidate_input",), required=("candidate_input", "model_ref"), gpu=True)
add("filter_candidates", "generation", "Filter existing samples by sample ID and parser status.",
    module="cse_dpo.filter_candidate_bank_rows", parameters={"input_jsonl": None},
    outputs={"output_jsonl": "candidates.jsonl", "summary_json": "summary.json"},
    inputs=("input_jsonl",), required=("input_jsonl",))
add("merge_candidates", "generation", "Merge banks with provenance and unique response IDs.",
    operation="merge_candidates", parameters={"banks": {}}, required=("banks",))

add("judge_candidates", "pairs", "Classify exact/equivalent/incorrect answers and build curriculum pairs.",
    module="cse_dpo.annotate_split_dpo_candidate_bank",
    parameters={"bank": None, "questions": None, "judge_model": "gpt-4.1-mini-2025-04-14",
                "api_key_file": "open_ai_api.txt", "max_new_judge_calls": 10,
                "request_delay_seconds": 1.0, "expected_samples": 20, "source_model": None,
                "previous_judgments": None},
    outputs={"output_root": "judgments"}, inputs=("bank", "questions"),
    required=("bank", "questions"), api=True)
add("wrong_entity_pairs", "pairs", "Construct standard factoid gold-versus-sampled-negative pairs.",
    module="cse_dpo.construct_factoid_label_wrong_entity_bank",
    parameters={"question_input": None, "candidate_bank_jsonl": None, "wrongs_per_question": 4,
                "allow_fewer_within_question_wrongs": True, "seed": 3407},
    outputs={"output_jsonl": "classes.jsonl", "pair_output_jsonl": "pairs.jsonl", "summary_json": "summary.json"},
    inputs=("question_input", "candidate_bank_jsonl"), required=("question_input", "candidate_bank_jsonl"))
add("span_pairs", "pairs", "Extract factoid answer-boundary/span preference data.",
    module="cse_dpo.extract_factoid_span_pairs_from_candidate_bank",
    parameters={"candidate_bank": None, "prepared_json": None, "split": "train"},
    outputs={"output_dir": "span_pairs"}, inputs=("candidate_bank", "prepared_json"),
    required=("candidate_bank", "prepared_json"))
for name, module, description in [
    ("whole_response_pairs", "construct_whole_response_dpo_pairs", "List-question whole-response F1 preference pairs."),
    ("gold_response_pairs", "construct_gold_response_dpo_pairs", "List-question gold-response preference pairs."),
]:
    add(name, "pairs", description, module="cse_dpo." + module,
        parameters={"question_input": None, "candidate_input": None, "max_resources": 0,
                    "max_resource_chars": 0, "seed": 3407},
        outputs={"output_jsonl": "pairs.jsonl", "summary_json": "summary.json", "manual_audit_md": "audit.md"},
        inputs=("question_input", "candidate_input"), required=("question_input", "candidate_input"))
add("curriculum_split", "pairs", "Split C3>C1, C3>C2 and C2>C1 pairs into curriculum stages.",
    operation="curriculum_split", parameters={"pair_file": None}, inputs=("pair_file",), required=("pair_file",))
add("tie_dataset", "pairs", "Build the tie-aware Stage 2 dataset for DPO-D.",
    module="cse_dpo.build_stage2_tie_aware_dataset", parameters={"input_root": None},
    outputs={"output_root": "staged"}, inputs=("input_root",), required=("input_root",))
add("filter_pairs", "pairs", "Filter by class, metric separation, cardinality, or edit distance.",
    module="cse_dpo.filter_preference_pairs", parameters={"input_jsonl": None},
    outputs={"output_jsonl": "pairs.jsonl", "summary_json": "summary.json"},
    inputs=("input_jsonl",), required=("input_jsonl",))
add("alignment_filter", "pairs", "Rank Stage 2 alignment potential; optionally use model log probabilities.",
    module="cse_dpo.filter_stage2_alignment_potential",
    parameters={"source_staged_root": None, "run_root": None, "model_preset": "qwen25_05b",
                "select_ratio": .3, "max_length": 4096},
    outputs={"output_root": "filtered"}, inputs=("source_staged_root", "run_root"),
    required=("source_staged_root", "run_root"), gpu=True)

add("answer_sft", "sft", "Answer-only LoRA SFT for any supported model, using explicit train/dev files.",
    module="src.utility.answer_gen_ft", schema_module="src.utility.config",
    parameters={**PROMPT, "train_input": None, "eval_input": None,
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct", "question_types": ["factoid"],
                "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 32, "warmup_steps": 5,
                "num_train_epochs": 6., "learning_rate": 6e-4, "weight_decay": .01,
                "lora_r": 32, "lora_alpha": 32, "lora_dropout": .05, "dataset_num_proc": 1,
                "selection_metric": "generated_primary_score", "selection_max_new_tokens": 64,
                "selection_max_seq_length": 4096, "save_strategy": "steps", "save_steps": 25,
                "early_stopping_patience": 3, "early_stopping_threshold": .001, "seed": 3407},
    outputs={"output_dir": "trainer_output", "save_model_dir": "adapter",
             "prepared_output_dir": "trainer_prepared", "artifacts_root": "managed",
             "registry_path": "model_registry.json"},
    inputs=("train_input", "eval_input", "prompt_file"), required=("train_input", "eval_input"), gpu=True)
add("evidence_sft", "sft", "Matched evidence-plus-answer / answer-only SFT arms from annotation exports.",
    module="cse_dpo.train_factoid_evidence_answer_sft",
    parameters={"variant": "occurrence", "arm": "evidence", "export_dir": None,
                "base_model": None, "initial_adapter": None, "dev_input": None,
                "dev_evidence_export": None, "epochs": 6., "learning_rate": 5e-5,
                "grad_accum": 32, "seed": 3407},
    outputs={"output_dir": "training"},
    inputs=("export_dir", "base_model", "initial_adapter", "dev_input", "dev_evidence_export"),
    required=("export_dir", "base_model", "dev_input", "dev_evidence_export"), gpu=True)
add("rationale_sft", "sft", "Controlled from-base or continued rationale SFT with optional answer replay.",
    operation="rationale_sft",
    parameters={"base_model": None, "initial_adapter": None, "train_source": None,
                "dev_source": None, "rationale_bank": None, "from_base": True,
                "epochs": 8., "learning_rate": 6e-4, "max_length": 4096, "seed": 3407,
                "replay_fraction": .2},
    inputs=("base_model", "initial_adapter", "train_source", "dev_source", "rationale_bank"),
    required=("base_model", "initial_adapter", "train_source", "dev_source", "rationale_bank"), gpu=True)

add("standard_dpo", "dpo", "TRL pairwise DPO, with configurable loss and question-level validation.",
    module="src.dpo_train",
    parameters={"preference_input": None, "model_name": None, "beta": .1, "loss_type": "sigmoid",
                "split_by": "question", "validation_ratio": .1, "seed": 3407,
                "learning_rate": 5e-6, "num_train_epochs": 1., "max_seq_length": 4096,
                "max_prompt_length": 3584, "max_completion_length": 512},
    outputs={"output_dir": "trainer_output", "save_model_dir": "adapter"},
    inputs=("preference_input", "selection_eval_input"),
    required=("preference_input", "model_name"), gpu=True)
add("staged_dpo", "dpo", "One/two/three-stage curriculum; DPO, DPO-D, APO, adaptive NLL or Cal-DPO.",
    operation="staged_dpo",
    parameters={"model_preset": "qwen25_05b", "base_model": None, "initial_adapter": None,
                "staged_root": None, "dev_source_input": None, "objective": "dpo",
                "stop_after_stage": "format_alignment", "skip_stages": [],
                "stage_settings": {}, "seed": 3407, "smoke_test": True,
                "max_length": 4096, "generated_eval_max_seq_length": 4096,
                "semantic_judge_enabled": False},
    inputs=("initial_adapter", "staged_root", "dev_source_input"),
    required=("base_model", "initial_adapter", "staged_root", "dev_source_input"), gpu=True)
add("orbit_dpo", "dpo", "Representative, softmax, permutation and class-level preference objectives.",
    module="cse_dpo.train_factoid_exact_orbit_dpo",
    parameters={"question_classes_with_reference_jsonl": None, "model_name": None,
                "method": "representative_dpo", "max_seq_length": 4096, "seed": 3407},
    outputs={"output_dir": "training", "save_model_dir": "adapter"},
    inputs=("question_classes_with_reference_jsonl", "validation_question_classes_with_reference_jsonl"),
    required=("question_classes_with_reference_jsonl", "model_name"), gpu=True)
add("error_aware_dpo", "dpo", "Answer DPO plus an auxiliary error-diagnosis loss.",
    operation="error_aware_dpo",
    parameters={"base_model": None, "initial_adapter": None, "stage1_pairs": None,
                "c1_rationales": None, "gold_rationales": None, "dev_source": None,
                "auxiliary_weight": .02, "max_length": 4096, "train": False},
    inputs=("base_model", "initial_adapter", "stage1_pairs", "c1_rationales", "gold_rationales", "dev_source"),
    required=("base_model", "initial_adapter", "stage1_pairs", "c1_rationales", "gold_rationales", "dev_source"),
    gpu=True)

add("local_evaluation", "evaluation", "Evaluate any model(s), split(s), prompt, greedy or sampled aggregation.",
    module="src.utility.evaluate_models", schema_module="src.utility.evaluation",
    parameters={**PROMPT, "eval_input": None, "model_ref": None, "question_types": ["factoid"],
                "num_generations": 1, "aggregation_strategy": "union", "temperature": 0.,
                "top_p": 1., "max_new_tokens": 64, "seed": 3407, "score_backend": "bioasq_java",
                "max_factoid_answers": 10000},
    outputs={"output_dir": "evaluation"}, inputs=("eval_input", "prompt_file"),
    required=("eval_input", "model_ref"), gpu=True)
add("gpt_reasoning", "evaluation", "Paired direct versus structured-reasoning API evaluation.",
    module="cse_dpo.compare_gpt_direct_vs_structured_reasoning_dev",
    parameters={"source": None, "api_key_file": "open_ai_api.txt", "model": "gpt-4.1-mini-2025-04-14",
                "question_limit": 10, "reasoning_prompt_version": "v2", "reuse_direct_cache_root": None},
    outputs={"output_root": "comparison"}, inputs=("source",), required=("source",), api=True)
add("openai_direct", "evaluation", "Direct API answer generation on raw or prepared data, with official scoring.",
    operation="openai_direct",
    parameters={"sources": TEST, "model": "gpt-4.1-mini-2025-04-14",
                "api_key_file": "open_ai_api.txt", "question_limit": 10, "max_new_tokens": 80},
    inputs=("sources",), required=("sources",), api=True)
add("sampling_comparison", "evaluation", "Compare greedy, first-N, frequency and direct-top-five outputs.",
    operation="sampling_comparison",
    parameters={"source": None, "model_ref": None, "samples": 10, "seed": 3407,
                "max_seq_length": 4096, "max_new_tokens": 64, "temperature": .7, "top_p": .9,
                "strategies": ["greedy", "first5", "frequency"]},
    inputs=("source",), required=("source", "model_ref"), gpu=True)
add("conditioned_sampling", "evaluation", "Compare frozen independent samples with history-conditioned sampling.",
    operation="conditioned_sampling",
    parameters={"base_model": None, "adapter": None, "source": None, "bank": None, "limit": None},
    inputs=("base_model", "adapter", "source", "bank"), required=("base_model", "adapter", "source", "bank"), gpu=True)

add("compare_banks", "analysis", "Compare candidate coverage, sample counts and answer-set differences.",
    operation="compare_banks", parameters={"banks": {}}, required=("banks",))
add("evidence_coverage", "analysis", "Count snippet-matching factoid questions in raw BioASQ files.",
    operation="evidence_coverage", parameters={"sources": TEST, "match_mode": "normalized"},
    inputs=("sources",), required=("sources",))
add("dpo_diagnostics", "analysis", "Compare evaluation, candidate and preference-pair artifacts.",
    module="cse_dpo.analyze_dpo_diagnostics",
    parameters={"eval_root": None, "question_input": TEST, "candidate_bank": None, "pair_jsonl": []},
    outputs={"output_json": "diagnostics.json", "output_md": "diagnostics.md"},
    inputs=("eval_root", "question_input", "candidate_bank"))
add("stage1_audit", "analysis", "Audit Stage 1 generations and retention on eligible training questions.",
    operation="stage1_audit",
    parameters={"run_root": None, "source": None, "base_model": None, "limit": 10,
                "max_prompt_tokens": 4096, "max_new_tokens": 64, "seed": 3407},
    inputs=("run_root", "source", "base_model"), required=("run_root", "source", "base_model"), gpu=True)

add("synthetic_qa", "synthetic", "Answer-first synthetic QA with separate generator and verifier models.",
    module="cse_dpo.build_synthetic_factoid_qa_pilot",
    parameters={"phase": "prepare", "source": None, "dev_source": None,
                "test_dir": "data/Task13BTest", "api_key_file": "open_ai_api.txt",
                "generator_model": "gpt-4.1-mini-2025-04-14", "verifier_model": "gpt-4.1-mini-2025-04-14",
                "source_count": 20, "validation_source_count": 4, "max_new_calls": 10, "seed": 3407,
                "prepared_run": None},
    outputs={"output_root": "synthetic"}, inputs=("source", "dev_source", "test_dir"),
    required=("source", "dev_source"), api=True)
