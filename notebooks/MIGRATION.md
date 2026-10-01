# Notebook migration

42 notebooks present at consolidation were preserved byte-for-byte in [the archive](../reproducibility/notebook_archive.zip). Names, original sizes and SHA-256 hashes are in [the manifest](../reproducibility/notebook_archive_manifest.json). Previously deleted notebooks were not restored or added to the archive.

The eight active notebooks reuse Python implementations and named presets. Model size, data split, prompts and hyperparameters are configuration values. The mapping below identifies the shared replacement; it does not imply identical historical hyperparameters or plots. Notebook-only historical plots, the older synthetic v1 configuration and specialized frozen experiment outputs remain in the archive.

| Original notebook | Active workflow | Presets |
|---|---|---|
| audit_stage1_training_generations.ipynb | [07_analysis.ipynb](07_analysis.ipynb) | compare_banks / dpo_diagnostics / stage1_training_audit / test_evidence_coverage |
| build_factoid_evidence_grounded_gold_response_dpo_pairs_qwen25_05b.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | standard_factoid / all_factoid_negatives |
| build_factoid_standard_dpo_pairs_from_strict_extractive_banks.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | standard_factoid / all_factoid_negatives |
| build_synthetic_factoid_qa_pilot.ipynb | [08_synthetic_qa.ipynb](08_synthetic_qa.ipynb) | pipeline / finish / prepare / generate / verify / finalize |
| build_synthetic_factoid_qa_pilot_v2.ipynb | [08_synthetic_qa.ipynb](08_synthetic_qa.ipynb) | pipeline / finish / prepare / generate / verify / finalize |
| candidates_evaluation.ipynb | [07_analysis.ipynb](07_analysis.ipynb) | compare_banks / dpo_diagnostics / stage1_training_audit / test_evidence_coverage |
| check_factoid_test_evidence_support.ipynb | [07_analysis.ipynb](07_analysis.ipynb) | compare_banks / dpo_diagnostics / stage1_training_audit / test_evidence_coverage |
| compare_frequency10_vs_conditioned5_qwen25_05b.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | history_conditioned |
| compare_gpt_direct_vs_structured_reasoning_dev.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | gpt_direct_vs_reasoning |
| compare_stage1_dpo_vs_sft_sampling_inference.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | greedy / compare_sampling / sample10_frequency |
| compare_synthetic_factoid_qa_generator_models.ipynb | [08_synthetic_qa.ipynb](08_synthetic_qa.ipynb) | pipeline / finish / prepare / generate / verify / finalize |
| evaluate_factoid_generation_methods_bioasq.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | greedy / compare_sampling / sample10_frequency |
| evaluate_factoid_models_on_dev_qwen25_05b.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | greedy / compare_sampling / sample10_frequency |
| evaluate_factoid_models_on_test_qwen25_05b.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | greedy / compare_sampling / sample10_frequency |
| evaluate_factoid_openai_only_on_test.ipynb | [06_evaluation.ipynb](06_evaluation.ipynb) | openai_direct |
| generate_factoid_evidence_grounded_qwen25_05b_dpo_candidate_bank_n20.ipynb | [02_candidate_generation.ipynb](02_candidate_generation.ipynb) | sample10 / sample20 / literal_copy |
| generate_factoid_qwen25_05b_dpo_candidate_bank_n20.ipynb | [02_candidate_generation.ipynb](02_candidate_generation.ipynb) | sample10 / sample20 / literal_copy |
| generate_factoid_qwen25_05b_split_dpo_candidate_bank.ipynb | [02_candidate_generation.ipynb](02_candidate_generation.ipynb) | sample10 / sample20 / literal_copy |
| generate_factoid_strict_extractive_qwen25_05b_3b_candidate_banks.ipynb | [02_candidate_generation.ipynb](02_candidate_generation.ipynb) | sample10 / sample20 / literal_copy |
| inspect_dev_candidate_bank_differences.ipynb | [07_analysis.ipynb](07_analysis.ipynb) | compare_banks / dpo_diagnostics / stage1_training_audit / test_evidence_coverage |
| inspect_train_candidate_bank_differences.ipynb | [07_analysis.ipynb](07_analysis.ipynb) | compare_banks / dpo_diagnostics / stage1_training_audit / test_evidence_coverage |
| judge_factoid_candidate_banks_train200_c3_c0_llm.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | judge_sample10 / judge_sample20 |
| judge_factoid_candidate_banks_train_next200_merge400_c3_c0_llm.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | judge_sample10 / judge_sample20 |
| judge_factoid_predictions_c3_c0_llm.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | judge_sample10 / judge_sample20 |
| judge_factoid_qwen25_05b_split_dpo_bank_gpt.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | judge_sample10 / judge_sample20 |
| model_evaluation_mistake_analysis.ipynb | [07_analysis.ipynb](07_analysis.ipynb) | compare_banks / dpo_diagnostics / stage1_training_audit / test_evidence_coverage |
| prepare_factoid_evidence_answer_sft.ipynb | [01_data_preparation.ipynb](01_data_preparation.ipynb) | judge_evidence / occurrence_evidence |
| prepare_factoid_evidence_answer_sft_occurrence_only.ipynb | [01_data_preparation.ipynb](01_data_preparation.ipynb) | judge_evidence / occurrence_evidence |
| score_stage2_alignment_potential.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | stage2_alignment |
| split_candidate_bank_dpo_pairs_into_three_stages.ipynb | [03_judging_and_preference_pairs.ipynb](03_judging_and_preference_pairs.ipynb) | three_stage_curriculum |
| train_factoid_error_aware_multitask_dpo_qwen25_05b.ipynb | [05_dpo_training.ipynb](05_dpo_training.ipynb) | error_aware |
| train_factoid_evidence_answer_sft_qwen25_05b.ipynb | [04_sft_training.ipynb](04_sft_training.ipynb) | evidence_plus_answer / evidence_answer_control |
| train_factoid_evidence_grounded_single_answer_qwen25_05b.ipynb | [04_sft_training.ipynb](04_sft_training.ipynb) | answer_only |
| train_factoid_evidence_grounded_single_answer_qwen25_3b.ipynb | [04_sft_training.ipynb](04_sft_training.ipynb) | answer_only |
| train_factoid_full_resources_single_answer_qwen25_05b.ipynb | [04_sft_training.ipynb](04_sft_training.ipynb) | answer_only |
| train_factoid_grounded_rationale_sft_qwen25_05b.ipynb | [04_sft_training.ipynb](04_sft_training.ipynb) | rationale_from_base / rationale_continued |
| train_factoid_snippet_supported_qwen25_05b_3b.ipynb | [01_data_preparation.ipynb](01_data_preparation.ipynb) + [04_sft_training.ipynb](04_sft_training.ipynb) | snippet_supported + answer_only |
| train_factoid_stage1_jaccard0_standard_dpo_qwen25_3b.ipynb | [05_dpo_training.ipynb](05_dpo_training.ipynb) | stage1 / two_stage / three_stage |
| train_factoid_stage2_dpo_d_retention_qwen25_3b.ipynb | [05_dpo_training.ipynb](05_dpo_training.ipynb) | stage2_retention |
| train_factoid_three_stage_dpo_qwen25_05b.ipynb | [05_dpo_training.ipynb](05_dpo_training.ipynb) | stage1 / two_stage / three_stage |
| train_factoid_two_stage_dpo_qwen25_05b_sft80_old_full_pairs.ipynb | [05_dpo_training.ipynb](05_dpo_training.ipynb) | stage1 / two_stage / three_stage |
| train_factoid_two_stage_standard_dpo_qwen25_3b.ipynb | [05_dpo_training.ipynb](05_dpo_training.ipynb) | stage1 / two_stage / three_stage |

To inspect an exact historical experiment, extract its notebook from the ZIP to a separate directory. Archived notebooks retain their original execution switches, paths and outputs; they do not acquire the new preview protections.

Important changes:

- The raw-data split filters before splitting, samples question IDs with a fixed seed, and expands aliases afterward. The previous notebooks often split before filtering.
- Snippet filtering searches snippet text only, excluding resource-ID matches.
- The candidate judge now accepts configurable sample counts and derives model labels from candidate provenance.
- Rationale SFT validates actual unique accepted questions and replay counts rather than requiring 1,111 questions.
- Every active workflow defaults to preview. GPU or API work requires explicit execution settings.
- Active runs get fresh directories, configuration manifests, file input hashes and logs. Earlier experiment outputs are never overwritten by the runner.
- Standardized diagnostics consolidate the repeated reports; exact exploratory tables and custom charts remain archived.
