1. Create Candidate bank
python -m cse_dpo.generate_candidate_bank \
  --eval-input data/training13b.json \
  --model-ref Artifacts/models/runs/20260803-094013-bioasq-8b-visible-gold-sft/adapter \
  --output-dir Artifacts/cse_dpo/candidate_banks/bioasq_8b_visible_gold_mr5_train \
  --question-types list \
  --samples-per-question-total 12 \
  --max-resources 5 \
  --max-resource-chars 0 \
  --max-seq-length 131072 \
  --max-new-tokens 1024 \
  --temperature 0.7 \
  --top-p 0.9 \
  --batch-size 1 \
  --local-files-only
2. Create Metric DPO
This uses all BioASQ gold answers, no gold fallback, drops risky semantic overlap, but allows evidence-supported gold-unmatched extras because it is metric-aligned.

python -m cse_dpo.construct_set_edit_pairs_hybrid \
  --question-input data/training13b.json \
  --candidate-input Artifacts/cse_dpo/candidate_banks/full_train_full_context_new/20260720-141912-sft/candidate_bank.jsonl \
  --output-jsonl Artifacts/cse_dpo/pairs/train_dpo_fullcontext.jsonl \
  --summary-json Artifacts/cse_dpo/audits/train_fullcontext_metric_table_policy_summary.json \
  --manual-audit-md Artifacts/cse_dpo/audits/train_fullcontext_metric_table_policy_audit.md \
  --max-resources 0 \
  --max-resource-chars 0 \
  --semantic-embedding-model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --semantic-embedding-match-threshold 0.9 \
  --semantic-embedding-uncertain-threshold 0.82 \
  --semantic-embedding-no-match-threshold 0.5 \
  --allow-evidence-supported-negative-pairs \
  --min-negative-addition-delta-f1 0.05 \
  --max-negative-addition-pairs-per-question 100 \
  --max-valid-omission-pairs-per-question 100

3. Create whole-response DPO

python -m cse_dpo.construct_whole_response_dpo_pairs \
  --question-input data/training13b.json \
  --candidate-input Artifacts/cse_dpo/candidate_banks/unitutor_8b_unitutor_train/answer-gen-unitutor-8b/candidate_bank.jsonl \
  --output-jsonl Artifacts/cse_dpo/pairs/train_8b_unitor_whole_response_dpo.jsonl \
  --summary-json Artifacts/cse_dpo/audits/train_8b_unitor_whole_response_dpo_summary.json \
  --manual-audit-md Artifacts/cse_dpo/audits/train_8b_unitor_whole_response_dpo_audit.md \
  --max-resources 0 \
  --max-resource-chars 0 \
  --min-delta-f1 0.05 \
  --max-pairs-per-question 8

4. Train the DPO model

python src/dpo_train.py \
  --preference-input Artifacts/cse_dpo/pairs/train_8b_unitutor_gold_anchor_strict_dpo.jsonl \
  --model-name Artifacts/models/runs/20260727-121150-unitutor-llama31-8b-answer-gen/adapter \
  --output-dir Artifacts/models/dpo_runs/unitutor_8b_gold_anchored/trainer_output \
  --save-model-dir Artifacts/models/dpo_runs/unitutor_8b_gold_anchored/adapter \
  --validation-ratio 0.2 \
  --split-by question \
  --early-stopping-patience 3 \
  --save-steps 100 \
  --max-seq-length 4096 \
  --local-files-only

5. Evaluate models with BioASQ test data.


One-shot:

# Allow full resource
python src/utility/evaluate_models.py \
  --model-ref Artifacts/models/runs/20260727-121150-unitor-llama31-answer-gen/adapter \
  --output-dir Artifacts/evaluations/unitutor_model \
  --question-types list \
  --num-generations 1 \
  --max-resources 0 \
  --max-resource-chars 0 \
  --max-seq-length 8192 \
  --max-new-tokens 512 \
  --local-files-only

python src/utility/evaluation.py \
  --model-ref answer-gen-unitutor-8b \
  --eval-input data/Task13BTest/13B1_golden.json data/Task13BTest/13B2_golden.json data/Task13BTest/13B3_golden.json data/Task13BTest/13B4_golden.json \
  --question-types list \
  --max-resources 5 \
  --resource-selection embedding \
  --resource-granularity snippet \
  --resource-reranker-model sentence-transformers/all-MiniLM-L12-v2 \
  --resource-reranker-device cpu

# Different resource numbers

python src/utility/evaluate_models.py \
  --model-ref Artifacts/models/dpo_runs/bioasq_8b_baseline_mr5_dpo/adapter \
  --eval-input data/Task13BTest/13B1_golden.json data/Task13BTest/13B2_golden.json data/Task13BTest/13B3_golden.json data/Task13BTest/13B4_golden.json \
  --question-types list \
  --max-resources 5 \
  --max-resource-chars 1200 \
  --max-new-tokens 512 \
  --score-backend both \
  --output-dir Artifacts/evaluations/dpo_evals/bioasq-8b-baseline_dpo_mr5_test \
  --local-files-only

# Unitor style evaluation
python src/utility/evaluate_models.py \
  --model-ref Artifacts/models/runs/20260802-174129-unitutor-8b-mr5-visible-gold-sft-f1-es/adapter \
  --eval-input data/Task13BTest/13B1_golden.json data/Task13BTest/13B2_golden.json data/Task13BTest/13B3_golden.json data/Task13BTest/13B4_golden.json \
  --question-types list \
  --max-resources 5 \
  --max-resource-chars 1200 \
  --max-new-tokens 512 \
  --prompt-format unitor_plain \
  --chat-template llama-3 \
  --output-dir Artifacts/evaluations/sft-unitor-8b-visible-gold-f1-es_mr5_test \
  --local-files-only

# Multi-sampling

python src/utility/evaluate_models.py \
  --model-ref Artifacts/models/runs/20260731-142856-unitutor-8b-mr5-visible-gold-sft/adapter \
  --output-dir Artifacts/evaluations/multi_sft-unitor-8b-visible-gold_mr5_test \
  --question-types list \
  --max-resources 5 \
  --max-resource-chars 0 \
  --num-generations 8 \
  --aggregation-strategy frequency \
  --aggregation-min-frequency 2 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --prompt-format unitor_plain \
  --chat-template llama-3 \
  --local-files-only

  python src/utility/evaluate_models.py \
  --model-ref Artifacts/models/dpo_runs/20260803-094013-bioasq-8b-visible-gold-sft_train_8b_whole_response_mr5_dpo_v3/adapter\
  --eval-input \
    data/Task13BTest_list_full_gold_in_snippets/13B1_golden.json \
    data/Task13BTest_list_full_gold_in_snippets/13B2_golden.json \
    data/Task13BTest_list_full_gold_in_snippets/13B3_golden.json \
    data/Task13BTest_list_full_gold_in_snippets/13B4_golden.json \
  --question-types list \
  --max-resources 5 \
  --max-resource-chars 1200 \
  --resource-selection first \
  --resource-granularity document \
  --max-seq-length 4096 \
  --max-new-tokens 512 \
  --num-generations 1 \
  --temperature 0.0 \
  --top-p 1.0 \
  --score-backend both \
  --local-files-only \
  --output-dir Artifacts/evaluations/dpo_evals/20260803-094013-bioasq-8b-visible-gold-sft_train_8b_whole_response_mr5_dpo_v3_test
