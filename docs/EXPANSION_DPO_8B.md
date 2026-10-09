# Expansion DPO for Llama, Qwen3 and Ministral

The workflow adapts the three completed **expansion SFT** checkpoints. It compares complete, naturally generated expansion JSON responses for the same question. The earlier Llama list-answer DPO dataset cannot be reused as factoid expansion preferences.

No expansion DPO dataset or trained DPO model has been produced by adding these scripts. Generation, annotation and GPU training are separate stages. The first stage below submits response generation only.

## Checkpoints and partitions

| Backbone | Expansion SFT training job | Execution |
| --- | --- | --- |
| Llama-3.1-8B | 180794570 | default |
| Qwen3-8B | 180794574 | default; thinking disabled |
| Ministral-3-8B | 180872635 | eager recovery |

All adapters are under `gadi_sft_8b_starter/outputs/matched_8b_sft/`, as pinned in `configs/matched_8b_evaluation.json`. The existing paired-run validator checks completion, backbone revision, adapter files, source hashes and fitting/development separation before submission. Training and generation preserve each model's execution setting and native chat template.

Use the existing 1,296 fitting questions and 144 internal-validation questions. The outer 160 development questions and official test questions never supply preferences. Each response's prompt is copied from the first two SFT messages; its assistant target is excluded from generation.

The default pilot selects the same reproducible **100 fitting and 32 validation questions** for all three models. Each generates one greedy response and four stochastic responses at temperature 0.8, top-p 0.95 and top-k 0. This produces 660 responses per model, or 1,980 across three models, excluding smoke jobs. Each response contains at most ten expressions. Complete fitting/validation generation would produce 21,600 responses across the models; inspect pilot pair yield before paying this cost.

## 1. Generate response banks on Gadi

From the bundle directory:

```bash
git pull --ff-only
bash scripts/submit_expansion_dpo_8b.sh generate
```

This submits one generation smoke job and one dependent generation job for each model. Output directories are named `outputs/expansion_dpo_8b/<model>-generate-<PBS_JOBID>`. Only the three completed **full jobs**, with `manifest.json` status `complete` and `smoke_test: false`, become preference sources. The script prints a submission record with actual IDs.

Check the printed IDs with `qstat -swx`, then inspect their manifests. Raw responses, seeds, exact prompts and costs are retained in `responses.jsonl`. Formatting failures stay in the bank and are excluded explicitly during review; the primary dataset does not repair them.

To expand the bank after a successful pilot:

```bash
bash scripts/submit_expansion_dpo_8b.sh generate --full-dataset
```

Create a fresh preference dataset from the three full-panel banks. Do not mix pilot and full-panel banks or overwrite reviewed pilot data.

## 2. Score and review responses

Replace the three example paths below with the actual completed generation directories:

```bash
python scripts/expansion_dpo_data.py --mode annotate \
  --banks outputs/expansion_dpo_8b/llama31-generate-LLAMA_JOB.gadi-pbs \
          outputs/expansion_dpo_8b/qwen3-generate-QWEN_JOB.gadi-pbs \
          outputs/expansion_dpo_8b/ministral3-generate-MINISTRAL_JOB.gadi-pbs \
  --output outputs/expansion_dpo_8b/review-template.jsonl

python ../scripts/score_expansion_dpo_annotations.py \
  outputs/expansion_dpo_8b/review-template.jsonl \
  --output outputs/expansion_dpo_8b/review-official.jsonl
```

The scoring step requires the original `data/training13b.json` corpus, its pinned hash, and the official BioASQ Java scorer with a working JDK. It applies the official matcher independently to each candidate and saves the scoring audit. It sets `official_accepted` to a boolean and assigns C3 only to matches. Unmatched candidates remain unresolved.

Review `review-official.jsonl` and save the completed file under a new name, such as `reviewed.jsonl`. Keep every saved response exactly once and retain its prompt, raw response and candidate strings unchanged.

For each included response:

- Set `reviewer` and `review_notes` to identify the annotation source and scope checks.
- Keep C3 for officially accepted candidates. For unmatched candidates, set C2 when scientifically correct in context, or C1 when incorrect. An exact mismatch alone cannot justify C1.
- Set `supported` to a boolean after checking snippets for semantic support. Literal occurrence alone does not establish support for answering the question.
- Set `equivalent_to_original` to a boolean after checking that the variant preserves the first answer's entity, scope, qualifiers, quantity and units. The original is equivalent to itself even when it is the wrong answer concept.
- Record the supporting snippet IDs/text or the reason for lack of support in `evidence`. An incorrect candidate can appear in a snippet while still answering the wrong question.
- Use `excluded_reason` for unresolved, conflicting or unreliable cases. Null labels are excluded automatically. Model-generated review should be identified and independently audited before research claims.

These annotations require biomedical judgment. The workflow does not call a paid judge or invent semantic/evidence labels. The review file retains model and draw identifiers for provenance; a blinded human audit should conceal these during judgment.

## 3. Construct shared whole-response pairs

Use the same three generation directories as stage 2:

```bash
python scripts/expansion_dpo_data.py --mode pairs \
  --banks outputs/expansion_dpo_8b/llama31-generate-LLAMA_JOB.gadi-pbs \
          outputs/expansion_dpo_8b/qwen3-generate-QWEN_JOB.gadi-pbs \
          outputs/expansion_dpo_8b/ministral3-generate-MINISTRAL_JOB.gadi-pbs \
  --annotations outputs/expansion_dpo_8b/reviewed.jsonl \
  --output outputs/expansion_dpo_8b/preferences-pilot-v1
```

Responses from all three models form a shared bank. A pair always compares two complete responses to exactly the same prompt. Cross-model pairs are allowed, and identical semantic JSON pairs are deduplicated. At most **two pairs per question in total**, across all sources, enter each partition. Unresolved responses, ties and tradeoffs are excluded. Existing validated SFT targets are not silently inserted as preferred responses.

The chosen response must start with a correct concept and contain no reviewed incorrect, unsupported or non-equivalent candidates. It must not decrease either accepted-expression count or semantically correct expression count, and must not increase any of the three error counts. At least one criterion must improve. C2 counts as correct. Formatting, raw text length and response length earn no independent preference.

Consequently:

- `TNF-alpha` plus its supported full name can beat `TNF-alpha` alone.
- The same valid response can beat one that adds `IL-6` as a false synonym, even though the chosen response is shorter.
- A response with more accepted aliases **and** more wrong candidates is excluded as a conflicting tradeoff.
- A scientifically correct but unaccepted answer is not relabelled as an incorrect concept.

This is a conservative **coverage and validity** pilot. It does not directly optimize candidate ordering or train a reranker. Count improvements concern distinct surface strings, not distinct biomedical entities; audit whether preferences overvalue low-value formatting variants. Report pair yield, exclusions, lengths, source combinations and categories before full training.

The output manifest records source banks, annotation hashes and pair counts. Empty fitting or validation preferences block training. Validation regenerates the preference decisions from the archived annotations and rejects modified responses, changed partitions or changed adapter provenance.

## 4. Train each model with its own SFT reference

```bash
python scripts/expansion_dpo_data.py --mode validate \
  --output outputs/expansion_dpo_8b/preferences-pilot-v1

bash scripts/submit_expansion_dpo_8b.sh train \
  outputs/expansion_dpo_8b/preferences-pilot-v1
```

This submits a two-step training smoke and a dependent training job for each model. Initial settings are beta 0.1, learning rate 5e-6, one epoch, batch size one, accumulation 16, and an 8,192-token sequence cap. Native assistant tokens supply completion-only masks; overflowing or incompatible template prefixes cause an error rather than truncation.

The trainer continues the existing language LoRA adapter. Before any update, it computes and caches each chosen/rejected completion's summed log probability under that model's initial expansion SFT policy. These fixed values are its frozen reference, avoiding a second 8B backbone in memory. Dropout is disabled for policy and reference likelihoods. The loss is the standard sigmoid DPO objective, implemented with the existing Transformers/Unsloth stack:

`-log sigmoid(beta * ((policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)))`.

The objective follows the [DPO paper](https://arxiv.org/abs/2305.18290). Cached references are rebuilt per run, recorded with hashes and never taken from a base-only policy. All three models receive identical preference rows but their reference likelihoods differ. Local tests verify completion masking, padding, the frozen-reference loss at initialization and the preference gradient direction. CUDA execution and memory feasibility remain subject to Gadi smoke tests.

The pilot selects the best epoch by **internal-validation preference loss**. This differs from the proposed confirmatory validation-MRR search in the methodology report. Treat pilot results as exploratory; freeze the final protocol before confirmatory runs.

## Comparison after training

Compare unchanged expansion SFT with its DPO continuation using the same development questions, snippets, native expansion prompt, greedy decoding and first-five candidate ordering. Ten-sample comparisons must use identical sampling settings on both sides. Score official MRR at five and coverage separately, with paired question-level confidence intervals and cost measurements.

A continued-SFT control, validation-MRR hyperparameter search, seed repeats and genuinely unseen final evaluation are still needed for a confirmatory claim. Improved fitting preferences or lower preference loss do not establish improved held-out BioASQ performance. DPO may add little benefit or reduce useful diversity; retain those outcomes.
