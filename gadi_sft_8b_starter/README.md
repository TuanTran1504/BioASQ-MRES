# BioASQ 8B Experiments for Gadi

This is a small, portable package for Gadi. It supports the original SFT jobs plus gold-blind exact-span and equivalent-expression expansion experiments. It deliberately excludes prior adapters, DPO data, candidate banks, notebooks, test evaluations, and historical artifacts.

For fresh, matched single-answer and expansion SFT across Llama-3.1-8B,
Qwen3-8B and Ministral-3-8B, use the [paired 8B training workflow](docs/MATCHED_8B_SFT.md).
It prepares six training conditions with shared question splits and base snapshots,
each gated by a training-and-generation smoke job.

To evaluate the historical Qwen2.5 0.5B/3B SFT checkpoints with the existing
expansion prompt and ten high-temperature single-answer draws, use the
[original SFT inference workflow](docs/ORIGINAL_QWEN25_INFERENCE.md). The exact
historical adapters are transferred separately, then smoke-tested before full runs.

It trains `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` with 4-bit LoRA and generated-dev BioASQ MRR checkpoint selection.

The expansion jobs run the unchanged instruction checkpoint without training. They give the model each of the fixed 160 development questions and all of its snippets. The extractive branch asks for literal answer spans; the equivalent branch uses the original GPT-4.1-mini prompt that produced 55% dev coverage and asks for synonymous or formatting-equivalent answer expressions. Both write candidates in the same scoreable artifact format.

## Included variants

| Variant | Training examples | Dev questions | Purpose |
| --- | ---: | ---: | --- |
| `full_resources_per_alias` | 1,804 | 160 | Full-resource per-gold-alias SFT, matching the full-resource notebook design. |
| `evidence_per_supported_alias` | 1,352 | 160, plus a 129-question supported subset | Evidence-grounded per-supported-alias SFT, matching the evidence-grounded notebook design. |

Both variants use a batch size of 1, gradient accumulation of 32, sequence length 4,096, rank-32 LoRA with dropout 0.05, linear warmup of 5 updates, and generated-dev MRR model selection. Training artifacts are written to `outputs/runs/<timestamp>-<run-name>/` and are never overwritten by a new run name.

The supplied environment setup pins the `gpuvolta` V100 nodes to the PyTorch 2.11 CUDA 12.6 wheel. The default CUDA 13 wheel does not contain compute capability 7.0 code for the V100.

## First use on Gadi

1. Copy this complete directory to `/scratch/nl78/$USER/`.
2. Load modern Python, then validate the bundle: `module load python3/3.12.13` followed by `python3 scripts/verify_bundle.py`.
3. Create the environment once on a login node:

```bash
PYTHON_MODULE=python3/3.12.13 bash scripts/setup_environment.sh
```

4. Accept Meta's Llama license using your Hugging Face account, export `HF_TOKEN`, then cache the model once on a login node:

```bash
export HF_HOME=/scratch/nl78/$USER/hf_cache
export HF_TOKEN=<your-token>
python3 scripts/download_base_model.py
unset HF_TOKEN
```

Alternatively, keep the token in a permission-protected file outside the repository and avoid exporting it:

```bash
python3 scripts/download_base_model.py \
  --token-file /scratch/nl78/$USER/.secrets/hf_token.txt
```

5. Edit the `#PBS -q` line in a job file if `gpuvolta` is not the GPU queue available to `nl78`. Check with `qstat -Q` or submit the smoke test first.
6. Submit the smoke test, then a full run:

```bash
qsub jobs/00_gpu_smoke_test.pbs
qsub jobs/train_full_resources_8b.pbs
# or
qsub jobs/train_evidence_grounded_8b.pbs
```

Monitor with `qstat -swx <job-id>`. PBS output and errors are retained by Gadi; model, checkpoints, logs, dev predictions, metrics, and manifests are stored under `outputs/`.

## Important notes

For the completed Qwen2.5-0.5B and 3B expansion SFT adapters, use the
[matched base/SFT evaluation workflow](docs/MATCHED_QWEN25_EVALUATION.md).
It provides validation, paired smoke/full GPU jobs, and official candidate scoring
with generated-order MRR and coverage diagnostics on all 160 development questions.

- Run `scripts/download_base_model.py` on a login node, not a compute node. Training defaults to `--local-files-only` to prevent compute-node downloads.
- The first GPU job should be the smoke test. It validates CUDA, the environment, data, model cache, and the BioASQ Java evaluator with only eight training/dev examples.
- Do not use `--resume-from-checkpoint auto` with a reused run directory. This launcher creates timestamped managed run folders, so every normal submission starts a distinct experiment.
- The PBS resource requests are a conservative one-GPU starting point for 8B 4-bit LoRA. If the smoke test reports out-of-memory, reduce `max_seq_length` or increase the requested GPU-memory class if your allocation permits it.

## Exact-span expansion experiment

Before transfer, create the fixed dev-set export from the full repository:

```bash
python scripts/export_gadi_expansion_dev.py
python gadi_sft_8b_starter/scripts/verify_expansion_bundle.py
```

The generated `gadi_sft_8b_starter/data/` directory is ignored by Git but must be included in the transfer to Gadi. It contains gold aliases for offline scoring; the runner sends only the question and snippets to the model.

On Gadi, after the environment and model cache have been prepared, submit the four-question smoke test and inspect its status before starting all 160 questions:

```bash
cd /scratch/nl78/$USER/gadi_sft_8b_starter
qsub jobs/01_expansion_smoke_test.pbs
qstat -swx <job-id>
cat outputs/expansion/*smoke*/status.json

qsub jobs/run_extractive_expansion_8b.pbs
```

The full job uses greedy decoding, a 6,144-token total sequence limit, 512 output tokens, and fails if all snippets do not fit. The validator keeps at most ten distinct literal spans in first-occurrence order. If generation reaches the token limit before closing the outer JSON object, it recovers complete candidate objects, records `incomplete_top_level_json_recovered`, and marks the response as schema-noncompliant. It writes progress after every question under `outputs/expansion/`. To resume an interrupted directory without repeating completed questions, run the same script in a GPU job with `--resume-run outputs/expansion/<run-directory>`.

After copying the completed run directory back to this repository, score it with the existing official BioASQ analyzer:

```bash
python scripts/analyze_gadi_expansion.py <run-directory>
```

See [docs/TRANSFER_TO_GADI.md](docs/TRANSFER_TO_GADI.md) for the transfer and operating procedure.

## Equivalent-expression expansion experiment

This is a separate experiment from the extractive branch. It uses the exact original equivalent-expansion prompt, returns `answer` plus `relation_type`, and does not require generated variants to be literal snippet substrings. Gold aliases remain excluded from model input.

Run the four-question smoke test first:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
python3 scripts/verify_expansion_bundle.py --config configs/equivalent_expansion_8b.json
JOB_ID=$(qsub jobs/02_equivalent_expansion_smoke_test.pbs)
echo "$JOB_ID"
```

After it finishes, inspect the newest status:

```bash
SMOKE_RUN=$(ls -td outputs/equivalent_expansion/*equivalent-smoke* | head -n 1)
python3 -m json.tool "$SMOKE_RUN/status.json"
```

If all four questions complete and candidates are present, submit the full dev run:

```bash
qsub jobs/run_equivalent_expansion_8b.pbs
```

## Cross-model equivalent-expansion comparison

The comparison uses the same fixed dev questions, all snippets, original equivalent-expansion prompt, greedy decoding, 512-token output limit, parser, and official BioASQ scorer for every model. The additional checkpoints are:

| Model | Configuration | Gadi loader |
| --- | --- | --- |
| Qwen3-8B | `configs/equivalent_expansion_qwen3_8b.json` | `FastLanguageModel`, with thinking disabled |
| Ministral-3-8B-Instruct-2512 | `configs/equivalent_expansion_ministral3_8b.json` | `FastModel` |
| Gemma-3-27B-IT | `configs/equivalent_expansion_gemma3_27b.json` | `FastModel` |

Cache the models from a Gadi login node. Gemma requires accepting its Hugging Face terms before the token can download it.

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
module purge
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
export HF_HOME=/scratch/nl78/$USER/hf_cache

python3 scripts/download_base_model.py \
  --model unsloth/Qwen3-8B-unsloth-bnb-4bit \
  --token-file /scratch/nl78/$USER/.secrets/hf_token.txt
python3 scripts/download_base_model.py \
  --model unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit \
  --token-file /scratch/nl78/$USER/.secrets/hf_token.txt
python3 scripts/download_base_model.py \
  --model unsloth/gemma-3-27b-it-unsloth-bnb-4bit \
  --token-file /scratch/nl78/$USER/.secrets/hf_token.txt
```

## Matched expansion SFT across Qwen sizes

The strict teacher-validated expansion dataset can be used unchanged for
Qwen2.5-0.5B, Qwen2.5-3B and Qwen3-8B. The three configurations use the same
1,296-question training split, 144-question internal-validation split, seed,
8,192-token limit, effective batch size, three-epoch budget and LoRA settings.
The separate 160-question outer development set is not used for training or
checkpoint selection.

Check the Gadi cache without downloading anything:

```bash
python3 scripts/download_base_model.py --check-only \
  --model unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
python3 scripts/download_base_model.py --check-only \
  --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit
python3 scripts/download_base_model.py --check-only \
  --model unsloth/Qwen3-8B-unsloth-bnb-4bit
```

If a check reports that files are unavailable locally, rerun that command
without `--check-only` (and add `--token-file` if the account requires it).
Then validate the two new configs and submit smoke tests:

```bash
python3 scripts/run_expansion_sft_qwen3.py --mode validate \
  --config configs/expansion_sft_qwen25_05b.json
python3 scripts/run_expansion_sft_qwen3.py --mode validate \
  --config configs/expansion_sft_qwen25_3b.json
qsub jobs/09_expansion_sft_qwen25_05b_smoke.pbs
qsub jobs/10_expansion_sft_qwen25_3b_smoke.pbs
qsub jobs/08_expansion_sft_qwen3_smoke.pbs
```

Only after each smoke test completes successfully, submit the corresponding
full job:

```bash
qsub jobs/train_expansion_sft_qwen25_05b.pbs
qsub jobs/train_expansion_sft_qwen25_3b.pbs
qsub jobs/train_expansion_sft_qwen3_8b.pbs
```

Run each smoke test before its full run. The Gemma smoke test intentionally uses only two questions because the 20 GB 4-bit checkpoint is close enough to the V100's 32 GB limit that model loading and generation must be confirmed first.

```bash
qsub jobs/03_qwen3_8b_equivalent_smoke_test.pbs
qsub jobs/04_ministral3_8b_equivalent_smoke_test.pbs
qsub jobs/05_gemma3_27b_equivalent_smoke_test.pbs
```

For each completed smoke test, inspect `status.json`, the PBS error log, and peak `GPU Memory Used`. Submit a full run only if the smoke status is `complete` and it produced candidates:

```bash
qsub jobs/run_equivalent_expansion_qwen3_8b.pbs
qsub jobs/run_equivalent_expansion_ministral3_8b.pbs
qsub jobs/run_equivalent_expansion_gemma3_27b.pbs
```

Copy each completed directory from `outputs/model_comparison/` back to the local repository and score it with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_gadi_expansion.py <run-directory>
```

Compare coverage at 1, 5, and 10, parse success, unique candidates per question, runtime, and peak GPU memory. Also measure the union with GPT-4.1 mini and the existing Llama-3.1-8B run: a model with lower standalone coverage can still be valuable if it covers questions the other generators miss.

## Multi-surface equivalent-expansion follow-up

The first cross-model comparison showed that many official misses contained a
semantically plausible answer but failed to reproduce the accepted surface. The v2
prompt remains gold-blind and tests a targeted remedy: it explicitly requests minimal
answers, exact evidence phrases, full clause forms, coordinated answers, complete
numeric ranges, abbreviations, parenthetical forms, and harmless typography variants.
It uses Qwen3-8B so the result can be compared directly with the v1 Qwen run.

Run the four-question smoke test and inspect its status before submitting all 160
questions:

```bash
QWEN_V2_SMOKE=$(qsub jobs/06_qwen3_8b_equivalent_v2_smoke_test.pbs)
qstat -fx "$QWEN_V2_SMOKE" | grep -E 'job_state|Exit_status|resources_used'

qsub jobs/run_equivalent_expansion_qwen3_8b_v2.pbs
```

Treat this as a prompt ablation selected from development-set error analysis. Report it
as exploratory evidence and freeze the chosen prompt before the locked test evaluation.

## Long-context neural reranker pilot

The neural pilot uses the pinned Qwen3-Reranker-0.6B checkpoint. Each encoded input
contains the question, one candidate, and every supplied snippet in its original
order. The tokenizer preflight fails if any input exceeds 8,192 tokens; evidence is
never selected, dropped, or silently truncated. Source identity, source rank, and gold
aliases are not model inputs.

Run the pretrained reranker zero-shot first. The fine-tuned condition uses LoRA and a
within-question pairwise ranking loss: accepted candidates are preferred to hard
negatives from the same authentic slate. Questions without an accepted candidate are
excluded from the training loss but are still ranked during evaluation.

The current pilot bundle is development-only and contains 7,362 candidates for 160
questions. Before submitting a job, make sure these local files have been copied to
the same relative paths on Gadi:

```text
Artifacts/reranker_pilot/20261002-023517-tfidf-logistic/candidate_pool_labeled.jsonl
gadi_sft_8b_starter/outputs/model_comparison/20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs/examples.jsonl
```

Cache the pinned checkpoint once from a Gadi login node while network access is
enabled:

```bash
module purge
module load python3/3.12.13
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf download Qwen/Qwen3-Reranker-0.6B \
  --revision fd9fb1d26c07223ced488065909faf522e29cc7d \
  --cache-dir "${HF_HUB_CACHE}"
```

Validate the data without loading the model, then use the cached tokenizer to prove
that all 7,362 candidate inputs fit without truncation:

```bash
python3 scripts/run_qwen3_reranker.py --mode validate
python3 scripts/run_qwen3_reranker.py --mode preflight --local-files-only
```

For the fixed development pool, the pinned tokenizer produced a median length of 581
tokens, a 95th percentile of 2,593, and a maximum of 4,437. All inputs fit within the
8,192-token runtime limit while retaining every snippet.

Run the smoke test, which deliberately selects the longest training and held-out
questions to exercise worst-case GPU memory:

```bash
RERANKER_SMOKE=$(qsub jobs/07_qwen3_reranker_smoke_test.pbs)
qstat -fx "$RERANKER_SMOKE" | grep -E 'job_state|Exit_status|resources_used'
```

Inspect the smoke-test `status.json`, log, token preflight, and rankings. If it passes,
run the full zero-shot baseline before launching all five LoRA folds:

```bash
qsub jobs/evaluate_qwen3_reranker_zero_shot.pbs
qsub jobs/train_qwen3_reranker_cv.pbs
```

The full cross-validation job saves each fold's LoRA adapter under
`fold-<n>/adapter/` together with its held-out rankings and summary. The smoke job
does not retain its temporary adapter.

The evidence-plus-metadata condition retains every snippet and additionally encodes
only fields available at inference: generator sources and ranks, source agreement,
relation and surface-operation types, format-variant status, literal evidence
occurrence, and candidate length. Gold aliases and exact labels remain excluded from
the model input. Preflight this condition, then launch its full five-fold run:

```bash
python3 scripts/run_qwen3_reranker.py \
  --mode preflight \
  --encode-source-metadata \
  --local-files-only

qsub jobs/train_qwen3_reranker_metadata_cv.pbs
```

This is a full 160-question cross-validation experiment. The earlier six-question
smoke run is used only to validate execution and is not used to estimate accuracy.

The cross-validation result is exploratory because the candidate pool and its
formatting branches were selected using this development set. Do not report a model
trained on all 160 questions as independently evaluated on the same questions.
