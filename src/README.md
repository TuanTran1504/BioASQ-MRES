# Training And Evaluation Workflow

This folder contains the BioASQ answer-generation training, DPO training, and evaluation code.

The main workflow is:

```text
SFT training -> candidate generation -> preference-pair construction -> DPO training -> evaluation
```

Most current experiments use the already-trained SFT adapter:

```text
Artifacts/models/runs/20260613-132508-llama32-3b-bioasq/adapter
```

The registry alias for this SFT model is:

```text
answer-gen-current
```

## Main Scripts

| Script | Purpose |
| --- | --- |
| `utility/answer_gen_ft.py` | Supervised fine-tuning for BioASQ answer generation. |
| `dpo_train.py` | DPO fine-tuning from preference-pair JSONL files. |
| `utility/evaluate_models.py` | Evaluate one or more models on BioASQ data. |
| `utility/ensemble_predictions.py` | Offline ensemble over saved prediction files. |
| `manage_models.py` | Inspect model registry entries and aliases. |

## Supporting Modules

| Module | Purpose |
| --- | --- |
| `utility/data.py` | Converts raw BioASQ questions into prompt/resource/output records. |
| `utility/eval_dataset.py` | Builds evaluation examples and renders prompts. |
| `utility/eval_models.py` | Loads models, generates answers, and aggregates multi-generation outputs. |
| `utility/eval_runner.py` | Orchestrates evaluation and writes artifacts. |
| `utility/bioasq_format.py` | Parses model answers and builds BioASQ submission records; it contains no metric implementation. |
| `utility/bioasq_official.py` | Runs the bundled official BioASQ Java scorer for aggregate and per-question Phase-B metrics. |
| `utility/evaluation.py` | CLI argument parser for evaluation. |
| `model_registry.py` | Resolves model aliases and artifact paths. |
| `prompt_registry.py` | Resolves prompt bundles. |

## Environment

For a new machine, create the local virtual environment from the project root:

```bash
./setup.sh
source .venv/bin/activate
```

See [`SETUP.md`](../SETUP.md) for migration and verification steps. A pre-existing Conda environment also works:

```bash
conda activate bioasq
```

For offline/local model loading, these environment variables are useful:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UNSLOTH_DISABLE_STATISTICS=1
```

Most commands should also include:

```bash
--local-files-only
```

## SFT Training
Unitutor SFT:
python src/utility/answer_gen_ft.py \
  --preset unitor-llama31-answer-gen \
  --train-input data/BioASQ_list_mr5_visible_gold_sft/train_set_answGEN.json \
  --validation-ratio 0.1 \
  --num-train-epochs 8 \
  --early-stopping-patience 2 \
  --early-stopping-threshold 0.002 \
  --run-name unitutor-8b-mr5-visible-gold-sft-es \
  --local-files-only
SFT finetuned model for list questions:

python src/utility/answer_gen_ft.py \
  --train-input data/training13b.json \
  --question-types list \
  --validation-ratio 0.1 \
  --model-name unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --run-name llama32-3b-bioasq-list-fullcontext-seq8192 \
  --register-alias answer-gen-list-fullcontext \
  --max-resources 0 \
  --max-resource-chars 0 \
  --max-seq-length 8192 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2.0 \
  --learning-rate 2e-4 \
  --dataset-num-proc 1 \
  --local-files-only
