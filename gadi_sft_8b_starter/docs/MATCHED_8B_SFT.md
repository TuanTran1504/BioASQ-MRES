# Matched original and expansion SFT for three 8B backbones

This workflow prepares six fresh adapters: single-answer SFT and expansion SFT
for Llama-3.1-8B-Instruct, Qwen3-8B and Ministral-3-8B-Instruct-2512.

"Original SFT" means the original single-answer formulation trained as a fresh
matched control, rather than reproducing the historical per-alias training runs.
Both arms have one training record per question. Single-answer targets use
`Answer: [BE]expression[EE]`, selecting the same first official alias that anchors
the expansion target. Expansion targets retain the existing validated lists.

## Controls

Both arms share 1,296 fitting questions, 144 validation questions, exact user
question/snippet JSON and evidence order. Only system instructions and assistant
targets change. The separate 160-question dev set is excluded from fitting and
checkpoint selection. Source hashes and dev IDs are verified against the pinned
strict teacher-validated dataset and its existing official-test overlap audit.

Within each model pair, both arms use the same pinned cached base snapshot,
seed 3407, native chat template, four-bit backbone, rank/alpha 32 LoRA, dropout
0.05, 8,192-token training limit, batch size 1, accumulation 16, learning rate
0.0002, weight decay 0.01, ten warmup updates and three epochs. Early stopping
is disabled to match update budgets. Each arm selects its checkpoint by its own
internal-validation loss at epoch boundaries; absolute losses across arms are
not compared. Longer expansion targets mean supervised tokens and compute are
not matched. Runtime, supervised-token counts, package versions and peak GPU
memory are retained. Qwen thinking is disabled; Ministral adapts language modules
and masks responses using its native `[INST]`/`[/INST]` delimiters.

## Prepare and submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES
git pull --ff-only
cd gadi_sft_8b_starter
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
export HF_HOME=/scratch/nl78/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
python scripts/prepare_matched_8b_sft.py
bash scripts/submit_matched_8b_sft.sh
```

The existing source folder must be present at
`../data/BioASQ_expansion_sft/seed3407_teacher_strict_v1/`, including `train.jsonl`,
`validation.jsonl`, `manifest.json` and `dev_question_ids.json`. The dev export is
`data/expansion_dev160.jsonl`. Git does not transfer these ignored data files.
Missing source data can be copied from Windows:

```powershell
scp -r "C:\Users\Dustin\Projects\BioASQ-MRES\data\BioASQ_expansion_sft\seed3407_teacher_strict_v1" dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/BioASQ-MRES/data/BioASQ_expansion_sft/
```

The helper builds the paired datasets locally on Gadi, checks all data and
caches before the first submission, and pins a shared snapshot per backbone.
It submits **six smoke jobs and six dependent full training jobs**. Each smoke
has a two-hour allocation; each full job has twelve hours on one V100. These
are allocation limits, not runtime estimates. To submit only one model:

```bash
bash scripts/submit_matched_8b_sft.sh qwen3
```

If a backbone is missing, cache it on the login node before submitting:

```bash
python scripts/download_base_model.py --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
python scripts/download_base_model.py --model unsloth/Qwen3-8B-unsloth-bnb-4bit
python scripts/download_base_model.py --model unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit
```

Add `--check-only` to check without downloads, or the existing protected
`--token-file` when needed. GPU jobs load cached models offline. The
[official Ministral guide](https://unsloth.ai/docs/models/tutorials/ministral-3)
requires Transformers v5 and compatible Unsloth. GPU smoke tests check the
actual Gadi stack; local CPU tests do not establish GPU compatibility.

The initial Ministral smoke jobs `180794575` and `180794577` loaded weights but
failed when attaching LoRA. FastModel's scoped target selection interpreted our
fully qualified module paths as projection leaf names and matched no layers.
The corrected loader supplies `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj` and `down_proj`, using its language/vision filters to select language
modules. It then checks that trainable LoRA parameters exist and exclude vision
and projector modules. After pulling this correction, retry only Ministral:

```bash
bash scripts/submit_matched_8b_sft.sh ministral3
```

This submits two fresh smoke jobs and two dependent full jobs. The original
failed smoke/full jobs are retained for audit. The datasets and matrix settings
are unchanged; existing Llama and Qwen jobs need no resubmission for this fix.

## Smoke checks and outputs

Preflight checks every fitting/validation example and refuses truncation. The
loss-mask audit rejects prompt supervision and empty assistant supervision.
Each smoke trains on the longest 16 fitting examples and validates on the
longest eight, then checks generation on four validation questions with its
saved adapter. Zero parseable answers fails the job and holds its dependent full
run. This tests execution/format, not scientific correctness. Full training
starts from the base, rather than continuing the smoke adapter.

Submission IDs and base pins are under `outputs/matched_8b_sft/`. Full adapters
are stored at:

```text
outputs/matched_8b_sft/<llama31|qwen3|ministral3>-8b-<original|expansion>-sft-<PBS_JOBID>/adapter/
```

Smoke directories have a `-smoke` suffix. Monitor the printed IDs with
`qstat -swx <job-id> ...`. Confirm exit status, `status.json` and
`adapter/training_complete.json`; PBS state F alone does not prove success.
Inspect failed smoke logs before resubmitting. Any memory-driven changes must
be logged and applied consistently within each matched pair. Changed matrix
settings require new snapshot pins. Avoid resubmitting active full jobs.

## Compare after training

Predeclare these inference conditions on identical dev questions and snippets:

1. Single-answer SFT with its own prompt and greedy decoding.
2. Single-answer SFT with its own prompt and ten seeded samples at temperature
   0.8, top-p 0.95, top-k 0; first five unique answers in draw order.
3. Expansion SFT with the existing expansion prompt and greedy decoding;
   first five unique candidates in generated order.

Original SFT with the expansion prompt remains a separate prompt-transfer
diagnostic, reporting format compliance separately from answer correctness.
Measure official MRR@5, first-answer accuracy, candidate coverage, parse success,
paired confidence intervals and request/token/runtime costs. Ten samples have a
larger request budget. Sampling expansion SFT is an optional equal-request arm.

This launcher submits training and smoke checks. Full dev inference/scoring
requires subsequent jobs after adapters exist; it is not submitted here.
The initial pilot uses one training seed. Additional seeds and an exposure-audited
unseen test batch are needed for confirmatory research claims.
