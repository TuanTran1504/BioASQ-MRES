# Transfer and Run Procedure

## Transfer

From the local workstation, transfer only this starter bundle. Generate the ignored expansion data first when running the 8B expansion experiment:

```bash
python scripts/export_gadi_expansion_dev.py
python gadi_sft_8b_starter/scripts/verify_expansion_bundle.py
```

```bash
cd "/home/dinh-tuan/Documents/Project/BioASQ_MRES/Task-Structured Counterfactual Preference Mining"
rsync -av --progress gadi_sft_8b_starter/ \
  dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/gadi_sft_8b_starter/
```

On Gadi:

```bash
cd /scratch/nl78/$USER/gadi_sft_8b_starter
module load python3/3.12.13
python3 scripts/verify_bundle.py
```

## Environment and model cache

Create the virtual environment once on a login node. The package uses the currently available `python3/3.12.13` module; use another modern version only if that module changes.

```bash
PYTHON_MODULE=python3/3.12.13 bash scripts/setup_environment.sh
```

The setup check deliberately does not import Unsloth: login nodes have no GPU.
Unsloth is validated by the GPU smoke-test job below. If setup previously ended
with `Unsloth cannot find any torch accelerator`, the package installation has
already completed; copy the updated bundle and proceed to the smoke test.

The 8B base model is intentionally excluded from the transfer. It must be cached once under `/scratch/nl78/$USER/hf_cache`. Access to Meta Llama on Hugging Face may require accepting the model license first.

```bash
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
export HF_HOME=/scratch/nl78/$USER/hf_cache
export HF_TOKEN=<your-token>
python3 scripts/download_base_model.py
unset HF_TOKEN
```

## Submit jobs

Confirm the actual GPU queue available to `nl78` before submission. The supplied PBS files use `gpuvolta` as a placeholder-compatible starting point; change that line if PBS rejects it.

```bash
qstat -Q
qsub jobs/00_gpu_smoke_test.pbs
qstat -swx <job-id>
```

After a successful smoke test:

```bash
qsub jobs/train_full_resources_8b.pbs
# or
qsub jobs/train_evidence_grounded_8b.pbs
```

For the prompt-only exact-span expansion experiment, use its separate smoke test and full job:

```bash
qsub jobs/01_expansion_smoke_test.pbs
qsub jobs/run_extractive_expansion_8b.pbs
```

The expansion run writes after every completed question. If PBS stops a run before completion, submit a copy of the full job whose final command includes:

```bash
python3 scripts/run_extractive_expansion_8b.py \
  --resume-run outputs/expansion/<incomplete-run-directory>
```

Every run is stored independently under `outputs/runs/`. The resulting manifest records the exact command-line configuration, metrics, best generated-dev-MRR checkpoint, and best eval-loss checkpoint.

## Retrieve results

Copy only selected completed outputs back to the workstation. Usually this means the adapter, `generated_dev_selection/`, `eval_loss_selection/`, `training_curves.png`, and the run manifest.

```bash
rsync -av --progress \
  dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/gadi_sft_8b_starter/outputs/runs/<run-id>/ \
  ./gadi_results/<run-id>/
```

For expansion results, replace `outputs/runs/<run-id>/` with `outputs/expansion/<run-id>/`.
