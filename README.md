# BioASQ-MRES

Research code for task-structured counterfactual preference mining and evidence-grounded biomedical factoid question answering on BioASQ. The repository covers supervised fine-tuning (SFT), candidate-bank generation, LLM judging, preference-pair construction, multi-stage DPO, synthetic factoid generation, official BioASQ evaluation, and diagnostic analyses.

## What is included

- Python training and evaluation code in `src/` and `cse_dpo/`
- SFT, DPO, annotation, generation, and analysis notebooks in `notebooks/`
- Reusable prompts in `prompts/`
- Experiment launchers and pipeline definitions in `scripts/`
- Unit tests in `tests/`
- BioASQ evaluation utilities in `third_party/Evaluation-Measures/`
- Model registry metadata in `models/registry.json`
- Gadi/HPC launch templates in `gadi_sft_8b_starter/`

Model weights, checkpoints, raw datasets, API credentials, cached model downloads, and generated experiment artifacts are intentionally excluded. See [DATA.md](DATA.md) and [ARTIFACTS.md](ARTIFACTS.md).

## Main workflows

### Evidence-grounded SFT

Use [01 Data preparation](notebooks/01_data_preparation.ipynb) to filter and split
questions, then [04 SFT training](notebooks/04_sft_training.ipynb) to train either
Qwen2.5-0.5B or Qwen2.5-3B on the same files. The default splits all 1,600 factoids
90/10 before filtering: 160 dev questions remain unfiltered, and the training
pool retains 1,137 snippet-matching questions (1,363 alias examples).
The 95 official test factoids stay separate.

The [notebook guide](notebooks/README.md) covers all eight workflows and named
variants. Execution defaults to preview. Original notebooks and their saved
outputs are preserved in the [verified archive](reproducibility/notebook_archive.zip);
see the [migration map](notebooks/MIGRATION.md).

### Candidate banks and annotation

Candidate generation, scoring, GPT-based class annotation, and dataset audits are implemented under `cse_dpo/`. Important entry points include:

- `generate_candidate_bank.py`
- `candidate_bank_class_judge.py`
- `annotate_remaining_candidate_bank_questions.py`
- `extract_factoid_span_pairs_from_candidate_bank.py`
- `classify_stage2_c2_subcategories.py`

### Multi-stage DPO

Use [05 DPO training](notebooks/05_dpo_training.ipynb) for standard, staged,
tie-aware and alternative preference objectives. It calls the existing trainers
with explicit data/model paths and defaults staged training to a smoke run.

### Synthetic factoid generation

The answer-first, source-disjoint synthetic QA pipeline is:

```bash
python -m cse_dpo.build_synthetic_factoid_qa_pilot --help
```

Use [08 Synthetic QA](notebooks/08_synthetic_qa.ipynb) for the controlled generator/verifier pilot. API responses and generated datasets are written under ignored `Artifacts/` paths.

### Evaluation

Evaluation code includes generated-answer scoring, inference strategy comparisons, official BioASQ Java scoring, and error-taxonomy analysis. See:

- `cse_dpo/generated_bioasq_eval.py`
- `src/utility/evaluate_models.py`
- [06 Evaluation](notebooks/06_evaluation.ipynb)
- [07 Analysis](notebooks/07_analysis.ipynb)

### Reproduce the best list-question result

The end-to-end pipeline for the best recorded list-question system trains the
two required SFT adapters, samples the candidate bank, constructs whole-response
MR5 v3 preference pairs, trains standard DPO, and runs official BioASQ scoring:

```bash
./scripts/reproduce_best_list_system.sh --dry-run
./scripts/reproduce_best_list_system.sh
```

The historical score on the 83 Task 13B list questions was **0.5312 mean F1**.
See [reproducibility/best_list_system/README.md](reproducibility/best_list_system/README.md)
for required data, exact stages, expected counts, and resumption instructions.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/TuanTran1504/BioASQ-MRES.git
cd BioASQ-MRES
./setup.sh
source .venv/bin/activate
```

GPU training requires a PyTorch/CUDA combination compatible with the local machine. `unsloth`, `transformers`, `trl`, `datasets`, `accelerate`, `sentence-transformers`, `scikit-learn`, and `bitsandbytes` are listed in `src/requirements.txt`.

Run basic checks with:

```bash
python -m pytest
python -m compileall -q src cse_dpo scripts
```

## Data and models

Place BioASQ data under `data/` and downloaded base models under `models/`, following the paths described in [DATA.md](DATA.md). Training outputs default to `Artifacts/`. These directories are excluded from Git so a local clone can hold large files without accidentally committing them.

For OpenAI-assisted annotation or synthesis, provide credentials through the mechanism documented by the relevant script. The local `open_ai_api.txt` file is ignored and must never be committed.

## Reproducibility notes

- The project generally uses seed `3407` for data splits and generation experiments.
- Real development and test questions are kept separate from synthetic-source selection.
- Active notebooks discover the project root and use explicit input paths. Historical machine-specific configurations remain in the archive.
- Historical run metadata is retained in `models/registry.json`, but the referenced weights and artifacts are not distributed here.

This is research code under active development. Review experiment configurations and data licensing requirements before reuse or redistribution.
