# Research notebooks

Eight notebooks share the same configuration and execution layer. Select a preset,
set input paths in **OVERRIDES**, and inspect the preview. Nothing executes until
**RUN=True**; API work also requires **ALLOW_API=True**.

Notebook 09 is a separate controlled-expression pilot with its own preview, budget,
cache, verification and export cells. Rebuild only that notebook with
`python scripts/build_answer_variants_notebook.py`; the eight-workflow builder
leaves it alone.

Notebook 10 compares GPT-4.1 mini rule-guided expansion (up to ten expressions in
one request) with ten independent high-temperature single-answer requests on all
160 original dev questions. It reports gold coverage using the official matcher,
paired gains/losses and token usage, without a verifier or reranker. Its default
budget is 1,760 API requests and execution is off. Rebuild only it with
`python scripts/build_coverage_comparison_notebook.py`.

Notebook 11 creates the publication-style SFT training comparison from saved
0.5B and 3B run histories without loading either model. Rebuild only it with
`python scripts/build_sft_training_plot_notebook.py`.

| Notebook | Main variants |
|---|---|
| [01 Data preparation](01_data_preparation.ipynb) | Snippet filtering, full-resource data, train/dev and SFT/DPO splits, evidence exports |
| [02 Candidate generation](02_candidate_generation.ipynb) | Models, sample counts, prompts, merging, filtering, log-probability scoring |
| [03 Judging and pairs](03_judging_and_preference_pairs.ipynb) | Class judging, factoid negatives, spans, list-response pairs, curriculum and ties |
| [04 SFT](04_sft_training.ipynb) | Model sizes; answer-only, evidence-plus-answer, rationale and control arms |
| [05 DPO](05_dpo_training.ipynb) | Standard, staged, DPO-D retention, Cal-DPO, APO, softmax, error-aware objectives |
| [06 Evaluation](06_evaluation.ipynb) | Dev/test, multiple models, greedy, sampled aggregation, conditioned sampling, API baselines |
| [07 Analysis](07_analysis.ipynb) | Bank comparisons, evidence coverage, model/preference diagnostics, Stage 1 audits |
| [08 Synthetic QA](08_synthetic_qa.ipynb) | Prepare, generate, verify, finalize; generator/verifier variants |
| [09 Controlled answer variants](09_controlled_answer_variants.ipynb) | Gold-seeded training construction; prediction-seeded expansion; explicit relation/direction controls; verification; optional reranker; top-1/top-5 diagnostics and training exports |
| [10 GPT-4.1 mini coverage comparison](10_gpt41mini_coverage_comparison.ipynb) | Full dev160: rule-guided expansion versus ten high-temperature single-answer draws; official candidate matching and oracle coverage |
| [11 SFT training plot](11_sft_training_plot.ipynb) | Reproducible optimization, validation-loss and generated-dev MRR comparison for saved 0.5B/3B SFT runs |

## Your Qwen 0.5B / 3B experiment

1. In notebook 01 select **snippet_supported**, keeping dev_ratio=0.1 and seed=3407.
   Execute and copy its data output directory.
2. In notebook 04 select **answer_only** and configure:

   ~~~python
   OVERRIDES = {
       "train_input": ["<data output>/train_prepared.json"],
       "eval_input": ["<data output>/dev_prepared.json"],
   }
   VARIANTS = [
       {"label": "qwen05b", "overrides": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}},
       {"label": "qwen3b", "overrides": {"model_name": "Qwen/Qwen2.5-3B-Instruct"}},
   ]
   ~~~

3. Preview both plans. Select your GPU training kernel, set RUN=True, and rerun
   configuration, preview and execution. Models run sequentially.
4. Use notebook 06 with dev data during selection; keep the official test batches
   for final evaluation.

The default split randomly samples all sorted factoid **question IDs**, without
stratification, using seed 3407. It is reproducible and independent of source-file
row order. Split the 1,600 questions into 1,440 train and 160 dev first, then
filter only training: **1,137 train questions (1,363 examples), 160 dev questions**,
and **95 separate official test questions**. Dev contains 121 questions with a
normalized snippet match and 39 without, retaining all accepted gold aliases.
Matching checks lexical occurrence, not semantic support.

A verified copy of the current split is at
data/BioASQ_factoid_sft_prepared/split_first_train90_dev10_supported_train_seed3407/.
Notebook 04 points to these files and requires RUN=True after reviewing new plans.
The earlier 1,132-train/126-dev files under
data/BioASQ_factoid_sft_prepared/snippet_supported_train90_dev10_seed3407/ and the
completed model runs are intact. Select supported_dev_legacy in notebook 01 to
reproduce the earlier filter-first protocol. Existing models were trained on
different question IDs; use fresh training runs for this new protocol.
New splits include train_questions.json for candidate generation.
Do not use the alias-expanded SFT file to generate candidate banks.

## Configuration and execution

- **PRESET** selects a named method configuration.
- **OVERRIDES** changes inputs, model, prompt, seed, objective or hyperparameters.
- **VARIANTS** compares parameter changes sequentially with shared settings.
- **describe(METHOD)** shows options without importing the training stack.
- Outputs go under Artifacts/notebook_runs/. Every execution records plan.json,
  input_hashes.json, status.json and run.log.
- Unknown options, invalid choices, unresolved inputs and common question-overlap
  cases fail before model execution. Output overrides, automatic resume and
  existing run-directory reuse are rejected.
- API methods require positive batch/question limits. Some underlying methods
  retry invalid/provider responses, so limits are not universally strict counts
  of HTTP requests.
- Input files are hashed during preview and checked again at execution. Directory inputs such as adapters are
  recorded as paths; underlying model/data manifests provide more provenance.

Staged DPO defaults to a smoke run. Data/preview checks do not replace CUDA
validation. Java is required for official scoring. Install the selected
operation's normal project dependencies in the notebook kernel environment.

Notebook 06 can evaluate local adapters and OpenAI candidate models in one run.
Use `openai:<model-name>` inside the same `model_ref` list as local models.
The grounded-semantic option classifies non-exact factoid answers and separately
checks whether supplied snippets support them. Candidate generation and judging
use separate explicit API budgets and reusable cache directories; neither API is
enabled unless `ALLOW_API=True`.

## Historical experiments and maintenance

[MIGRATION.md](MIGRATION.md) maps every former notebook to the shared workflows.
The original 42 notebooks are preserved byte-for-byte in
[notebook_archive.zip](../reproducibility/notebook_archive.zip), verified by
[SHA-256 hashes](../reproducibility/notebook_archive_manifest.json).
Previously deleted notebooks remain deleted. Exact historical plots,
configurations and saved outputs remain available in the archive.

Shared implementations live in src/notebook_workflows/. Existing ML algorithms
remain in src/ and cse_dpo/. Add named variants in presets.py, methods in catalog.py
and shared notebook operations in operations.py. Regenerate with:

~~~bash
python scripts/build_workflow_notebooks.py
~~~

Regeneration resets notebook edits and outputs; preserve custom configurations first.

## Validation of the consolidation

108 focused CPU tests passed, including every preset, execution guards, question
separation, configurable rationale/sample counts, evidence exports, notebook
preview execution and archive integrity. The real-data default preparation and
official-test evidence coverage workflows completed successfully.

~~~bash
python -m pytest tests/test_notebook_workflows.py tests/test_factoid_snippet_sft.py tests/test_candidate_bank_pair_dedup.py tests/test_gold_supported_sft_dpo_split.py tests/test_synthetic_factoid_qa_pilot.py -q
~~~

GPU training, model inference and paid API workflows were not executed during
this refactor. Broader legacy checks in test_candidate_bank_strict_equivalence.py
have three unresolved failures: one requires an absent historical prepared-data
file, and two supply judge-response fixtures that no longer match the current
seven-field response schema.
