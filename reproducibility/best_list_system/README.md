# Reproducing the best list-question system

This workflow reconstructs the locally best system evaluated on the 83 list
questions in BioASQ Task 13B batches 1-4. The historical official BioASQ score
was **0.5311819 mean F1** (0.5808109 precision and 0.5165600 recall).

The experiment has two SFT dependencies:

1. A full-snippet SFT model generates 12 sampled responses per training
   question. This model supplies diverse candidates only.
2. A first-five-resource, visible-gold SFT model initializes the policy that is
   trained with DPO.

The candidate responses are ranked by BioASQ-style set F1. The v3 filter keeps
whole-response pairs with sufficient F1 and recall separation and controls
large cardinality gaps. Standard sigmoid DPO is then run for one epoch with
`beta=0.1`.

## Required local inputs

The repository intentionally excludes licensed datasets and model weights.
Before running the pipeline, provide:

```text
data/training13b.json
data/Task13BTest/13B1_golden.json
data/Task13BTest/13B2_golden.json
data/Task13BTest/13B3_golden.json
data/Task13BTest/13B4_golden.json
```

The official evaluator JAR is included at:

```text
third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
```

The default base model is
`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. It must be available from the
Hugging Face Hub or the local cache. A CUDA-capable GPU with enough memory for
8B QLoRA training is required.

## Run the pipeline

Install the environment from the repository root, then validate every resolved
command without training:

```bash
./setup.sh
source .venv/bin/activate
./scripts/reproduce_best_list_system.sh --dry-run
```

Run all stages:

```bash
./scripts/reproduce_best_list_system.sh
```

Outputs are written under the ignored directory:

```text
Artifacts/reproductions/best_list_system/
```

The pipeline is resumable. A completed step is skipped when all of its declared
outputs exist. To restart at or run only one named/indexed stage, pass the
pipeline runner options through the wrapper:

```bash
./scripts/reproduce_best_list_system.sh --start-at 6
./scripts/reproduce_best_list_system.sh --only-step 9
```

The stages are:

1. Validate BioASQ inputs and the official evaluator.
2. Export visible-gold first-five-resource SFT records.
3. Create the deterministic question-level split with seed 3407.
4. Train the full-snippet SFT candidate generator.
5. Train the visible-gold SFT policy initialization.
6. Sample 12 candidates per training question.
7. Construct whole-response MR5 v3 preference pairs.
8. Train standard sigmoid DPO.
9. Evaluate greedily with five document resources and the official scorer.
10. Report the reproduced score beside the historical result.

## Expected artifacts and counts

The historical run produced:

| Artifact | Expected count |
|---|---:|
| Visible-gold SFT train/dev questions | 899 / 100 |
| Full-snippet SFT train/dev questions | 947 / 100 |
| Candidate-bank rows | 12,564 |
| Whole-response v3 preference pairs | 2,327 |
| DPO train/eval pairs | 2,212 / 115 |
| Task 13B test list questions | 83 |

See [`expected_result.json`](expected_result.json) for the machine-readable
historical result. Candidate sampling and GPU training are stochastic, so a
fresh run may differ slightly even with seed 3407. The final verifier reports
the deviation without rejecting a valid completed run.

To enforce a tolerance manually:

```bash
python scripts/verify_best_list_reproduction.py result \
  --scores Artifacts/reproductions/best_list_system/evaluation/visible-gold-whole-response-dpo-v3/official_bioasq/official_scores.json \
  --strict-tolerance 0.01
```
