# Bundle Provenance

Created on 7 September 2026 from the local BioASQ MRES workspace.

This bundle packages only the components required to reproduce the initial 8B factoid SFT experiments:

- prepared full-resource and evidence-grounded SFT data;
- the single-answer prompt registry;
- the SFT pipeline and its direct runtime dependencies;
- the official BioASQ Java evaluator distribution;
- Gadi-specific environment, model-cache, smoke-test, and PBS launch helpers.

It intentionally excludes all existing model weights, checkpoints, DPO scripts/data, candidate banks, test runs, notebooks, and historical artifacts. The base model is downloaded separately from Hugging Face into the Gadi cache.

The 8B settings are based on the existing local Llama run configuration in `Artifacts/models/runs/20260820-125510-bioasq-8b-factoid-full-resources-single-answer-per-alias-sft/manifest.json`, updated to use LoRA dropout 0.05 to match the current 0.5B notebook setup.
