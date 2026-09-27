Model management in this repo uses two pieces:

1. `models/registry.json`
   This is the source of truth for tracked fine-tuning runs and promoted aliases.

2. `Artifacts/models/runs/<run_id>/manifest.json`
   Each managed training run writes its own manifest with paths, settings, and status.

Useful commands:

```bash
python src/manage_models.py list-runs
python src/manage_models.py show-run <run_id>
python src/manage_models.py promote <run_id> answer-gen-current
python src/manage_models.py resolve answer-gen-current
```

Recommended workflow:

```bash
python src/utility/answer_gen_ft.py ...
python src/manage_models.py list-runs
python src/manage_models.py promote <run_id> answer-gen-llama32-3b
```
