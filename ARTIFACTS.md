# Models and experiment artifacts

The following local content is excluded from Git:

- downloaded base models under `models/<model-name>/`
- LoRA adapters and checkpoints
- trainer state and optimizer files
- candidate banks and preference datasets
- API response caches and judgments
- generated evaluations, plots, and run directories under `Artifacts/`
- ablation outputs under `Abalations/`

Portable model metadata remains in `models/registry.json`. To reproduce a run, download the named base model, reconstruct the referenced dataset with the repository scripts, and run the corresponding notebook or CLI configuration.

Curated aggregate scores and source hashes are versioned under `results/`.
See [the results summary](results/README.md). The exporter
`scripts/export_main_results.py` reads the original ignored local artifacts;
regenerating the snapshot requires those artifacts to be present.

Sanitized model-generation snapshots may also be versioned under
`results/expansion_generations/`. These snapshots omit BioASQ question text,
snippets, gold answers, prompts, model weights and checkpoints. Their source
artifacts remain ignored under `Artifacts/`.

For sharing trained weights, use a model registry such as Hugging Face Hub or a versioned object store and document the model URL and checksum in the repository. GitHub source control should contain configuration and metadata rather than multi-gigabyte weight files.
