# Data layout

Datasets are intentionally excluded from Git because they are large, generated, or governed by their original distribution terms.

Expected local layout:

```text
data/
├── training13b.json
├── Task13BTest/
├── BioASQ_factoid_sft_prepared/
└── ...
```

Obtain BioASQ data from the official BioASQ distribution and follow its terms of use. Preparation notebooks and scripts create the derived SFT, DPO-split, candidate-generation, and synthetic-data views used by this project.

Representative preparation entry points:

- `notebooks/prepare_factoid_evidence_answer_sft.ipynb`
- `cse_dpo/split_gold_supported_sft_dpo.py`
- `cse_dpo/build_candidate_bank_gold_snippet_dpo_inputs.py`
- `cse_dpo/build_synthetic_factoid_qa_pilot.py`

The `.gitignore` permits this directory to be populated locally without staging its contents.
