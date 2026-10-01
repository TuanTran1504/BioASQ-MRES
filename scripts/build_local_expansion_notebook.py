"""Build notebook 12: local Qwen2.5-3B fixed-rule expansion pilot."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, text: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build() -> dict:
    cells = [
        cell("markdown", """# Qwen2.5-3B base: extractive candidate expansion

Test whether the unmodified **Qwen2.5-3B-Instruct** model can build a high-recall pool of exact answer spans from the supplied snippets. Generation receives only the question and prepared snippets. Full BioASQ gold aliases are loaded separately for scoring and never enter the prompt.

This experiment measures:

- strict JSON and snippet-citation compliance;
- number and diversity of generated candidates;
- literal containment of every candidate in its cited snippet;
- official exact coverage at 1, 5 and 10 candidates.

The default is a 10-question prompt-only pilot. Run that first because a 3B model may not follow the structured expansion protocol reliably. Set `LIMIT=None` only after inspecting the raw outputs. This notebook uses the base instruct model without the SFT or DPO adapter and performs no training.

Do not run it while notebook 05 is training: both jobs require the same GPU. The execution function checks free CUDA memory before loading the model.
"""),
        cell("code", """from pathlib import Path
import importlib
import json
import sys
from IPython.display import display

PROJECT_ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                    if (p / "src/notebook_workflows").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_workflows.coverage_comparison import load_dev
import src.notebook_workflows.local_expansion as local_expansion
local_expansion = importlib.reload(local_expansion)
DEFAULT_MODEL = local_expansion.DEFAULT_MODEL
EXTRACTIVE_EXPANSION_PROMPT = local_expansion.EXTRACTIVE_EXPANSION_PROMPT
analyze_local_expansion = local_expansion.analyze_local_expansion
run_local_expansion = local_expansion.run_local_expansion
print("Project:", PROJECT_ROOT)
print("Python:", sys.executable)
"""),
        cell("markdown", """## 1. Configuration

The complete 160-question dev set was measured with the Qwen tokenizer. The longest fixed-rule prompt containing every snippet is 4,817 tokens. `MAX_SEQ_LENGTH=6144` therefore preserves every snippet and reserves 512 tokens for the JSON answer list with additional headroom. This prompt-only base-model test is separate from the 4096-token SFT/DPO setting.

`LOCAL_FILES_ONLY=True` prevents an unexpected download. The same model is already used by the current 3B DPO run and should be cached locally.
"""),
        cell("code", """DATA_DIR = PROJECT_ROOT / "data/BioASQ_factoid_sft_prepared/split_first_train90_dev10_supported_train_seed3407"
DEV_PATH = DATA_DIR / "dev_prepared.json"
TRAIN_PATH = DATA_DIR / "train_questions.json"
RAW_PATH = PROJECT_ROOT / "data/training13b.json"

MODEL_NAME = DEFAULT_MODEL  # unmodified Qwen2.5-3B-Instruct 4-bit base
MAX_SEQ_LENGTH = 6144
MAX_NEW_TOKENS = 512
LIMIT = 10                 # None runs all 160 dev questions
LOCAL_FILES_ONLY = True
MIN_FREE_CUDA_GB = 6.0
REQUIRE_ALL_SNIPPETS = True
RESPONSE_MODE = "extractive"

RUN = False
EXISTING_RUN = None
OUTPUT_PARENT = PROJECT_ROOT / "Artifacts/notebook_runs/local_expansion/qwen25_3b_base"
EVALUATOR_JAR = PROJECT_ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
"""),
        cell("markdown", """## 2. Load the held-out questions and inspect the extractive prompt

The model must copy every answer character-for-character from one cited snippet. It searches for minimal direct spans, qualified spans, canonical surfaces, abbreviations, numeric surfaces and alternative evidence spans. The Python validator rejects the complete response if any retained candidate is not literally contained in its cited snippet.
"""),
        cell("code", """ALL_EXAMPLES = load_dev(DEV_PATH, RAW_PATH, train_path=TRAIN_PATH, expected_count=160)
EXAMPLES = ALL_EXAMPLES if LIMIT is None else ALL_EXAMPLES[:LIMIT]
SETTINGS = dict(
    model_name=MODEL_NAME,
    system_prompt=EXTRACTIVE_EXPANSION_PROMPT,
    max_seq_length=MAX_SEQ_LENGTH,
    max_new_tokens=MAX_NEW_TOKENS,
    local_files_only=LOCAL_FILES_ONLY,
    min_free_cuda_gb=MIN_FREE_CUDA_GB,
    require_all_snippets=REQUIRE_ALL_SNIPPETS,
    response_mode=RESPONSE_MODE,
)
PREVIEW = run_local_expansion(EXAMPLES, OUTPUT_PARENT, **SETTINGS)
display(PREVIEW)
print("\\nEXTRACTIVE EXPANSION PROMPT:\\n")
print(EXTRACTIVE_EXPANSION_PROMPT)
print("\\nFirst gold-blind user payload:\\n")
print(json.dumps({"question": EXAMPLES[0]["question"],
                  "snippets": [{"id": s["snippet_id"], "text": s["text"]}
                               for s in EXAMPLES[0]["snippets"]]},
                 indent=2, ensure_ascii=False))
"""),
        cell("markdown", """## 3. Generate expansions locally

Wait until the active DPO run has finished, then set `RUN=True` and execute this cell. No API key or `ALLOW_API` setting is needed.

Every raw model response is saved. Invalid JSON, unknown candidate types, invented snippet IDs and non-verbatim candidates are counted as parse failures rather than repaired. This is part of the base model's measured performance.
"""),
        cell("code", """RUN_DIR = Path(EXISTING_RUN) if EXISTING_RUN else None
if RUN:
    if EXISTING_RUN:
        raise ValueError("Clear EXISTING_RUN before starting a new run.")
    RUN_DIR = run_local_expansion(EXAMPLES, OUTPUT_PARENT, run=True, **SETTINGS)
if RUN_DIR:
    print("Run:", RUN_DIR)
    display(json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8")))
else:
    print("Preview only. Set RUN=True after the DPO job finishes.")
"""),
        cell("markdown", """## 4. Score the generated candidate pool

Coverage is measured by applying the official BioASQ matcher independently to every candidate. `coverage_at1` measures the top-ranked extractive span; `coverage_at5` and `coverage_at10` show whether evidence-span expansion recovers additional accepted answer forms.

This is oracle candidate-pool coverage. There is no reranker or semantic judge in this pilot.
"""),
        cell("code", """REPORT_DIR = None
SUMMARY = None
QUESTION_RESULTS = []
if RUN_DIR:
    REPORT_DIR, SUMMARY, QUESTION_RESULTS = analyze_local_expansion(RUN_DIR, jar_path=EVALUATOR_JAR)
    display(SUMMARY)
    print("Expansion gain at 10 over original answer (percentage points):",
          100 * (SUMMARY["coverage_at10"] - SUMMARY["coverage_at1"]))
    print("Report:", REPORT_DIR)
else:
    print("Run the pilot or select EXISTING_RUN before scoring.")
"""),
        cell("markdown", """## 5. Inspect gains, failures and relation types

Inspect raw answers before deciding whether the base model is suitable. Exact coverage can improve even if some additional candidates are semantically invalid. Conversely, a valid synonym can remain unmatched by the benchmark.
"""),
        cell("code", """if QUESTION_RESULTS:
    import pandas as pd
    frame = pd.DataFrame(QUESTION_RESULTS)
    columns = ["question", "gold_aliases", "answers", "candidate_types", "snippet_ids",
               "matching_answers", "extractive_answers", "snippets_truncated"]
    print("Recovered by expansion after the original answer missed:")
    display(frame.loc[(~frame.coverage_at1) & frame.coverage_at10, columns])
    print("Still not covered:")
    display(frame.loc[~frame.coverage_at10, columns])
    print("Parse failures:")
    display(frame.loc[~frame.parse_success, ["question_id", "question"]])
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    path = ROOT / "notebooks/12_qwen25_3b_base_expansion.ipynb"
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
