"""Build only the GPT-4.1 mini dev160 coverage comparison notebook."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    cells = [
        cell("markdown", """# GPT-4.1 mini: extractive expansion versus high-temperature sampling

**Question:** Does asking for a high-recall pool of exact snippet spans improve gold-answer coverage compared with ten independent single-answer generations?

| Arm | Requests per question | Temperature | Candidate pool |
|---|---:|---:|---|
| Extractive expansion | 1 | 0.0 | Up to 10 distinct answer spans copied exactly from cited snippets |
| Independent sampling | 10 | 1.2 | One answer per request; each request receives the same question/snippets and no previous answers |

Both arms use **gpt-4.1-mini-2025-04-14** and the same complete prepared snippets for all **160 original dev questions**. No gold answer seeds generation. Prepared single-answer SFT instructions are excluded. Full accepted aliases from the raw BioASQ data are loaded for scoring only.

The primary result is **oracle gold coverage@10**: how many questions have at least one accepted answer in the pool? There is no reranker, semantic judge or answer-text repair. Every expansion candidate is checked programmatically against all supplied snippets before it can enter the pool. A wrong snippet ID is corrected only when the answer is an exact span in another supplied snippet.

This compares two practical generation strategies at a maximum of ten candidate slots, not equal API/token cost. The extractive arm searches across snippets and may return different plausible concepts, but it cannot exactly recover any accepted gold alias absent from the supplied snippets.

Execution is off by default. A full uncached trial makes **1,760 requests**. Exact requests and responses are cached per question/arm/draw so the ten sampling calls cannot collapse into one cached response.
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
import src.notebook_workflows.coverage_comparison as coverage_comparison
importlib.reload(coverage_comparison)
from src.notebook_workflows.coverage_comparison import (
    MODEL, EXTRACTIVE_EXPANSION_PROMPT, SINGLE_PROMPT, load_dev, request_payload,
    run_comparison, analyze_run,
)
print("Project:", PROJECT_ROOT)
print("Python:", sys.executable)
"""),
        cell("markdown", """## 1. Configuration

All 160 dev questions are retained, including the 39 without a normalized gold-alias occurrence in their snippets. No test questions or training-bank predictions are used.

Keep the model fixed between arms. Temperatures are experimental choices and editable. Use `TRIAL_ID` to distinguish independent repetitions. For an interrupted run, keep the same trial ID and prompts and set `PREVIOUS_CACHE` to its `cache/` directory. Change the trial ID for genuinely new draws, even if you reuse a prior cache directory.
"""),
        cell("code", """DATA_DIR = PROJECT_ROOT / "data/BioASQ_factoid_sft_prepared/split_first_train90_dev10_supported_train_seed3407"
DEV_PATH = DATA_DIR / "dev_prepared.json"
TRAIN_PATH = DATA_DIR / "train_questions.json"
RAW_PATH = PROJECT_ROOT / "data/training13b.json"
EXPECTED_DEV_QUESTIONS = 160

MODEL_NAME = MODEL  # gpt-4.1-mini-2025-04-14
EXPANSION_MODE = "extractive"
EXPANSION_TEMPERATURE = 0.0
SAMPLING_TEMPERATURE = 1.2
TRIAL_ID = "dev160-v1"
API_KEY_FILE = PROJECT_ROOT / "open_ai_api.txt"
REQUEST_DELAY_SECONDS = 0.25
MAX_TRANSPORT_RETRIES = 3  # Only connection/timeouts and HTTP 408/500/502/503/504; each attempt counts.

RUN = False
ALLOW_API = False
REUSE_BASELINE_CACHE = PROJECT_ROOT / "Artifacts/notebook_runs/coverage_comparison/20260929-070458-a6267ca4/cache"
PREVIOUS_CACHE = REUSE_BASELINE_CACHE if REUSE_BASELINE_CACHE.is_dir() else None
# With the prior cache, all 1,600 sampling calls are reused and only 160 new expansion calls are needed.
MAX_NEW_API_CALLS = 160 if PREVIOUS_CACHE else 1760
EXISTING_RUN = None    # Completed notebook 10 run to score/reinspect without generation.
OUTPUT_PARENT = PROJECT_ROOT / "Artifacts/notebook_runs/coverage_comparison"
EVALUATOR_JAR = PROJECT_ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
"""),
        cell("markdown", """## 2. Inspect the extractive and baseline prompts

The extractive prompt searches all snippets for up to ten exact answer spans. Each candidate returns `answer`, `snippet_id` and `candidate_type`. It does not receive an accepted answer or a model answer generated in a separate call. The baseline makes ten independent calls with the unchanged single-answer prompt.

Both arms use structured JSON. The API constrains the expansion fields and candidate types; local validation additionally proves literal containment in the cited snippet. Neither arm sees gold aliases. Repeated outputs remain spent candidate slots.
"""),
        cell("code", """# Edit these strings here if you want a different prompt, then rerun preview.
EXPANSION_SYSTEM = EXTRACTIVE_EXPANSION_PROMPT
SINGLE_ANSWER_SYSTEM = SINGLE_PROMPT
print("EXPANSION PROMPT:\\n", EXPANSION_SYSTEM)
print("SINGLE-ANSWER PROMPT:\\n", SINGLE_ANSWER_SYSTEM)
"""),
        cell("markdown", """## 3. Load dev questions and preview exact requests

This checks the expected question count, unique IDs, factoid type and disjointness from training IDs. It uses only `input_1` and snippet resources (`input_2`, etc.) for model input; `instruction`, `output`, supported-alias metadata and raw gold are excluded.

No context truncation or snippet selection is performed. Both arms see identical evidence. The raw gold list is kept outside the request for later scoring.
"""),
        cell("code", """EXAMPLES = load_dev(DEV_PATH, RAW_PATH, train_path=TRAIN_PATH, expected_count=EXPECTED_DEV_QUESTIONS)
SETTINGS = dict(model=MODEL_NAME, expansion_temperature=EXPANSION_TEMPERATURE,
                sampling_temperature=SAMPLING_TEMPERATURE, expansion_prompt=EXPANSION_SYSTEM,
                single_prompt=SINGLE_ANSWER_SYSTEM, expansion_mode=EXPANSION_MODE, trial_id=TRIAL_ID,
                api_key_file=API_KEY_FILE, max_new_calls=MAX_NEW_API_CALLS,
                previous_cache=PREVIOUS_CACHE, request_delay=REQUEST_DELAY_SECONDS,
                max_transport_retries=MAX_TRANSPORT_RETRIES)
PREVIEW = run_comparison(EXAMPLES, OUTPUT_PARENT, **SETTINGS)
display({k: v for k, v in PREVIEW.items() if not k.endswith("prompt")})
display({"dev_questions": len(EXAMPLES), "total_snippets": sum(len(e["snippets"]) for e in EXAMPLES),
         "maximum_snippet_characters_per_question": max(sum(len(s["text"]) for s in e["snippets"]) for e in EXAMPLES)})
for arm in ("expansion", "sampling"):
    payload = request_payload(EXAMPLES[0], arm, model=MODEL_NAME,
                              expansion_temperature=EXPANSION_TEMPERATURE,
                              sampling_temperature=SAMPLING_TEMPERATURE,
                              expansion_prompt=EXPANSION_SYSTEM, single_prompt=SINGLE_ANSWER_SYSTEM,
                              expansion_mode=EXPANSION_MODE)
    print("\\n", arm, "example request:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
"""),
        cell("markdown", """## 4. Run both arms

Set `RUN=True` and `ALLOW_API=True`, then rerun configuration, prompts, preview and this cell. The API key is read from its file and never written to artifacts. The configured previous cache reuses the exact 1,600 sampling responses from the original experiment; changed expansion payloads cannot collide with the old expansion cache entries.

Temporary connection/timeouts and HTTP 408/500/502/503/504 failures retry up to `MAX_TRANSPORT_RETRIES` times with backoff. Every attempt consumes the same hard request budget. Refusals, malformed JSON/schema responses, authentication errors and other request failures are not retried; successful responses are never resampled. There are no padding calls. In extractive mode, a wrong snippet citation is corrected when the exact span occurs in another supplied snippet. A candidate absent from every supplied snippet is excluded and written to `invalid_candidates.jsonl`; it does not abort the run. If retries are exhausted or execution is interrupted, successful calls remain cached and the run is marked incomplete. Resume into a fresh directory using `PREVIOUS_CACHE` with the same trial ID/settings. The default 1,760-attempt budget can require a resumed run when retries consume some of it.

For scoring only, leave `RUN=False` and set `EXISTING_RUN` to a completed run. Each new run keeps its prompts, inputs, candidates, request usage and status. API prompt caching may still reduce provider input cost; that does not reuse the sampled answer.
"""),
        cell("code", """RUN_DIR = Path(EXISTING_RUN) if EXISTING_RUN else None
if RUN:
    if EXISTING_RUN:
        raise ValueError("Clear EXISTING_RUN to generate a fresh comparison.")
    RUN_DIR = run_comparison(EXAMPLES, OUTPUT_PARENT, run=True, allow_api=ALLOW_API, **SETTINGS)
if RUN_DIR:
    print("Run:", RUN_DIR)
    display(json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8")))
else:
    print("Preview only: no API calls or run files created.")
"""),
        cell("markdown", """## 5. Score coverage using the official BioASQ matcher

Java and a JDK must be available for the repository's existing official scorer adapter. Each unique question/candidate combination is evaluated as a separate **single-candidate** answer against all original gold aliases. Coverage@10 is then the union of those acceptance results, rather than submitting ten candidates to an evaluator designed for top-five submissions.

Results include coverage@1/@5/@10 in original generation order, mean unique candidates, request/token usage, and paired outcomes: both, expansion only, sampling only, neither. This is an offline candidate-pool experiment, not an official ten-answer submission. No reranking is applied; sampling order is draw order, not a confidence ranking.
"""),
        cell("code", """REPORT_DIR = None
SUMMARY = None
QUESTION_RESULTS = []
if RUN_DIR:
    REPORT_DIR, SUMMARY, QUESTION_RESULTS = analyze_run(RUN_DIR, jar_path=EVALUATOR_JAR)
    import pandas as pd
    display(pd.DataFrame(SUMMARY["methods"]))
    display(SUMMARY["paired_outcomes"])
    print("Expansion minus sampling coverage@10 (percentage points):",
          SUMMARY["coverage_at10_difference_percentage_points"])
    print("Reports:", REPORT_DIR)
else:
    print("Select or execute a complete run before scoring.")
"""),
        cell("markdown", """## 6. Inspect questions gained and lost

Use these rows to distinguish gains from minimal answer boundaries, alternative source surfaces and alternative evidence concepts. Matching a gold alias establishes benchmark acceptance. Literal containment establishes provenance but does not by itself prove that a span answers the question correctly.
"""),
        cell("code", """if QUESTION_RESULTS:
    import pandas as pd
    frame = pd.DataFrame(QUESTION_RESULTS)
    columns = ["question_id", "question", "gold_aliases", "expansion_answers", "sampling_answers",
               "expansion_matching_answers", "sampling_matching_answers"]
    print("Covered only by expansion:")
    display(frame.loc[frame.outcome == "expansion_only", columns])
    print("Covered only by independent sampling:")
    display(frame.loc[frame.outcome == "sampling_only", columns])
    print("Neither method covers:", int((frame.outcome == "neither").sum()))
"""),
        cell("markdown", """## Interpreting this experiment

- Primary comparison: coverage@10 over all 160 dev questions. Also inspect absolute counts, the paired gains/losses and diversity rather than only the percentage difference.
- Ten candidates is a cap, not a quota for expansion. If only three defensible exact spans exist, the arm returns three. Sampling always spends ten draws, even if many are duplicates.
- The extractive arm has a hard exact-coverage ceiling on the 39 dev questions where no accepted gold alias occurs in the snippets. Evaluate that group separately.
- Equal maximum candidate count is not equal compute: one expansion call versus ten sampling calls, with different output limits. Request and token totals are reported, including separate newly spent totals on resumed runs.
- An improvement makes expansion a promising candidate-pool strategy. No improvement weakens that case; it does not prove a reranker could not improve selection from either existing pool.
- A single trial is exploratory. Freeze prompts/settings, change `TRIAL_ID`, and collect independent repetitions before making a strong claim. Do not tune against official test batches.
- This notebook intentionally does not add a verifier or reranker, since either would change the experiment. Notebook 09 remains available for the separate controlled-construction/reranking study.

References: [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4, "nbformat_minor": 4}


if __name__ == "__main__":
    path = ROOT / "notebooks/10_gpt41mini_coverage_comparison.ipynb"
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
