"""Build only notebook 09; leave existing experiment notebooks untouched."""
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
        cell("markdown", """# Controlled answer expressions: construction, expansion and ranking

Construct equivalent answer expressions intentionally under named transformation rules.
At inference, **start from the answering model's prediction**, not from an unknown gold answer.
Expand it in both directions where applicable, verify equivalence, and rank the resulting expressions.
A reranker can only select an expression present in its candidate set.

This notebook has three modes:
- `gold`: construct training variants from one known accepted answer. Other gold aliases are withheld from all API requests. Recovery metrics in this mode are not evidence of inference performance: an accepted answer is already in the pool.
- `judged_c2`: expand existing C2 predictions from notebook 03. This is a development pilot on training questions, not a held-out evaluation.
- `predictions`: use your own JSONL of actual model predictions, including genuinely held-out questions. Gold aliases are optional and used only afterward for diagnostic labels.

Generation and verification receive question, snippets and seed answer. The reranker receives **only question, snippets and the candidate being scored**: it cannot identify the gold seed from an input field, provenance or generation order.

The default is ten training questions and preview only. No API requests, artifacts, model downloads or training occur unless enabled. This notebook prepares data for training a reranker and can apply an existing checkpoint; it does not train one.
"""),
        cell("code", """from pathlib import Path
import json
import sys
from collections import Counter
from IPython.display import display

PROJECT_ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                    if (p / "src/notebook_workflows").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.notebook_workflows.answer_variants import (
    TRANSFORMS, load_examples, read_jsonl, public_context, request_payload,
    run_expansion, rank_candidates, evaluate_and_export,
)
print("Python:", sys.executable)
"""),
        cell("markdown", """## 1. Select seed answers and budget

Use `gold` to construct training data, then `judged_c2` to test reverse expansion from real model wording.
For inference, use `predictions` and set `PREDICTIONS_JSONL` to a file with one row per question:

```json
{"question_id":"q1","question":"Which receptor?","initial_answer":"model prediction","snippets":[{"snippet_id":"1","text":"Your evidence text."}]}
```

An optional `gold_aliases` list supplies post-hoc evaluation targets; it is excluded from API requests and reranker inputs. Custom predictions are not assumed semantically correct.
Keep all variants of a question in one split. Use original dev questions for held-out pipeline evaluation; the existing notebook 03 banks were generated from SFT training questions.

Every attempted API request counts against the shared budget, including failed requests. There are no automatic retries. One generation request is made per example/operation, followed by at most one verification request per proposed variant. Successful requests are cached by their complete payload.
"""),
        cell("code", """SEED_MODE = "gold"  # gold | judged_c2 | predictions
LIMIT = 10
SEED = 3407
RAW_QUESTIONS = PROJECT_ROOT / "data/training13b.json"
ELIGIBLE_QUESTIONS = PROJECT_ROOT / "data/BioASQ_factoid_sft_prepared/split_first_train90_dev10_supported_train_seed3407/train_questions.json"
JUDGMENTS = PROJECT_ROOT / "Artifacts/notebook_runs/pairs/judge_candidates/20260929-030110-b9ec9c29/judgments/candidate_class_judgments.jsonl"
PREDICTIONS_JSONL = None

OPERATIONS = list(TRANSFORMS)  # Remove operations here to run a narrower experiment.
MAX_VARIANTS_PER_OPERATION = 2
GENERATOR_MODEL = "gpt-4.1-mini-2025-04-14"
VERIFIER_MODEL = "gpt-4.1-2025-04-14"
API_KEY_FILE = PROJECT_ROOT / "open_ai_api.txt"  # Plain key file; never display its contents.
MAX_NEW_API_CALLS = 210  # 10 examples * 7 operations * (1 generation + up to 2 verifications)
RUN = False
ALLOW_API = False
PREVIOUS_CACHE = None  # Prior expansion run's cache/ directory; not notebook 03's judge cache.
EXISTING_RUN = None  # Set to a completed notebook 09 run to inspect it without generating again.
OUTPUT_PARENT = PROJECT_ROOT / "Artifacts/notebook_runs/answer_variants"

# Optional: your acceptance-trained local cross-encoder checkpoint.
# None uses a snippet-occurrence baseline, not a learned reranker.
RERANKER_PATH = None
RERANKER_MAX_LENGTH = 4096
"""),
        cell("markdown", """## 2. Inspect transformation rules and examples

These are controlled directions, not random paraphrases. All six C2 relation types are represented; abbreviation expansion has two directions. An operation may abstain. A finite candidate set does not enumerate every possible equivalent expression.
Raw BioASQ snippets and the full raw accepted-alias list are used in built-in modes. These may be broader than the prepared context/alias subset used by notebook 03. Gold aliases are not appended to generated candidate pools.
"""),
        cell("code", """display([{"operation": name, "relation_type": relation, "instruction": instruction}
         for name, (relation, instruction) in TRANSFORMS.items() if name in OPERATIONS])
if SEED_MODE == "predictions":
    if PREDICTIONS_JSONL is None:
        raise ValueError("Set PREDICTIONS_JSONL for prediction-seeded expansion.")
    EXAMPLES = [{**row, "seed_mode": "predictions"} for row in read_jsonl(PREDICTIONS_JSONL)[:LIMIT]]
    for example in EXAMPLES:
        public_context(example)
else:
    EXAMPLES = load_examples(RAW_QUESTIONS, ELIGIBLE_QUESTIONS, mode=SEED_MODE,
                             judgments=JUDGMENTS, limit=LIMIT, seed=SEED)
print("Examples:", len(EXAMPLES), "| Mode:", SEED_MODE)
display([{k: e[k] for k in ("question_id", "question", "initial_answer")} for e in EXAMPLES[:5]])
"""),
        cell("markdown", """## 3. Preview the exact request and call bound

Inspect the request below. In `gold` mode, the one seed answer is deliberately known; the remaining accepted aliases and all class labels are withheld. In `predictions` mode no gold is required.
Structured output validates the response shape. A separate verification call checks the proposed relation and semantic equivalence; this is still model judgment, not a proof. Manually inspect the saved audit before using synthetic variants as training examples.
"""),
        cell("code", """SETTINGS = dict(operations=OPERATIONS, generator_model=GENERATOR_MODEL,
                verifier_model=VERIFIER_MODEL, api_key_file=API_KEY_FILE,
                max_new_calls=MAX_NEW_API_CALLS, max_variants=MAX_VARIANTS_PER_OPERATION,
                previous_cache=PREVIOUS_CACHE)
PREVIEW = run_expansion(EXAMPLES, OUTPUT_PARENT, **SETTINGS)
display(PREVIEW)
print(json.dumps(request_payload(EXAMPLES[0], OPERATIONS[0], model=GENERATOR_MODEL,
                                max_variants=MAX_VARIANTS_PER_OPERATION), indent=2, ensure_ascii=False))
"""),
        cell("markdown", """## 4. Execute expansion and verification

Set `RUN=True` and `ALLOW_API=True` in configuration, then rerun the cells above and this cell.
Each execution creates a fresh directory. If interrupted or budget-limited, use its `cache/` as `PREVIOUS_CACHE` and create a fresh run. Cache-only execution permits `ALLOW_API=False` when a prior cache is supplied.
The original prediction is retained even if no transformation applies. That preserves a baseline, not a guarantee of correctness. Rejected variants remain in `audit.jsonl`; request failures appear in `errors.jsonl`.
"""),
        cell("code", """RUN_DIR = Path(EXISTING_RUN) if EXISTING_RUN else None
if RUN:
    if EXISTING_RUN:
        raise ValueError("Clear EXISTING_RUN before generating a new run.")
    RUN_DIR = run_expansion(EXAMPLES, OUTPUT_PARENT, run=True, allow_api=ALLOW_API, **SETTINGS)
    print("Saved run:", RUN_DIR)
if RUN_DIR:
    display(json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8")))
else:
    print("Preview only. No API calls or output files created.")
"""),
        cell("markdown", """## 5. Review and rank candidates

With `RERANKER_PATH=None`, candidates are ranked by literal, case-insensitive snippet occurrence count; ties use answer text. This is an explicitly simple baseline. It has no gold access.
An acceptance-trained local cross-encoder can replace the baseline. Its input is `(question + snippets, candidate expression)` with no seed identity or class metadata. This cell rejects overlong inputs instead of silently truncating evidence. It needs `sentence-transformers` only when a checkpoint is selected.
Top-five expression coverage here is an offline diagnostic; do not assume five synonyms constitute five distinct BioASQ submission entities.
"""),
        cell("code", """RANKED = []
RUN_EXAMPLES = []
if RUN_DIR:
    status = json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise RuntimeError("Expansion is incomplete. Inspect status/errors and reuse its cache in a fresh run before scoring.")
    RUN_EXAMPLES = read_jsonl(RUN_DIR / "examples.jsonl")
    CANDIDATES = read_jsonl(RUN_DIR / "candidates.jsonl")
    SCORER = None
    if RERANKER_PATH:
        from sentence_transformers import CrossEncoder
        checkpoint = Path(RERANKER_PATH)
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        RERANKER = CrossEncoder(str(checkpoint), max_length=RERANKER_MAX_LENGTH, local_files_only=True)
        def SCORER(pairs):
            for context, candidate in pairs:
                encoded = RERANKER.tokenizer(context, candidate, truncation=False)
                if len(encoded["input_ids"]) > RERANKER_MAX_LENGTH:
                    raise ValueError("Reranker input exceeds RERANKER_MAX_LENGTH; use a supported longer context or explicitly select evidence.")
            return RERANKER.predict(pairs, show_progress_bar=True).reshape(-1).tolist()
    RANKED = rank_candidates(RUN_EXAMPLES, CANDIDATES, scorer=SCORER)
    print("Ranking:", str(RERANKER_PATH) if RERANKER_PATH else "snippet occurrence baseline")
    display(RANKED[:15])
else:
    print("Run expansion or select EXISTING_RUN first.")
"""),
        cell("markdown", """## 6. Label afterward and export training rows

Matching is a **case-insensitive exact diagnostic preserving punctuation**, not a replacement for the official BioASQ evaluator. Multiple accepted aliases are positive. C2 means model-verified equivalence to a known-correct seed without an accepted-string match. For arbitrary predictions, equivalence to the seed alone cannot establish C2, so such rows stay `unverified` unless the seed itself matches gold.

`reranker_train.jsonl` and `reranker_validation.jsonl` contain only question ID, question/snippet context, candidate and acceptance label. A stable question hash assigns 80/20 folds; tiny pilots may have an empty fold. All candidates of a question share a fold. No seed/provenance/rank field is a model feature. Label 0 means not matched by the recorded gold, **not necessarily semantically wrong**. Review synthetic data before training. Gold-free predictions produce rankings but no labeled training rows.

The gold-seeded experiment always retains an accepted answer, so its oracle coverage is trivial. To test inference, compare original accuracy, generated-pool oracle coverage, top-one/top-five results and harms to already-correct answers on held-out **prediction-seeded** examples, without injecting gold candidates.
"""),
        cell("code", """if RUN_DIR and RANKED:
    # Separate scoring outputs per evaluation to preserve earlier baseline/checkpoint reports.
    from datetime import datetime, timezone
    import uuid
    REPORT_DIR = RUN_DIR / ("analysis-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    REPORT_DIR.mkdir(exist_ok=False)
    SUMMARY, LABELED = evaluate_and_export(RUN_EXAMPLES, RANKED, REPORT_DIR, seed=SEED)
    (REPORT_DIR / "ranking_config.json").write_text(json.dumps({
        "reranker": str(RERANKER_PATH) if RERANKER_PATH else "snippet_frequency",
        "max_length": RERANKER_MAX_LENGTH, "fold_seed": SEED,
        "source_run": str(RUN_DIR),
    }, indent=2), encoding="utf-8")
    display(SUMMARY)
    display([{k: r[k] for k in ("question_id", "answer", "operation", "rank", "gold_match", "diagnostic_class")}
             for r in LABELED if r["rank"] <= 5][:25])
    print("Reports and reranker data:", REPORT_DIR)
"""),
        cell("markdown", """## Suggested sequence

1. Review the ten-example gold-seeded pilot and its transformations; use a larger question set only after inspecting the audit.
2. Switch to `judged_c2` to check reverse expansion on real model wording. The relation is not a universally reversible string operation: unsupported alternatives must be omitted.
3. Train a reranker using only `context`, `candidate`, and `label`, keeping question groups intact. Compare against the supplied frequency baseline.
4. Evaluate once on held-out prediction-seeded questions using the same generation rules and a frozen reranker. Report candidate-pool oracle coverage separately from achieved ranking performance. Wrong initial concepts need a separate answer-generation strategy.

API reference: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Model judgment and temperature zero do not guarantee semantic correctness or identical future outputs. Exact request/response caches and operation provenance support reproducibility.
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4, "nbformat_minor": 4}


if __name__ == "__main__":
    path = ROOT / "notebooks/09_controlled_answer_variants.ipynb"
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)
