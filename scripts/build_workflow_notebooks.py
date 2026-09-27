"""Regenerate the eight thin workflow notebooks from shared presets."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.notebook_workflows.presets import PRESETS

WORKFLOWS = [
    ("01_data_preparation", "data", "Data preparation and splitting", "snippet_supported"),
    ("02_candidate_generation", "generation", "Candidate generation and scoring", "sample20"),
    ("03_judging_and_preference_pairs", "pairs", "Judging and preference pairs", "standard_factoid"),
    ("04_sft_training", "sft", "Supervised fine-tuning", "answer_only"),
    ("05_dpo_training", "dpo", "Preference training", "two_stage"),
    ("06_evaluation", "evaluation", "Evaluation and inference comparisons", "greedy"),
    ("07_analysis", "analysis", "Candidate, evidence and model diagnostics", "test_evidence_coverage"),
    ("08_synthetic_qa", "synthetic", "Synthetic factoid generation", "prepare"),
]

GUIDES = {
    "data": """The default starts from all 1,600 factoid questions in data/training13b.json and randomly splits sorted IDs **without stratification**, using seed 3407. The initial split contains 1,440 train and 160 dev questions. Dev remains unfiltered: **121 snippet-matching and 39 non-matching questions**. Only the training split is filtered, retaining **1,137 questions / 1,363 alias rows**. The 95 official test questions stay separate.

split_order=split_first is the default. The supported_dev_legacy preset (split_order=filter_first) reproduces the earlier 1,132-train/126-dev protocol. Changing training filter mode with split_first never changes dev membership. All question types here means supported/unsupported factoids; list, yes/no and summary questions are outside this experiment.

Change dev_ratio, seed, filter_mode (normalized, literal, none), and alias_mode (per_alias, first, all). Normalized matching changes punctuation/case/Greek and letter-digit forms; literal matching preserves case and punctuation. Neither verifies semantic support. The all-alias output requires a multi-answer prompt rather than the single-answer instruction.

The output train_prepared.json is for SFT; train_questions.json has one record per question for candidate generation or further SFT/DPO splitting; dev_prepared.json retains all accepted aliases for scoring. The full_resources preset includes questions without snippet matches. reserve_dpo_questions creates a separate SFT/DPO reservation, not a replacement dev split.

For prepared data, split question records before expanding aliases. occurrence_evidence takes prepared source and exports evidence-plus-answer/control records without API calls. judge_evidence uses semantic annotation; export_judged_evidence requires a complete judgment cache as annotation_dir. To continue annotation into a fresh run, supply the prior cache as annotation_dir. Only historical flags actually present in the supplied source are carried forward.

Dev evidence references have a separate pipeline: dev_occurrence, dev_judge, then dev_export, using prepared dev source and question-level train_source to verify separation. Occurrence-only references are diagnostics; the evidence SFT trainer requires the completed semantic v2 export from dev_export. Annotation batches can continue using the prior annotation_dir cache.""",
    "generation": """Set eval_input to a list of question-level prepared/raw files and model_ref to a list of model IDs, adapter directories or registry aliases. Generate training preferences only from training or held-out DPO questions.

sample20, sample10 and sample5 differ only in sample count. literal_copy changes the answer prompt. Keep the source, prompt and seed the same when comparing models. Each model writes its own candidate bank.

merge_models takes banks mapping labels to JSONL paths and prefixes response IDs with provenance. filter takes input_jsonl and sample/parser filters. score_logprobs accepts an optional reference_model_ref. No model is loaded during preview.""",
    "pairs": """Use standard_factoid or all_factoid_negatives for factoids; whole_response_list and gold_response_list are specifically for list questions.

judge_sample20/judge_sample10 require bank and a question-level prepared questions file in matching question order. Change expected_samples for other bank sizes. source_model overrides the label; otherwise candidate provenance is used. max_new_judge_calls limits new judgments. previous_judgments can point to an earlier judgments directory to reuse its response cache in a fresh run.

three_stage_curriculum takes pair_file and routes C3>C1, C3>C2 and C2>C1 into concept, format and ranking stages. Unexpected directions fail. stage2_ties takes a complete staged input_root. stage2_alignment scores filtering potential from a source_staged_root and Stage 1 run_root.""",
    "sft": """answer_only takes train_input and eval_input lists, pointing to notebook 01's train_prepared.json and dev_prepared.json. model_name can be Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-3B-Instruct or a local model directory. Other supported models require their matching chat template.

The current split-first data lives in data/BioASQ_factoid_sft_prepared/split_first_train90_dev10_supported_train_seed3407/: 1,137 supported training questions (1,363 rows) and 160 unfiltered dev questions. The earlier 126-dev experiment remains separate. Previously trained adapters used different split membership, so train fresh adapters for the new protocol; their old dev scores are not directly comparable. Recreate plans after changing data paths. Any retained outputs below may describe earlier runs.

Defaults: 4-bit LoRA, rank/alpha 32, dropout 0.05, batch size 1, accumulation 32, up to 6 epochs and generated-dev MRR checkpoint selection. These are inherited experiment settings, not newly tuned values. Model loading may download weights unless local_files_only=True. Long contexts can be truncated: source-level eligibility is not a tokenizer-level guarantee that matching evidence survives.

evidence_plus_answer/evidence_answer_control use notebook 01 exports. Supply variant (occurrence or judge), export_dir, local base_model, initial_adapter (or from_base=True), dev_input and a separately prepared dev_evidence_export in the trainer's v2 schema. A training export is not a dev evidence reference.

rationale_from_base/rationale_continued take accepted rationale_bank, question-level train_source, dev_source, local base_model and initial_adapter. They retain evidence/token-budget checks but no longer require exactly 1,111 questions. replay_fraction controls answer-only replay.""",
    "dpo": """standard takes preference_input and model_name (normally an SFT adapter). It splits by question ID and exposes the trainer's loss_type, beta and declared CLI options.

Staged presets require base_model, initial_adapter, staged_root and dev_source_input. Set model_preset to qwen25_05b or qwen25_3b. The staged directory must contain all three stage files, even if a selected stage is empty. stage1/two_stage/three_stage set the stopping point; stage_settings overrides per-stage hyperparameters.

**Staged presets default to smoke_test=True.** After checking a smoke run, set False for full training. stage2_retention needs a completed Stage 1 adapter plus tie-aware data; it skips concept learning but may use its pairs as retention anchors. cal_dpo, apo_zero and adaptive_nll select other objectives.

softmax/representative require question_classes_with_reference_jsonl, not ordinary pairs. error_aware takes accepted C1/gold rationales and Stage 1 pairs; it prepares data unless its separate train=True parameter is set. Every run is fresh; checkpoints are not silently resumed.""",
    "evaluation": """greedy, sample10_frequency, sample5_union and direct_top5 share the official evaluation CLI. model_ref and eval_input are lists. Use dev data for comparisons and official test batches only after selection. Multiple model refs compare SFT/DPO/base models under identical settings.

compare_sampling creates one shared independent bank for first-five and frequency ranking, optionally adding greedy/direct-top-five runs. Its source is one prepared/raw file and model_ref one string. history_conditioned takes local base_model, adapter, prepared source and a frozen raw_generations.jsonl bank in the existing comparison schema.

openai_direct accepts raw/prepared sources and an explicit question_limit (default 10); it makes no retry calls. gpt_direct_vs_reasoning accepts a prepared source and paired prompt versions. Its question limit bounds questions; validation/provider retries may make additional calls. Both remain API-disabled until enabled. Official Java scoring is required.""",
    "analysis": """test_evidence_coverage counts normalized/literal gold occurrences in raw BioASQ sources without GPU or API calls.

compare_banks accepts a banks mapping of labels to JSONL paths. It aligns question IDs and reports sample counts, unique answers and disagreements across any split.

dpo_diagnostics takes eval_root, optional candidate_bank, raw question_input and optional pair_jsonl list (label=path entries). It writes shared JSON/Markdown metrics, cardinality and preference diagnostics.

stage1_training_audit takes a completed Stage 1 directory as run_root, prepared training source and local base_model. It uses the selected manifest checkpoint and excludes official dev/test questions. Default limit: 10.

Historical custom plots and exploratory tables remain available in the archived notebooks. The standardized reports are not claimed to reproduce every historical plot exactly.""",
    "synthetic": """Select phases explicitly: prepare, generate, verify, finalize. Supply question-level prepared source, held-out dev_source and test_dir. Source selection excludes protected PubMed IDs.

After preparation, set prepared_run to the previous run's synthetic directory. The runner copies its manifest/cache into a fresh directory for subsequent phases, preserving earlier results. Keep source paths, seed, model settings and preparation sizes fixed across phases.

generator_model and verifier_model are independent. Compare generators with per-variant overrides while keeping source selection identical. max_new_calls is a per-phase budget. Generate/verify require RUN and ALLOW_API; prepare/finalize make no calls.

This uses the current v2 pipeline. The older v1 configuration remains archived for historical reproduction.""",
}


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(category, title, preset):
    variants = "\n".join(f"- **{name}**: {method}" for name, (method, _) in PRESETS[category].items())
    extra = ""
    if category == "sft":
        extra = '''
# Example: run both sizes on the shared train_input/eval_input in OVERRIDES:
# VARIANTS = [
#     {"label": "qwen05b", "overrides": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}},
#     {"label": "qwen3b", "overrides": {"model_name": "Qwen/Qwen2.5-3B-Instruct"}},
# ]
'''
    cells = [
        cell("markdown", f"# {title}\n\n{GUIDES[category]}\n\n"
             "Execution is **off by default**. Preview needs only Python and the repository. "
             "Select a kernel with the project's runtime dependencies before executing. "
             "Each method uses that kernel's Python in a separate worker process.\n"),
        cell("code", '''from pathlib import Path
import json
import sys

PROJECT_ROOT = next(
    p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (p / "src/notebook_workflows").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.notebook_workflows import describe, plan, execute
from src.notebook_workflows.presets import PRESETS, configuration
print("Project:", PROJECT_ROOT)
print("Python:", sys.executable)
'''),
        cell("markdown", "## Choose a preset\n\n" + variants +
             "\n\nEdit the configuration, then rerun the remaining cells. OVERRIDES uses "
             "the option names printed by describe(), with underscores. For CLI flags, "
             "True includes the named flag and False omits it. Missing inputs block execution.\n"),
        cell("code", f'''CATEGORY = {category!r}
PRESET = {preset!r}
OVERRIDES = {{
    # Set input/model paths here using the fields printed below.
}}
RUN = False
ALLOW_API = False

# Optional sequential comparisons: shared OVERRIDES plus per-variant changes.
VARIANTS = [{{"label": "default", "overrides": {{}}}}]
{extra}
METHOD, PARAMETERS = configuration(CATEGORY, PRESET, OVERRIDES)
print(json.dumps({{"method": METHOD, "parameters": PARAMETERS}}, indent=2))
'''),
        cell("markdown", "## Inspect and preview\n\n"
             "Options are read from source without importing the training stack. Unknown options "
             "and invalid choices are rejected. Outputs go under Artifacts/notebook_runs and "
             "cannot overwrite earlier runs. Rerun this cell to allocate fresh plans.\n"),
        cell("code", '''print(json.dumps(describe(METHOD), indent=2, default=str))
PLANS = []
for variant in VARIANTS:
    method, parameters = configuration(CATEGORY, PRESET, {**OVERRIDES, **variant["overrides"]})
    preview = plan(method, parameters, project_root=PROJECT_ROOT)
    PLANS.append(preview)
    print("\\nVariant:", variant["label"])
    print("Method:", preview["method"])
    print("Output:", preview["run_dir"])
    print("Requires GPU:", preview["gpu"], "| Calls API:", preview["api"])
    print("Unresolved inputs:", preview["missing"] or "none")
    print(json.dumps(preview["parameters"], indent=2))
'''),
        cell("markdown", "## Execute reviewed plans\n\n"
             "Set RUN=True, then rerun configuration and preview. API methods also need "
             "ALLOW_API=True. Plans cannot be changed after preview. Each run retains its "
             "configuration, input hashes, status and log. A repeated execution of the same "
             "plan refuses to overwrite its directory; create a fresh plan instead.\n"),
        cell("code", '''RESULTS = []
for preview in PLANS:
    if RUN:
        result = execute(preview, run=True, allow_api=ALLOW_API)
        RESULTS.append(str(result))
    else:
        print("Preview only:", preview["method"], "- set RUN=True to execute.")
'''),
        cell("markdown", "## Inspect outputs\n\n"
             "Use these explicit paths in the next notebook. Historical commands, plots and "
             "outputs are preserved in reproducibility/notebook_archive.zip; "
             "notebooks/MIGRATION.md maps old notebooks to these workflows.\n"),
        cell("code", '''for run in RESULTS:
    path = Path(run)
    print("\\nRun:", path)
    print((path / "status.json").read_text(encoding="utf-8"))
    print("Log:", path / "run.log")
    print("Outputs:", [p.name for p in path.iterdir()
                      if p.name not in {"run.log", "plan.json", "input_hashes.json"}])
'''),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }, "nbformat": 4, "nbformat_minor": 4}


def main():
    for name, category, title, preset in WORKFLOWS:
        path = ROOT / "notebooks" / f"{name}.ipynb"
        path.write_text(json.dumps(build(category, title, preset), indent=1) + "\n", encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
