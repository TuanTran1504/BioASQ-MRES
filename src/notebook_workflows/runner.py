"""Preview-first, subprocess-isolated workflow execution without GPU imports."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .catalog import METHODS, TEST

ROOT = Path(__file__).resolve().parents[2]
BLOCKED_OPTIONS = {"overwrite", "overwrite_merged", "resume_from_checkpoint", "resume_from",
                   "all_registry_runs", "archive_incomplete_chunks"}
SPECIAL_OPTIONS = {"synthetic_qa": {"prepared_run"}, "judge_candidates": {"previous_judgments"}}


class WorkflowExecutionError(subprocess.CalledProcessError):
    """Keep the worker's diagnostic visible in truncated notebook tracebacks."""

    def __init__(self, returncode, command, *, log_path, output):
        super().__init__(returncode, command, output=output)
        self.log_path = log_path

    def __str__(self):
        return (
            f"Workflow worker exited with code {self.returncode}.\n"
            f"Worker output (last lines):\n{self.output.rstrip()}\n"
            f"Full log: {self.log_path}"
        )


def _openai_model_refs(params):
    return [
        str(value)
        for value in (params.get("model_ref") or [])
        if str(value).casefold().startswith("openai:")
    ]


def _local_model_refs(params):
    return [
        str(value)
        for value in (params.get("model_ref") or [])
        if not str(value).casefold().startswith("openai:")
    ]


def available_methods(category=None):
    return {name: spec.description for name, spec in METHODS.items()
            if category is None or spec.category == category}


def defaults(method):
    return copy.deepcopy(METHODS[method].parameters)


def _literal(node, fallback=None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return fallback


def cli_schema(module):
    """Read argparse declarations without importing torch, Unsloth, or credentials."""
    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [_literal(arg) for arg in node.args]
        flags = [flag for flag in flags if isinstance(flag, str) and flag.startswith("--")]
        if not flags:
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        info = {key: _literal(value) for key, value in kwargs.items()}
        if isinstance(kwargs.get("type"), ast.Name):
            info["type"] = kwargs["type"].id
        for flag in flags:
            result[flag[2:].replace("-", "_")] = {**info, "flag": flag}
    return result


def stage_schema():
    tree = ast.parse((ROOT / "cse_dpo/train_factoid_three_stage_dpo.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def describe(method):
    spec = METHODS[method]
    options = cli_schema(spec.schema_module or spec.module) if spec.module else {k: {} for k in spec.parameters}
    if method == "staged_dpo":
        options = {key: {} for key in stage_schema()}
    return {
        "method": method, "description": spec.description,
        "defaults": defaults(method), "required": list(spec.required),
        "options": {k: v for k, v in options.items() if k not in spec.outputs and k not in BLOCKED_OPTIONS
                    and k not in {"output_root", "stage_output_suffixes"}},
        "outputs": spec.outputs or {"artifacts": "operation-specific files below the run directory"},
        "uses_gpu": spec.gpu, "may_call_api": spec.api,
    }


def _validate_cli_value(key, value, info):
    if value is None:
        return
    action = info.get("action")
    if action in {"store_true", "store_false"}:
        if type(value) is not bool:
            raise ValueError(f"{key} must be a boolean (True includes the named flag).")
        return
    multiple = info.get("nargs") in {"+", "*"} or action == "append"
    if multiple and not isinstance(value, list):
        raise ValueError(f"{key} must be a list.")
    if not multiple and isinstance(value, (list, dict)):
        raise ValueError(f"{key} must be a scalar.")
    values = value if multiple else [value]
    for item in values:
        kind = info.get("type")
        if kind == "int" and type(item) is not int:
            raise ValueError(f"{key} must contain integers.")
        if kind == "float" and (type(item) not in {int, float}):
            raise ValueError(f"{key} must contain numbers.")
        choices = info.get("choices")
        if isinstance(choices, (tuple, list)) and item not in choices:
            raise ValueError(f"{key}: {item!r} is not one of {choices}.")
    if info.get("nargs") == "+" and not values:
        raise ValueError(f"{key} cannot be empty.")


def _argv(module, parameters, schema):
    argv = [sys.executable, "-m", module]
    for key, value in parameters.items():
        if value is None:
            continue
        info = schema[key]
        flag = info["flag"]
        if info.get("action") in {"store_true", "store_false"}:
            if value:
                argv.append(flag)
        elif info.get("action") == "append":
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.append(flag)
            argv.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return argv


def _as_path(value, root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def question_ids(path):
    """Recover original question IDs even after per-alias SFT expansion."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("questions", []) if isinstance(payload, dict) else payload
    ids = set()
    for row in rows:
        qid = row.get("source_question_id") or row.get("original_id") or row.get("question_id") or row.get("id")
        if qid:
            ids.add(re.sub(r"__(?:supported_alias|alias)_.*$", "", str(qid)))
    return ids


def is_alias_expanded(path):
    if Path(path).suffix != ".json":
        return False
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return False
    return any(
        row.get("source_question_id", row.get("original_id", row.get("id"))) != row.get("id")
        or re.search(r"__(?:supported_alias|alias)_", str(row.get("id", "")))
        for row in payload
    )


def _validate_semantics(method, params, schema):
    budget = {"judge_candidates": "max_new_judge_calls", "evidence_annotation": "max_new_calls",
              "synthetic_qa": "max_new_calls", "openai_direct": "question_limit",
              "gpt_reasoning": "question_limit", "dev_evidence": "max_new_calls"}.get(method)
    if budget and (type(params.get(budget)) is not int or params[budget] <= 0):
        raise ValueError(f"{budget} must be an explicit positive limit.")
    if method == "local_evaluation":
        if (params.get("openai_model") or _openai_model_refs(params)) and (
            type(params.get("max_new_api_calls")) is not int or params["max_new_api_calls"] <= 0
        ):
            raise ValueError("OpenAI candidates require max_new_api_calls to be a positive integer.")
        if params.get("semantic_judge") and (
            type(params.get("semantic_judge_max_new_calls")) is not int
            or params["semantic_judge_max_new_calls"] <= 0
        ):
            raise ValueError(
                "Grounded semantic evaluation requires semantic_judge_max_new_calls "
                "to be a positive integer."
            )
    if not METHODS[method].module:
        for key, default in METHODS[method].parameters.items():
            if type(default) is bool and type(params.get(key)) is not bool:
                raise ValueError(f"{key} must be a boolean.")
    for key in ("dev_ratio", "validation_ratio", "eval_fraction", "sft_fraction"):
        if params.get(key) is not None and not 0 < params[key] < 1:
            raise ValueError(f"{key} must be between zero and one.")
    for key in ("max_new_calls", "max_new_judge_calls", "question_limit", "samples", "samples_per_question_total"):
        if params.get(key) is not None and (type(params[key]) is not int or params[key] <= 0):
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("seed", "max_seq_length", "max_length"):
        if params.get(key) is not None and (type(params[key]) is not int or params[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer.")
    if method == "prepared_split" and params.get("group_by") != "id":
        raise ValueError("The consolidated splitter requires question-level splitting.")
    if method == "standard_dpo" and params.get("split_by") != "question":
        raise ValueError("The consolidated DPO workflow requires question-level validation.")
    if method == "rationale_sft" and not 0 <= params["replay_fraction"] <= .5:
        raise ValueError("replay_fraction must be between zero and 0.5.")
    if method == "staged_dpo":
        if params.get("objective") not in {"dpo", "dpo_d", "apo_zero", "apo_down", "dpo_adaptive_nll", "cal_dpo"}:
            raise ValueError("Unknown staged DPO objective.")
        names = {"concept_learning", "format_alignment", "hierarchical_ranking"}
        if params.get("stop_after_stage") not in names | {None}:
            raise ValueError("Unknown stop_after_stage.")
        if not set(params.get("skip_stages", [])) <= names:
            raise ValueError("Unknown skipped stage.")
        settings = params.get("stage_settings", {})
        if not isinstance(settings, dict) or not set(settings) <= names:
            raise ValueError("Unknown stage_settings stage.")
        for stage, values in settings.items():
            if not isinstance(values, dict) or not set(values) <= stage_schema():
                raise ValueError(f"Unknown configuration fields for {stage}.")
            forbidden = {"output_root", "base_model", "initial_adapter", "staged_root", "dev_source_input",
                         "stage_settings", "stage_output_suffixes", "semantic_judge_enabled", "semantic_judge_endpoint"}
            if set(values) & forbidden:
                raise ValueError("Stage overrides may change training settings, not paths or API access.")
    choices = {
        "match_mode": {"normalized", "literal"}, "strategies": {"greedy", "first5", "frequency", "direct_top5"},
    }
    for key, allowed in choices.items():
        if key in params:
            values = params[key] if isinstance(params[key], list) else [params[key]]
            if not values or not set(values) <= allowed:
                raise ValueError(f"{key} must select from {sorted(allowed)}.")


def plan(method, parameters=None, *, run_name=None, project_root=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    spec = METHODS[method]
    root = Path(project_root or ROOT).resolve()
    parameters = copy.deepcopy(parameters or {})
    if any("api_key" in key and not key.endswith("_file") for key in parameters):
        raise ValueError("Pass an API key file path, never a secret in notebook configuration.")
    params = {**defaults(method), **parameters}
    schema = cli_schema(spec.schema_module or spec.module) if spec.module else {key: {} for key in spec.parameters}
    if method == "staged_dpo":
        schema = {key: {} for key in stage_schema()}
    specials = SPECIAL_OPTIONS.get(method, set())
    unknown = set(params) - set(schema) - specials
    protected = set(parameters) & (set(spec.outputs) | BLOCKED_OPTIONS | {"output_root", "stage_output_suffixes"})
    if unknown or protected:
        raise ValueError(f"Unknown or protected options: {sorted(unknown | protected)}")
    if spec.module:
        for key, value in params.items():
            if key not in specials:
                _validate_cli_value(key, value, schema[key])
    _validate_semantics(method, params, schema)
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_name):
        raise ValueError("run_name must be a plain name, not a path.")
    run_dir = root / "Artifacts/notebook_runs" / spec.category / method / run_name
    missing = []
    inputs = {}
    required = set(spec.required) | {key for key, info in schema.items() if info.get("required") is True}
    required -= set(spec.outputs)
    if method == "evidence_sft" and not params.get("from_base"):
        required.add("initial_adapter")
    if method in {"evidence_annotation", "dev_evidence"} and params["phase"] == "export":
        required.add("annotation_dir")
    for key in required:
        if params.get(key) in (None, "", [], {}):
            missing.append(f"Set {key}")
    if method == "local_evaluation" and not (
        params.get("model_ref") or params.get("all_registry_runs") or params.get("openai_model")
    ):
        missing.append("Set model_ref, openai_model, or all_registry_runs")
    for key in spec.inputs:
        value = params.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        resolved = [_as_path(v, root) for v in values]
        inputs[key] = [str(path) for path in resolved]
        params[key] = [str(path) for path in resolved] if isinstance(value, list) else str(resolved[0])
        for path in resolved:
            if not path.exists():
                missing.append(f"Missing {key}: {path}")
    if "banks" in params:
        if not isinstance(params["banks"], dict):
            raise ValueError("banks must map labels to JSONL paths.")
        params["banks"] = {name: str(_as_path(path, root)) for name, path in params["banks"].items()}
        inputs["banks"] = list(params["banks"].values())
        missing.extend(f"Missing bank: {p}" for p in inputs["banks"] if not Path(p).is_file())
    for key in ("prepared_run", "previous_judgments"):
        if params.get(key):
            source = _as_path(params[key], root)
            params[key] = str(source)
            inputs[key] = [str(source)]
            if not source.is_dir():
                missing.append(f"Missing {key}: {source}")
    if method == "synthetic_qa" and params["phase"] not in {"prepare", "all"} and not params.get("prepared_run"):
        missing.append("Set prepared_run to the previous synthetic output directory")
    outputs = {key: str(run_dir / relative) for key, relative in spec.outputs.items()}
    needs_api = spec.api
    if method == "synthetic_qa":
        needs_api = params["phase"] in {"generate", "verify", "verify_finalize", "all"}
    if method == "evidence_annotation":
        if params["phase"] not in {"annotate", "export"}:
            raise ValueError("phase must be annotate or export")
        needs_api = params["phase"] == "annotate"
    if method == "dev_evidence":
        if params["phase"] not in {"occurrence", "annotate", "export"}:
            raise ValueError("Unknown dev evidence phase.")
        needs_api = params["phase"] == "annotate"
    if method == "staged_dpo":
        needs_api = bool(params.get("semantic_judge_enabled"))
        if needs_api and not params.get("semantic_judge_max_new_calls"):
            raise ValueError("API-assisted checkpoint selection requires semantic_judge_max_new_calls.")
    if method == "local_evaluation":
        needs_api = bool(
            params.get("openai_model") or _openai_model_refs(params) or params.get("semantic_judge")
        )
    if "train_input" in inputs and "eval_input" in inputs and not missing:
        train_ids = set().union(*(question_ids(p) for p in inputs["train_input"]))
        dev_ids = set().union(*(question_ids(p) for p in inputs["eval_input"]))
        if train_ids & dev_ids:
            raise ValueError("Training and dev inputs contain the same original question IDs.")
        tests = [root / path for path in TEST]
        test_ids = set().union(*(question_ids(p) for p in tests if p.is_file()))
        if (train_ids | dev_ids) & test_ids:
            raise ValueError("Training/dev inputs overlap official test question IDs.")
    if method == "prepared_split" and not missing:
        if is_alias_expanded(params["input"]):
            raise ValueError("This file is alias-expanded. Split question records first, then expand aliases.")
    if method == "sample_candidates" and not missing:
        if any(is_alias_expanded(path) for path in inputs["eval_input"]):
            raise ValueError("Candidate generation requires question-level records, not alias-expanded SFT rows.")
    if method == "staged_dpo" and not missing:
        stages = list(Path(params["staged_root"]).glob("dpo_stage*.jsonl"))
        training_ids = set().union(*(question_ids(path) for path in stages))
        dev_ids = question_ids(params["dev_source_input"])
        test_ids = set().union(*(question_ids(root / path) for path in TEST if (root / path).is_file()))
        if training_ids & (dev_ids | test_ids):
            raise ValueError("Staged preference questions overlap dev or official test questions.")
    if method == "rationale_sft" and not missing:
        if question_ids(params["train_source"]) & question_ids(params["dev_source"]):
            raise ValueError("Rationale SFT train/dev question IDs overlap.")
    if method == "dev_evidence" and not missing:
        if question_ids(params["source"]) & question_ids(params["train_source"]):
            raise ValueError("Dev evidence source overlaps training questions.")
    command = None
    if spec.module:
        values = {key: value for key, value in params.items() if key not in specials}
        values.update(outputs)
        command = _argv(spec.module, values, schema)
    uses_gpu = spec.gpu
    if method == "local_evaluation":
        uses_gpu = bool(_local_model_refs(params) or params.get("all_registry_runs"))
    result = {"version": 1, "method": method, "category": spec.category, "project_root": str(root),
              "run_dir": str(run_dir), "parameters": params, "outputs": outputs,
              "inputs": inputs, "command": command, "missing": missing,
              "gpu": uses_gpu, "api": needs_api, "description": spec.description}
    result["input_sha256"] = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for paths in inputs.values() for path in paths if Path(path).is_file()
    }
    result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def execute(preview, *, run=False, allow_api=False):
    """Preview is side-effect free. Execution never reuses or overwrites a run."""
    if not run:
        print(json.dumps(preview, indent=2))
        return None
    unsigned = {key: value for key, value in preview.items() if key != "fingerprint"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != preview["fingerprint"]:
        raise ValueError("Plan changed after preview; build a fresh plan.")
    if preview["api"] and not allow_api:
        raise ValueError("This method calls an API. Set ALLOW_API=True explicitly.")
    if preview["missing"]:
        raise ValueError("\n".join(preview["missing"]))
    for paths in preview["inputs"].values():
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(path)
    for path, expected_hash in preview["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Input changed after preview: {path}; create a fresh plan.")
    run_dir = Path(preview["run_dir"])
    root = Path(preview["project_root"])
    expected = (root / "Artifacts/notebook_runs").resolve()
    if not run_dir.resolve().is_relative_to(expected):
        raise ValueError("Run output escaped Artifacts/notebook_runs.")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_text(json.dumps(preview, indent=2) + "\n", encoding="utf-8")
    sources = {}
    for paths in preview["inputs"].values():
        for path in paths:
            if Path(path).is_file():
                sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (run_dir / "input_hashes.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "src.notebook_workflows.worker", str(run_dir / "plan.json")]
    status = {"status": "running", "method": preview["method"], "fingerprint": preview["fingerprint"]}
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    log_path = run_dir / "run.log"
    output_tail = deque(maxlen=20)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    output_tail.append(line[-2000:])
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
                if returncode:
                    raise WorkflowExecutionError(
                        returncode, command, log_path=log_path, output="".join(output_tail))
            except BaseException:
                process.terminate()
                process.wait()
                raise
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=type(error).__name__)
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print("Saved run:", run_dir)
    return run_dir
