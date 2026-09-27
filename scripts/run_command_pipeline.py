#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-") or "pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequence of train/eval shell commands from one JSON config file, "
            "with step-by-step logging and optional dry-run support."
        )
    )
    parser.add_argument(
        "config",
        help="Path to the pipeline JSON config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps even if one step fails.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Start at this step name or 1-based step index.",
    )
    parser.add_argument(
        "--only-step",
        default=None,
        help="Run only this step name or 1-based step index.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional directory for pipeline logs and status files. "
            "Defaults to Artifacts/pipeline_runs/<timestamp>-<run_name>."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return payload


def expand_string(value: str, variables: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Unknown variable '{key}'")
        return variables[key]

    return VARIABLE_PATTERN.sub(replace, value)


def expand_value(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return expand_string(value, variables)
    if isinstance(value, list):
        return [expand_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {str(key): expand_value(item, variables) for key, item in value.items()}
    return value


def resolve_path(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_config_path(path_value: str, *, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (project_root / path).resolve()
    return project_candidate


def normalize_command(command: Any) -> tuple[list[str] | None, str | None]:
    if isinstance(command, str):
        cleaned = command.strip()
        if not cleaned:
            raise ValueError("Step command string is empty.")
        return None, cleaned
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        if not command:
            raise ValueError("Step command list is empty.")
        return list(command), None
    raise ValueError("Each step command must be either a string or a list of strings.")


def command_display(argv: Sequence[str] | None, shell_command: str | None) -> str:
    if argv is not None:
        return shlex.join(argv)
    assert shell_command is not None
    return shell_command


def resolve_selector(selector: str, step_names: Sequence[str]) -> int:
    if selector.isdigit():
        index = int(selector)
        if index < 1 or index > len(step_names):
            raise ValueError(
                f"Step index {index} is out of range. Valid range is 1..{len(step_names)}."
            )
        return index - 1

    try:
        return step_names.index(selector)
    except ValueError as error:
        raise ValueError(
            f"Unknown step '{selector}'. Expected a 1-based index or one of: {step_names}"
        ) from error


def build_variables(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    config_path: Path,
    run_dir: Path,
) -> dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    variables: dict[str, str] = {
        "project_root": str(project_root),
        "artifacts_root": str(project_root / "Artifacts"),
        "config_path": str(config_path),
        "config_dir": str(config_path.parent.resolve()),
        "pipeline_run_dir": str(run_dir),
        "timestamp": timestamp,
        "date": date,
    }

    raw_variables = config.get("variables", {})
    if raw_variables is None:
        return variables
    if not isinstance(raw_variables, dict):
        raise ValueError("Config field 'variables' must be an object if provided.")

    for key, raw_value in raw_variables.items():
        if not isinstance(key, str):
            raise ValueError("Config variable names must be strings.")
        expanded = expand_value(raw_value, variables)
        if isinstance(expanded, (dict, list)):
            raise ValueError(
                f"Variable '{key}' resolved to a non-string value. Variables must resolve to strings."
            )
        variables[key] = str(expanded)
    return variables


def write_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.resolve()
    config_path = resolve_config_path(args.config, project_root=project_root)
    config = load_json(config_path)

    run_name = str(config.get("run_name") or config_path.stem)
    default_run_dir = project_root / "Artifacts" / "pipeline_runs" / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(run_name)}"
    )
    run_dir = (
        resolve_path(args.run_dir, base_dir=project_root)
        if args.run_dir
        else default_run_dir
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    variables = build_variables(
        config=config,
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
    )

    global_cwd = str(expand_value(config.get("cwd", "${project_root}"), variables))
    global_env = expand_value(config.get("env", {}), variables)
    if not isinstance(global_env, dict):
        raise ValueError("Config field 'env' must be an object if provided.")

    raw_steps = config.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Config field 'steps' must be a non-empty array.")

    resolved_steps: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"Step {step_index} must be an object.")
        if raw_step.get("enabled", True) is False:
            continue

        name = str(raw_step.get("name") or f"step-{step_index}")
        command_value = expand_value(raw_step.get("command"), variables)
        argv, shell_command = normalize_command(command_value)

        cwd_value = str(expand_value(raw_step.get("cwd", global_cwd), variables))
        cwd = resolve_path(cwd_value, base_dir=project_root)

        env_overrides = expand_value(raw_step.get("env", {}), variables)
        if not isinstance(env_overrides, dict):
            raise ValueError(f"Step '{name}' field 'env' must be an object if provided.")

        skip_value = expand_value(raw_step.get("skip_if_exists", []), variables)
        if isinstance(skip_value, str):
            skip_items = [skip_value]
        elif isinstance(skip_value, list) and all(isinstance(item, str) for item in skip_value):
            skip_items = list(skip_value)
        else:
            raise ValueError(
                f"Step '{name}' field 'skip_if_exists' must be a string or list of strings if provided."
            )

        resolved_steps.append(
            {
                "index": len(resolved_steps) + 1,
                "name": name,
                "argv": argv,
                "shell_command": shell_command,
                "cwd": cwd,
                "env": {str(key): str(value) for key, value in env_overrides.items()},
                "skip_if_exists": [resolve_path(item, base_dir=project_root) for item in skip_items],
            }
        )

    if not resolved_steps:
        raise ValueError("No enabled steps were found in the config.")

    step_names = [str(step["name"]) for step in resolved_steps]
    if args.only_step is not None:
        selected_index = resolve_selector(args.only_step, step_names)
        selected_steps = [resolved_steps[selected_index]]
    elif args.start_at is not None:
        start_index = resolve_selector(args.start_at, step_names)
        selected_steps = resolved_steps[start_index:]
    else:
        selected_steps = resolved_steps

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_payload = {
        "config_path": str(config_path),
        "run_name": run_name,
        "cwd": str(resolve_path(global_cwd, base_dir=project_root)),
        "env": {str(key): str(value) for key, value in global_env.items()},
        "variables": variables,
        "selected_step_names": [step["name"] for step in selected_steps],
        "steps": [
            {
                "index": step["index"],
                "name": step["name"],
                "command": command_display(step["argv"], step["shell_command"]),
                "cwd": str(step["cwd"]),
                "env": step["env"],
                "skip_if_exists": [str(path) for path in step["skip_if_exists"]],
            }
            for step in resolved_steps
        ],
    }
    write_status(run_dir / "resolved_config.json", resolved_config_payload)

    status_path = run_dir / "pipeline_status.json"
    pipeline_status: dict[str, Any] = {
        "config_path": str(config_path),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "started_at": datetime.now().isoformat(),
        "selected_step_names": [step["name"] for step in selected_steps],
        "steps": [],
    }
    write_status(status_path, pipeline_status)

    overall_exit_code = 0
    base_env = os.environ.copy()
    base_env.update({str(key): str(value) for key, value in global_env.items()})

    for step in selected_steps:
        display = command_display(step["argv"], step["shell_command"])
        log_path = logs_dir / f"{int(step['index']):02d}-{slugify(str(step['name']))}.log"
        started_at = datetime.now()

        print(f"\n=== Step {step['index']}: {step['name']} ===")
        print(f"CWD: {step['cwd']}")
        print(f"Log: {log_path}")
        print(f"Command: {display}")

        skipped = False
        return_code = 0
        duration_seconds = 0.0
        if step["skip_if_exists"] and all(path.exists() for path in step["skip_if_exists"]):
            skipped = True
            print("Skipping because all skip_if_exists paths already exist.")
        elif args.dry_run:
            print("Dry run only; command not executed.")
        else:
            step_env = dict(base_env)
            step_env.update(step["env"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            start_time = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log_handle:
                try:
                    process = subprocess.Popen(
                        step["argv"] if step["argv"] is not None else ["bash", "-lc", step["shell_command"]],
                        cwd=str(step["cwd"]),
                        env=step_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log_handle.write(line)
                    return_code = process.wait()
                except KeyboardInterrupt:
                    print("\nInterrupted. Terminating running process...")
                    process.terminate()
                    raise
            duration_seconds = time.monotonic() - start_time
            if return_code != 0:
                overall_exit_code = return_code

        step_status = {
            "index": int(step["index"]),
            "name": str(step["name"]),
            "cwd": str(step["cwd"]),
            "command": display,
            "log_path": str(log_path),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "return_code": return_code,
            "skipped": skipped,
            "dry_run": bool(args.dry_run),
        }
        pipeline_status["steps"].append(step_status)
        write_status(status_path, pipeline_status)

        if return_code != 0 and not args.continue_on_error:
            print(f"Stopping after failure in step '{step['name']}'.")
            break

    pipeline_status["finished_at"] = datetime.now().isoformat()
    pipeline_status["exit_code"] = overall_exit_code
    write_status(status_path, pipeline_status)

    if args.dry_run:
        print(f"\nDry run complete. Resolved pipeline files are in {run_dir}")
    else:
        print(f"\nPipeline complete. Status file: {status_path}")
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
