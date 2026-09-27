from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REGISTRY_VERSION = 1


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def slugify(value: str, fallback: str = "run") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return normalized or fallback


def short_model_name(model_name: str) -> str:
    return slugify(str(model_name).split("/")[-1], fallback="model")


def build_run_id(
    task: str,
    model_name: str,
    question_types: Iterable[str],
    run_name: Optional[str] = None,
) -> str:
    type_part = "-".join(slugify(item, fallback="type") for item in question_types)
    base_label = run_name or f"{task}-{short_model_name(model_name)}-{type_part}"
    return f"{timestamp_slug()}-{slugify(base_label)}"


def resolve_repo_path(path_value: Optional[str | Path], project_root: Optional[Path] = None) -> Optional[Path]:
    if path_value in {None, ""}:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = project_root or get_project_root()
    return (base / path).resolve()


def to_project_relative(path_value: Optional[str | Path], project_root: Optional[Path] = None) -> Optional[str]:
    if path_value in {None, ""}:
        return None
    path = Path(path_value).resolve()
    base = (project_root or get_project_root()).resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def empty_registry() -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "version": REGISTRY_VERSION,
        "updated_at": now,
        "aliases": {},
        "runs": {},
    }


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return empty_registry()

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a JSON object: {path}")

    data.setdefault("version", REGISTRY_VERSION)
    data.setdefault("updated_at", utc_now_iso())
    data.setdefault("aliases", {})
    data.setdefault("runs", {})
    return data


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = utc_now_iso()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)


def upsert_run(registry_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(manifest["run_id"])
    registry = load_registry(registry_path)
    registry["runs"][run_id] = manifest
    save_registry(registry_path, registry)
    return registry


def get_run(registry_path: Path, run_id: str) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    run = registry.get("runs", {}).get(run_id)
    if run is None:
        raise KeyError(f"Unknown run_id: {run_id}")
    return run


def list_runs(
    registry_path: Path,
    task: Optional[str] = None,
    status: Optional[str] = None,
) -> list[Dict[str, Any]]:
    registry = load_registry(registry_path)
    runs = list(registry.get("runs", {}).values())
    if task:
        runs = [run for run in runs if run.get("task") == task]
    if status:
        runs = [run for run in runs if run.get("status") == status]
    runs.sort(key=lambda run: (run.get("created_at") or "", run.get("run_id") or ""), reverse=True)
    return runs


def normalize_alias(alias: str) -> str:
    return slugify(alias, fallback="alias")


def promote_alias(registry_path: Path, alias: str, run_id: str) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    run = registry.get("runs", {}).get(run_id)
    if run is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    alias_name = normalize_alias(alias)
    registry["aliases"][alias_name] = {
        "alias": alias_name,
        "run_id": run_id,
        "task": run.get("task"),
        "status": run.get("status"),
        "base_model": run.get("base_model"),
        "adapter_dir": run.get("paths", {}).get("adapter_dir"),
        "manifest_path": run.get("paths", {}).get("manifest_path"),
        "updated_at": utc_now_iso(),
    }
    save_registry(registry_path, registry)
    return registry["aliases"][alias_name]


def resolve_alias(registry_path: Path, alias: str) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    alias_name = normalize_alias(alias)
    alias_data = registry.get("aliases", {}).get(alias_name)
    if alias_data is None:
        raise KeyError(f"Unknown alias: {alias_name}")
    return alias_data
