"""Internal worker used by the preview-first notebook runner."""
from __future__ import annotations
import json
import runpy
import shutil
import sys
from pathlib import Path

from .catalog import METHODS
from .operations import write_json


def main():
    preview = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    spec = METHODS[preview["method"]]
    out = Path(preview["run_dir"])
    root = Path(preview["project_root"])
    p = preview["parameters"]
    if preview["method"] == "synthetic_qa" and p.get("prepared_run"):
        # Continue into a new run directory, keeping the earlier manifest/cache intact.
        shutil.copytree(p["prepared_run"], preview["outputs"]["output_root"])
    if preview["method"] == "judge_candidates" and p.get("previous_judgments"):
        # Only reuse response cache, never completed rows from a different bank.
        source = Path(p["previous_judgments"]) / "cache"
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, Path(preview["outputs"]["output_root"]) / "cache")
    if spec.module:
        sys.argv = [spec.module, *preview["command"][3:]]
        runpy.run_module(spec.module, run_name="__main__")
    else:
        from . import operations
        result = getattr(operations, spec.operation)(p, out, root)
        if result is not None:
            write_json(out / "summary.json", result)
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
