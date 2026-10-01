#!/usr/bin/env python3
"""Score a retrieved Gadi expansion run with the official BioASQ matcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_workflows.local_expansion import analyze_local_expansion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--jar",
        type=Path,
        default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",
    )
    args = parser.parse_args()
    report, summary, _ = analyze_local_expansion(args.run_dir, jar_path=args.jar)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Report:", report.resolve())


if __name__ == "__main__":
    main()
