#!/usr/bin/env python3
"""Validate the portable 8B SFT bundle without loading the model."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "full_resources_per_alias": (1804, 160),
    "evidence_per_supported_alias": (1352, 160),
}


def count_rows(path: Path) -> int:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return len(rows)


def main() -> None:
    assert (ROOT / "src/utility/answer_gen_ft.py").is_file()
    assert (ROOT / "prompts/factoid_single_answer_aligned.json").is_file()
    assert (ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar").is_file()
    for variant, (expected_train, expected_eval) in EXPECTED.items():
        config = json.loads((ROOT / "configs" / f"{variant}.json").read_text(encoding="utf-8"))
        train_count = count_rows(ROOT / config["train_input"])
        eval_count = count_rows(ROOT / config["eval_input"])
        assert train_count == expected_train, (variant, train_count, expected_train)
        assert eval_count == expected_eval, (variant, eval_count, expected_eval)
        subset = config.get("generated_metric_subset_ids")
        if subset:
            subset_payload = json.loads((ROOT / subset).read_text(encoding="utf-8"))
            subset_ids = (
                subset_payload.get("question_ids", [])
                if isinstance(subset_payload, dict)
                else subset_payload
            )
            assert len(subset_ids) == 129, len(subset_ids)
        print(f"{variant}: train={train_count}, dev={eval_count}")
    print("Bundle data and required SFT files are valid.")


if __name__ == "__main__":
    main()
