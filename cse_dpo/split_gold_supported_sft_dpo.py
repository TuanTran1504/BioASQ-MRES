"""Create a deterministic question-level SFT/DPO split.

The split is made before supported aliases are expanded into separate SFT rows.
This prevents aliases (and resources) from one BioASQ question leaking across the
SFT and DPO training pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utility.bioasq_format import parse_prediction_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/BioASQ_factoid_sft_prepared"
DEFAULT_QUESTION_SOURCE = DATA_ROOT / "evidence_grounded_single_answer_qwen25_05b/train_prepared.json"
DEFAULT_ALIAS_SOURCE = DATA_ROOT / "evidence_grounded_per_supported_alias_qwen25_05b/train_prepared.json"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "gold_supported_question_split_sft80_dpo20_seed3407"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _answer_token_count(answer: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", answer, flags=re.UNICODE))


def _context_chars(row: dict[str, Any]) -> int:
    return sum(
        len(str(value))
        for key, value in row.items()
        if re.fullmatch(r"input_\d+", key) and int(key.split("_")[1]) >= 2
    )


def _size_bucket(value: int, boundaries: tuple[int, int]) -> str:
    if value <= boundaries[0]:
        return f"le_{boundaries[0]}"
    if value <= boundaries[1]:
        return f"{boundaries[0] + 1}_to_{boundaries[1]}"
    return f"ge_{boundaries[1] + 1}"


def question_features(row: dict[str, Any]) -> dict[str, Any]:
    aliases = parse_prediction_items(str(row.get("output", "")), "factoid")
    if not aliases:
        raise ValueError(f"Question {row.get('id')} has no parseable supported alias")
    alias_count = len(aliases)
    max_answer_tokens = max(_answer_token_count(alias) for alias in aliases)
    context_chars = _context_chars(row)
    alias_bucket = "1" if alias_count == 1 else "2" if alias_count == 2 else "3_plus"
    answer_bucket = _size_bucket(max_answer_tokens, (2, 5))
    context_bucket = _size_bucket(context_chars, (4_000, 10_000))
    return {
        "aliases": aliases,
        "alias_count": alias_count,
        "max_answer_tokens": max_answer_tokens,
        "context_chars": context_chars,
        "stratum": f"aliases={alias_bucket}|answer_tokens={answer_bucket}|context_chars={context_bucket}",
    }


def _stable_question_order(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def stratified_question_split(
    question_rows: list[dict[str, Any]],
    *,
    sft_fraction: float = 0.8,
    seed: int = 3407,
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    if not 0.0 < sft_fraction < 1.0:
        raise ValueError("sft_fraction must be strictly between 0 and 1")
    ids = [str(row.get("id", "")).strip() for row in question_rows]
    if any(not question_id for question_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be nonempty and unique")

    features = {str(row["id"]): question_features(row) for row in question_rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for question_id in ids:
        groups[features[question_id]["stratum"]].append(question_id)

    dpo_target = len(ids) - round(len(ids) * sft_fraction)
    allocations: dict[str, int] = {}
    fractional_remainders: dict[str, float] = {}
    for stratum, members in groups.items():
        exact = len(members) * dpo_target / len(ids)
        capacity = len(members) - 1 if len(members) > 1 else 0
        allocations[stratum] = min(math.floor(exact), capacity)
        fractional_remainders[stratum] = exact - math.floor(exact)

    remaining = dpo_target - sum(allocations.values())
    priority = sorted(groups, key=lambda key: (-fractional_remainders[key], key))
    while remaining:
        progressed = False
        for stratum in priority:
            capacity = len(groups[stratum]) - 1 if len(groups[stratum]) > 1 else 0
            if allocations[stratum] >= capacity:
                continue
            allocations[stratum] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("Could not allocate the requested DPO question count without emptying a stratum")

    dpo_ids: set[str] = set()
    for stratum, members in groups.items():
        ordered = sorted(members, key=lambda question_id: (_stable_question_order(question_id, seed), question_id))
        dpo_ids.update(ordered[: allocations[stratum]])
    sft_ids = [question_id for question_id in ids if question_id not in dpo_ids]
    ordered_dpo_ids = [question_id for question_id in ids if question_id in dpo_ids]

    if len(sft_ids) != round(len(ids) * sft_fraction) or len(ordered_dpo_ids) != dpo_target:
        raise AssertionError("Split sizes do not match their exact targets")
    if set(sft_ids) & set(ordered_dpo_ids) or set(sft_ids) | set(ordered_dpo_ids) != set(ids):
        raise AssertionError("SFT/DPO question assignments are not a disjoint exhaustive partition")
    return sft_ids, ordered_dpo_ids, features


def build_split(
    question_rows: list[dict[str, Any]],
    alias_rows: list[dict[str, Any]],
    *,
    sft_fraction: float = 0.8,
    seed: int = 3407,
) -> dict[str, Any]:
    sft_ids, dpo_ids, features = stratified_question_split(
        question_rows, sft_fraction=sft_fraction, seed=seed
    )
    sft_set, dpo_set = set(sft_ids), set(dpo_ids)
    question_by_id = {str(row["id"]): row for row in question_rows}

    alias_question_ids = [str(row.get("source_question_id", "")) for row in alias_rows]
    if any(not question_id for question_id in alias_question_ids):
        raise ValueError("Every expanded alias row must have source_question_id")
    unknown = set(alias_question_ids) - set(question_by_id)
    if unknown:
        raise ValueError(f"Expanded alias rows contain unknown question IDs: {sorted(unknown)[:5]}")

    sft_alias_rows = [row for row in alias_rows if str(row["source_question_id"]) in sft_set]
    dpo_alias_rows = [row for row in alias_rows if str(row["source_question_id"]) in dpo_set]
    alias_row_counts = Counter(alias_question_ids)
    for question_id, feature in features.items():
        if alias_row_counts[question_id] != feature["alias_count"]:
            raise ValueError(
                f"Alias expansion mismatch for {question_id}: "
                f"question output has {feature['alias_count']}, expanded source has {alias_row_counts[question_id]}"
            )

    assignments = []
    for question_id in question_by_id:
        feature = features[question_id]
        assignments.append(
            {
                "question_id": question_id,
                "split": "sft" if question_id in sft_set else "dpo",
                "question": str(question_by_id[question_id].get("input_1", "")),
                "supported_aliases": feature["aliases"],
                "alias_count": feature["alias_count"],
                "alias_row_count": alias_row_counts[question_id],
                "max_answer_tokens": feature["max_answer_tokens"],
                "context_chars": feature["context_chars"],
                "stratum": feature["stratum"],
            }
        )

    return {
        "sft_question_ids": sft_ids,
        "dpo_question_ids": dpo_ids,
        "sft_questions": [question_by_id[question_id] for question_id in sft_ids],
        "dpo_questions": [question_by_id[question_id] for question_id in dpo_ids],
        "sft_alias_rows": sft_alias_rows,
        "dpo_alias_rows": dpo_alias_rows,
        "assignments": assignments,
    }


def _counts_by_stratum(assignments: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assignments:
        counts[row["stratum"]][row["split"]] += 1
    return {
        stratum: {
            "total": sum(split_counts.values()),
            "sft": split_counts["sft"],
            "dpo": split_counts["dpo"],
        }
        for stratum, split_counts in sorted(counts.items())
    }


def write_split(
    output_dir: Path,
    split: dict[str, Any],
    *,
    question_source: Path,
    alias_source: Path,
    sft_fraction: float,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sft_question_ids": output_dir / "sft_question_ids.json",
        "dpo_question_ids": output_dir / "dpo_question_ids.json",
        "sft_questions": output_dir / "sft_questions_train_prepared.json",
        "dpo_questions": output_dir / "dpo_questions_for_candidate_generation.json",
        "sft_alias_rows": output_dir / "sft_per_supported_alias_train_prepared.json",
        "dpo_alias_rows": output_dir / "dpo_per_supported_alias_audit.json",
        "assignments_csv": output_dir / "question_split_audit.csv",
        "manifest": output_dir / "manifest.json",
    }
    _write_json(paths["sft_question_ids"], {"question_ids": split["sft_question_ids"]})
    _write_json(paths["dpo_question_ids"], {"question_ids": split["dpo_question_ids"]})
    _write_json(paths["sft_questions"], split["sft_questions"])
    _write_json(paths["dpo_questions"], split["dpo_questions"])
    _write_json(paths["sft_alias_rows"], split["sft_alias_rows"])
    _write_json(paths["dpo_alias_rows"], split["dpo_alias_rows"])

    with paths["assignments_csv"].open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "question_id",
            "split",
            "question",
            "supported_aliases",
            "alias_count",
            "alias_row_count",
            "max_answer_tokens",
            "context_chars",
            "stratum",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in split["assignments"]:
            encoded = dict(row)
            encoded["supported_aliases"] = json.dumps(row["supported_aliases"], ensure_ascii=False)
            writer.writerow(encoded)

    manifest = {
        "version": "gold-supported-sft-dpo-question-split-v1",
        "seed": seed,
        "sft_fraction": sft_fraction,
        "dpo_fraction": round(1.0 - sft_fraction, 10),
        "split_unit": "original BioASQ question_id before supported-alias expansion",
        "stratification": [
            "supported alias count: 1, 2, or 3+",
            "maximum supported-alias token count: <=2, 3-5, or >=6",
            "resource context characters: <=4000, 4001-10000, or >=10001",
        ],
        "question_count": len(split["assignments"]),
        "sft_question_count": len(split["sft_question_ids"]),
        "dpo_question_count": len(split["dpo_question_ids"]),
        "expanded_alias_row_count": len(split["sft_alias_rows"]) + len(split["dpo_alias_rows"]),
        "sft_expanded_alias_row_count": len(split["sft_alias_rows"]),
        "dpo_expanded_alias_row_count": len(split["dpo_alias_rows"]),
        "disjoint": not (set(split["sft_question_ids"]) & set(split["dpo_question_ids"])),
        "exhaustive": len(set(split["sft_question_ids"]) | set(split["dpo_question_ids"]))
        == len(split["assignments"]),
        "counts_by_stratum": _counts_by_stratum(split["assignments"]),
        "source_files": {
            "question_source": str(question_source.resolve()),
            "question_source_sha256": _sha256_file(question_source),
            "expanded_alias_source": str(alias_source.resolve()),
            "expanded_alias_source_sha256": _sha256_file(alias_source),
        },
        "assignment_sha256": _sha256_json(
            [{"question_id": row["question_id"], "split": row["split"]} for row in split["assignments"]]
        ),
        "files": {key: str(path.resolve()) for key, path in paths.items() if key != "manifest"},
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split gold-supported BioASQ training questions into disjoint SFT and DPO pools."
    )
    parser.add_argument("--question-source", type=Path, default=DEFAULT_QUESTION_SOURCE)
    parser.add_argument("--expanded-alias-source", type=Path, default=DEFAULT_ALIAS_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sft-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_source = args.question_source.resolve()
    alias_source = args.expanded_alias_source.resolve()
    question_rows = _read_json(question_source)
    alias_rows = _read_json(alias_source)
    if len(question_rows) != 1_130:
        raise ValueError(f"Expected 1,130 gold-supported questions, got {len(question_rows):,}")
    if len(alias_rows) != 1_352:
        raise ValueError(f"Expected 1,352 expanded supported-alias rows, got {len(alias_rows):,}")
    split = build_split(
        question_rows,
        alias_rows,
        sft_fraction=args.sft_fraction,
        seed=args.seed,
    )
    manifest = write_split(
        args.output_dir.resolve(),
        split,
        question_source=question_source,
        alias_source=alias_source,
        sft_fraction=args.sft_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
