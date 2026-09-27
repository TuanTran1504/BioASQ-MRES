"""Prepare shared Qwen SFT splits directly from raw BioASQ factoid questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cse_dpo.normalize_set_answers import normalize_answer_surface
from src.utility.data import clean_text, convert_bioasq_question, resolve_prompt_instructions


def gold_aliases(value) -> list[str]:
    if isinstance(value, str):
        return [clean_text(value)] if clean_text(value) else []
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(alias for item in value for alias in gold_aliases(item)))


def supported_aliases(question: dict, mode="normalized") -> list[str]:
    # Search each snippet independently: never match document IDs, section labels,
    # or an expression assembled across the boundary between two snippets.
    if mode == "none":
        return gold_aliases(question.get("exact_answer"))
    if mode == "literal":
        return [alias for alias in gold_aliases(question.get("exact_answer"))
                if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", str(s.get("text", "")))
                       for s in question.get("snippets", []) if isinstance(s, dict))]
    if mode != "normalized":
        raise ValueError("Unknown matching mode")
    snippets = [
        f" {normalize_answer_surface(snippet.get('text', ''))} "
        for snippet in question.get("snippets", [])
        if isinstance(snippet, dict)
    ]
    return [
        alias for alias in gold_aliases(question.get("exact_answer"))
        if (key := normalize_answer_surface(alias))
        and any(f" {key} " in snippet for snippet in snippets)
    ]


def build_splits(questions: list[dict], test_questions: list[dict], dev_ratio=0.1, seed=3407,
                 filter_mode="normalized", alias_mode="per_alias", split_order="split_first"):
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between zero and one.")
    factoids = [q for q in questions if q.get("type") == "factoid"]
    ids = [clean_text(q.get("id")) for q in factoids]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Training factoid question IDs must be nonempty and unique.")
    test_ids = {clean_text(q.get("id")) for q in test_questions}
    if set(ids) & test_ids:
        raise ValueError("Training and official test question IDs overlap.")
    if alias_mode not in {"per_alias", "first", "all"}:
        raise ValueError("Unknown alias mode")
    if split_order not in {"split_first", "filter_first"}:
        raise ValueError("Unknown split order")
    eligible = {q["id"]: supported_aliases(q, filter_mode) for q in factoids}
    eligible = {qid: aliases for qid, aliases in eligible.items() if aliases}
    split_pool = set(ids) if split_order == "split_first" else set(eligible)
    dev_count = math.ceil(len(split_pool) * dev_ratio)
    if not 0 < dev_count < len(split_pool):
        raise ValueError("Need enough questions for nonempty train and dev splits.")
    dev_ids = set(random.Random(seed).sample(sorted(split_pool), dev_count))
    train_pool_ids = split_pool - dev_ids
    train_ids = train_pool_ids & set(eligible)
    if not train_ids:
        raise ValueError("No training questions remain after filtering; dev is kept unchanged.")
    args = argparse.Namespace(
        question_types=["factoid"], max_resources=0, max_resource_chars=0,
        resource_granularity="document", resource_selection="first",
        max_factoid_answers=1, max_summary_answers=5, max_list_items=100,
        prompt_file=str(ROOT / "prompts/factoid_single_answer_aligned.json"),
        prompt="factoid-single-answer-extractive-v1", prompt_registry_path=None,
    )
    instructions = resolve_prompt_instructions(args)
    train_rows, dev_rows = [], []
    for q in sorted(factoids, key=lambda item: item["id"]):
        if q["id"] not in train_ids | dev_ids:
            continue
        # Do not cap the gold aliases before filtering. Some questions have more
        # than five accepted spellings, and a later spelling may be in a snippet.
        aliases = gold_aliases(q["exact_answer"])
        args.max_factoid_answers = len(aliases)
        row = convert_bioasq_question(q, args, prompt_instructions=instructions)
        if row is None:
            raise ValueError(f"Cannot prepare selected question {q['id']}; refusing to silently drop it")
        row["supported_aliases"] = eligible.get(q["id"], [])
        if q["id"] in dev_ids:
            # One generation per question, scored against ALL accepted aliases.
            dev_rows.append(row)
        else:
            targets = eligible[q["id"]]
            if alias_mode == "first":
                targets = targets[:1]
            if alias_mode == "all":
                targets = ["[EE][BE]".join(targets)]
            for index, alias in enumerate(targets):
                train_rows.append({
                    **row, "id": f"{q['id']}__supported_alias_{index}",
                    "source_question_id": q["id"], "output": f"[BE]{alias}[EE]",
                })
    test_factoids = [q for q in test_questions if q.get("type") == "factoid"]
    snippet_matched_ids = {q["id"] for q in factoids if supported_aliases(q)}
    manifest = {
        "seed": seed, "dev_ratio": dev_ratio, "filter_mode": filter_mode, "alias_mode": alias_mode,
        "split_order": split_order,
        "matching": {
            "normalized": "Whole normalized token sequence within one raw snippet; uses normalize_answer_surface. Lexical presence, not semantic evidence verification or literal copying.",
            "literal": "Case-sensitive alias text with word boundaries inside one raw snippet.",
            "none": "All factoid questions with accepted answers; no snippet-presence filter.",
        }[filter_mode],
        "split_method": (
            "Sample all sorted factoid IDs first with random.Random(seed); ceil(N * dev_ratio) dev questions; keep dev unfiltered; filter training only, then expand training aliases."
            if split_order == "split_first" else
            "Filter first; sample sorted eligible IDs with random.Random(seed); ceil(N * dev_ratio) dev questions; expand only training aliases afterward."
        ),
        "raw_factoid_questions": len(factoids),
        "eligible_questions": len(eligible),
        "excluded_questions": len(set(ids) - train_ids - dev_ids),
        "questions_without_eligible_aliases": len(factoids) - len(eligible),
        "eligible_supported_aliases": sum(map(len, eligible.values())),
        "train_questions": len(train_ids), "train_alias_rows": len(train_rows),
        "train_questions_before_filter": len(train_pool_ids),
        "train_questions_removed_by_filter": len(train_pool_ids - train_ids),
        "dev_questions": len(dev_ids),
        "dev_questions_with_normalized_snippet_match": len(dev_ids & snippet_matched_ids),
        "dev_questions_without_normalized_snippet_match": len(dev_ids - snippet_matched_ids),
        "official_test_factoid_questions": len(test_factoids),
        "official_test_supported_questions": sum(bool(supported_aliases(q)) for q in test_factoids),
        "train_question_ids": sorted(train_ids), "dev_question_ids": sorted(dev_ids),
        "train_question_ids_before_filter": sorted(train_pool_ids),
        "train_filtered_out_question_ids": sorted(train_pool_ids - train_ids),
        "test_question_ids": sorted(q["id"] for q in test_factoids),
        "excluded_question_ids": sorted(set(ids) - train_ids - dev_ids),
        "question_id_splits_disjoint": True,
    }
    return train_rows, dev_rows, manifest


def prepare(input_path: Path, test_paths: list[Path], output_dir: Path, dev_ratio=0.1, seed=3407,
            filter_mode="normalized", alias_mode="per_alias", split_order="split_first"):
    questions = json.loads(input_path.read_text(encoding="utf-8"))["questions"]
    tests = [q for path in test_paths for q in json.loads(path.read_text(encoding="utf-8"))["questions"]]
    train, dev, manifest = build_splits(questions, tests, dev_ratio, seed, filter_mode, alias_mode, split_order)
    manifest["source_files"] = [
        {
            "path": path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in [input_path, *test_paths]
    ]
    question_rows = {}
    for row in train:
        qid = row["source_question_id"]
        question_rows[qid] = {**row, "id": qid, "output": "".join(f"[BE]{a}[EE]" for a in row["supported_aliases"])}
    files = {"train_prepared.json": train, "dev_prepared.json": dev,
             "train_questions.json": list(question_rows.values())}
    encoded = {name: json.dumps(value, ensure_ascii=False, indent=2) + "\n" for name, value in files.items()}
    manifest["output_sha256"] = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in encoded.items()
    }
    encoded["split_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    # Re-running an identical preparation is safe; changing a split requires a
    # different directory, so existing experiments retain their original data.
    for name, value in encoded.items():
        path = output_dir / name
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"Different data already exists at {path}; choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in encoded.items():
        (output_dir / name).write_bytes(value.encode("utf-8"))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/training13b.json")
    parser.add_argument("--test-input", type=Path, nargs="+", default=[ROOT / f"data/Task13BTest/13B{i}_golden.json" for i in range(1, 5)])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--filter-mode", choices=["normalized", "literal", "none"], default="normalized")
    parser.add_argument("--alias-mode", choices=["per_alias", "first", "all"], default="per_alias")
    parser.add_argument("--split-order", choices=["split_first", "filter_first"], default="split_first",
                        help="Default: split all factoids, retain unfiltered dev, filter only training. filter_first reproduces the earlier supported-dev protocol.")
    args = parser.parse_args()
    manifest = prepare(args.input, args.test_input, args.output_dir, args.dev_ratio, args.seed, args.filter_mode, args.alias_mode, args.split_order)
    print(json.dumps({key: value for key, value in manifest.items() if not isinstance(value, (dict, list))}, indent=2))


if __name__ == "__main__":
    main()
