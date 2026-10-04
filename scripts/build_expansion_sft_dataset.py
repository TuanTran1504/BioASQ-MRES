#!/usr/bin/env python3
"""Build question-level SFT data for controlled factoid answer expansion.

Gold aliases appear only in assistant targets. Model inputs contain the question
and the supplied snippets, matching gold-blind inference. The fixed outer split
is reconstructed from all BioASQ factoids before any extractability filtering.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cse_dpo.candidate_bank_class_judge import extract_snippets
from src.utility.data import build_resources, clean_text


DEFAULT_PROMPT = ROOT / "gadi_sft_8b_starter/prompts/equivalent_expansion_v1.txt"
DEFAULT_OUTPUT = (
    ROOT / "data/BioASQ_expansion_sft/seed3407_gold_aliases_v1"
)
RELATIONS = {
    "original",
    "synonym",
    "abbreviation_expansion",
    "nomenclature_variant",
    "spelling_or_inflection",
    "numerically_equivalent",
    "harmless_formatting",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def flatten_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        value = clean_text(value)
        return [value] if value else []
    if isinstance(value, list):
        return [alias for item in value for alias in flatten_aliases(item)]
    return []


def conservative_key(value: str) -> str:
    """Deduplicate case and whitespace only; punctuation remains meaningful."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def formatting_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def acronym(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    return "".join(word[0] for word in words if word).casefold()


def looks_like_abbreviation(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    return bool(compact) and len(compact) <= 12 and " " not in value and (
        any(char.isupper() for char in value) or any(char.isdigit() for char in value)
    )


def infer_relation(primary: str, alias: str) -> str:
    """Assign only obvious surface relations; accepted aliases default to synonym."""
    if formatting_key(primary) == formatting_key(alias):
        return "harmless_formatting"
    if looks_like_abbreviation(primary) and re.sub(r"[^A-Za-z0-9]", "", primary).casefold() == acronym(alias):
        return "abbreviation_expansion"
    if looks_like_abbreviation(alias) and re.sub(r"[^A-Za-z0-9]", "", alias).casefold() == acronym(primary):
        return "abbreviation_expansion"
    return "synonym"


def load_validated_variants(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if path is None:
        return {}
    by_question: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"question_id", "answer", "relation_type", "validated"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"{path}:{line_number}: missing required variant fields")
            if row["validated"] is not True:
                continue
            qid = clean_text(row["question_id"])
            answer = clean_text(row["answer"])
            relation = clean_text(row["relation_type"])
            if not qid or not answer or relation not in RELATIONS - {"original"}:
                raise ValueError(f"{path}:{line_number}: invalid validated variant")
            by_question.setdefault(qid, []).append(
                {"answer": answer, "relation_type": relation}
            )
    return by_question


def load_official_test_ids(test_dir: Path) -> tuple[set[str], list[dict[str, str]]]:
    ids: set[str] = set()
    sources: list[dict[str, str]] = []
    for path in sorted(test_dir.glob("*_golden.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for question in payload.get("questions", []):
            if question.get("type") == "factoid":
                ids.add(clean_text(question.get("id")))
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)})
    return ids, sources


def question_snippets(question: dict[str, Any]) -> list[dict[str, str]]:
    resources = build_resources(
        question,
        max_resources=0,
        max_resource_chars=0,
        question_text=clean_text(question.get("body")),
        resource_granularity="document",
        resource_selection="first",
    )
    snippets = extract_snippets(resources)
    if not snippets:
        raise ValueError(f"{question.get('id')}: no usable snippets")
    return snippets


def build_target(
    aliases: list[str],
    extra_variants: list[dict[str, str]],
    *,
    maximum_candidates: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    aliases = list(dict.fromkeys(alias for alias in aliases if clean_text(alias)))
    if not aliases:
        raise ValueError("Factoid question is missing accepted aliases")
    primary = aliases[0]
    candidates = [{"answer": primary, "relation_type": "original"}]
    provenance = [{"answer": primary, "source": "official_alias", "gold_alias_index": 0}]
    seen = {conservative_key(primary)}
    for index, alias in enumerate(aliases[1:], 1):
        key = conservative_key(alias)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"answer": alias, "relation_type": infer_relation(primary, alias)})
        provenance.append({"answer": alias, "source": "official_alias", "gold_alias_index": index})
    for variant in extra_variants:
        key = conservative_key(variant["answer"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dict(variant))
        provenance.append({"answer": variant["answer"], "source": "validated_variant"})
    return candidates[:maximum_candidates], provenance[:maximum_candidates]


def write_json(path: Path, value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    payload = text.encode("utf-8")
    # Write the exact bytes that are hashed. Path.write_text translates LF to
    # CRLF on Windows, which made the recorded manifest digest disagree with
    # the resulting file even though its parsed contents were correct.
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/training13b.json")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/Task13BTest")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validated-variants", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--internal-validation-ratio", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-candidates", type=int, default=10)
    args = parser.parse_args()

    if not 0 < args.dev_ratio < 1 or not 0 < args.internal_validation_ratio < 1:
        raise ValueError("Split ratios must be between zero and one")
    if args.folds < 2 or not 1 <= args.maximum_candidates <= 10:
        raise ValueError("Use at least two folds and between one and ten candidates")

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    factoids = [q for q in raw.get("questions", []) if q.get("type") == "factoid"]
    by_id = {clean_text(q.get("id")): q for q in factoids}
    if len(factoids) != 1600 or len(by_id) != len(factoids) or "" in by_id:
        raise ValueError(f"Expected 1,600 factoids with unique nonempty IDs; got {len(factoids)}/{len(by_id)}")

    all_ids = sorted(by_id)
    outer_dev_count = math.ceil(len(all_ids) * args.dev_ratio)
    outer_dev_ids = set(random.Random(args.seed).sample(all_ids, outer_dev_count))
    outer_train_ids = sorted(set(all_ids) - outer_dev_ids)
    if len(outer_train_ids) != 1440 or len(outer_dev_ids) != 160:
        raise ValueError("The fixed outer split must contain 1,440 train and 160 dev questions")

    official_test_ids, test_sources = load_official_test_ids(args.test_dir)
    if (set(outer_train_ids) | outer_dev_ids) & official_test_ids:
        raise ValueError("Training corpus overlaps the official test question IDs")

    internal_validation_count = math.ceil(len(outer_train_ids) * args.internal_validation_ratio)
    internal_validation_ids = set(
        random.Random(args.seed + 1).sample(outer_train_ids, internal_validation_count)
    )
    internal_train_ids = set(outer_train_ids) - internal_validation_ids

    shuffled_for_folds = list(outer_train_ids)
    random.Random(args.seed).shuffle(shuffled_for_folds)
    fold_by_id = {qid: index % args.folds for index, qid in enumerate(shuffled_for_folds)}

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    validated_variants = load_validated_variants(args.validated_variants)
    unknown_variant_ids = set(validated_variants) - set(outer_train_ids)
    if unknown_variant_ids:
        raise ValueError(f"Validated variants contain non-training IDs: {sorted(unknown_variant_ids)[:5]}")

    training_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidate_counts: Counter[int] = Counter()
    relation_counts: Counter[str] = Counter()
    extra_variant_count = 0

    for qid in outer_train_ids:
        question = by_id[qid]
        aliases = list(dict.fromkeys(flatten_aliases(question.get("exact_answer"))))
        snippets = question_snippets(question)
        candidates, provenance = build_target(
            aliases,
            validated_variants.get(qid, []),
            maximum_candidates=args.maximum_candidates,
        )
        if candidates[0]["relation_type"] != "original" or any(
            row["relation_type"] == "original" for row in candidates[1:]
        ):
            raise ValueError(f"{qid}: target must have exactly one original in first position")
        if len({conservative_key(row["answer"]) for row in candidates}) != len(candidates):
            raise ValueError(f"{qid}: duplicate target candidates")

        user_content = json.dumps(
            {
                "question": clean_text(question.get("body")),
                "snippets": [
                    {"id": snippet["snippet_id"], "text": snippet["text"]}
                    for snippet in snippets
                ],
            },
            ensure_ascii=False,
        )
        assistant_content = json.dumps({"answers": candidates}, ensure_ascii=False)
        row = {
            "question_id": qid,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        training_rows.append(row)

        candidate_counts[len(candidates)] += 1
        relation_counts.update(item["relation_type"] for item in candidates)
        extra_variant_count += sum(item["source"] == "validated_variant" for item in provenance)
        audit_rows.append(
            {
                "question_id": qid,
                "outer_split": "train",
                "internal_split": "validation" if qid in internal_validation_ids else "train",
                "crossfit_fold": fold_by_id[qid],
                "gold_aliases": aliases,
                "target_candidates": candidates,
                "candidate_provenance": provenance,
                "snippet_count": len(snippets),
            }
        )

    by_training_id = {row["question_id"]: row for row in training_rows}
    internal_train_rows = [by_training_id[qid] for qid in sorted(internal_train_ids)]
    internal_validation_rows = [by_training_id[qid] for qid in sorted(internal_validation_ids)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {
        "all_train.jsonl": write_jsonl(args.output_dir / "all_train.jsonl", training_rows),
        "train.jsonl": write_jsonl(args.output_dir / "train.jsonl", internal_train_rows),
        "validation.jsonl": write_jsonl(args.output_dir / "validation.jsonl", internal_validation_rows),
        "audit.jsonl": write_jsonl(args.output_dir / "audit.jsonl", audit_rows),
        "train_question_ids.json": write_json(args.output_dir / "train_question_ids.json", outer_train_ids),
        "dev_question_ids.json": write_json(args.output_dir / "dev_question_ids.json", sorted(outer_dev_ids)),
        "fold_assignments.json": write_json(args.output_dir / "fold_assignments.json", fold_by_id),
    }

    manifest = {
        "format_version": "bioasq-controlled-expansion-sft-v1",
        "seed": args.seed,
        "split_method": "Sample ceil(10%) of all 1,600 sorted factoid IDs with random.Random(seed) for unfiltered outer dev before constructing targets.",
        "raw_factoid_questions": len(factoids),
        "outer_train_questions": len(training_rows),
        "outer_dev_questions": len(outer_dev_ids),
        "internal_train_questions": len(internal_train_rows),
        "internal_validation_questions": len(internal_validation_rows),
        "crossfit_folds": args.folds,
        "train_dev_overlap": len(set(outer_train_ids) & outer_dev_ids),
        "official_test_overlap": len((set(outer_train_ids) | outer_dev_ids) & official_test_ids),
        "candidate_count": sum(len(row["target_candidates"]) for row in audit_rows),
        "mean_candidates_per_question": sum(k * v for k, v in candidate_counts.items()) / len(training_rows),
        "candidate_count_histogram": dict(sorted(candidate_counts.items())),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "validated_variant_count": extra_variant_count,
        "gold_usage": "Gold aliases occur only in assistant targets and audit metadata; system and user inputs contain no gold fields.",
        "target_policy": "First official alias is original; remaining official aliases are accepted equivalents with conservatively inferred relation labels; optional variants require validated=true.",
        "known_limitation": "Gold aliases alone provide one target candidate for most questions; add independently validated training-only variants before treating this as a diversity-complete expansion dataset.",
        "sources": [
            {"path": args.input.relative_to(ROOT).as_posix(), "sha256": sha256_file(args.input)},
            {"path": args.prompt.relative_to(ROOT).as_posix(), "sha256": sha256_file(args.prompt)},
            *test_sources,
        ],
        "output_sha256": output_hashes,
    }
    manifest_hash = write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({**manifest, "manifest_sha256": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()
