#!/usr/bin/env python
"""Build a verified answer-first synthetic BioASQ factoid QA pilot.

The pipeline is intentionally staged and resumable:

1. ``prepare``: select 250 training-only PubMed resources, exclude dev/test
   PMIDs, split 200/50 by source, balance clean/distractor arms, and retrieve
   related distractor candidates.
2. ``generate``: ask a teacher for two distinct questions conditioned on two
   exact continuous answer spans from each target resource.
3. ``verify``: construct the final evidence pack and ask a blinded verifier to
   extract each answer.  Distractor packs contain three related training-only
   resources that do not contain either intended answer.
4. ``finalize``: keep only locally valid, verifier-agreed examples and export
   generic JSONL plus SFT-compatible prepared JSON files.

No API calls occur during ``prepare`` or ``finalize``.  Every generation and
verification response is cached by its full prompt, model, and rubric.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "evidence_grounded_single_answer_qwen25_05b/train_prepared.json"
)
DEFAULT_DEV_SOURCE = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "single_answer_full_resources_qwen25_05b/eval_prepared.json"
)
DEFAULT_TEST_DIR = ROOT / "data/Task13BTest"
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/synthetic_factoid_qa/"
    "gpt41mini_answer_first_250x2_v2"
)
DEFAULT_API_KEY_FILE = ROOT / "open_ai_api.txt"
DEFAULT_GENERATOR_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_VERIFIER_MODEL = "gpt-4.1-mini-2025-04-14"
SEED = 3407
SOURCE_COUNT = 250
VALIDATION_SOURCE_COUNT = 50
QUESTIONS_PER_SOURCE = 2
DISTRACTOR_COUNT = 3
RETRIEVAL_CANDIDATE_COUNT = 30
GENERATION_RUBRIC_VERSION = "answer-first-bioasq-question-generation-v2"
VERIFICATION_RUBRIC_VERSION = "blinded-roundtrip-bioasq-verification-v1"

ANSWER_TYPES = {
    "gene",
    "mutation_or_variant",
    "protein_or_receptor",
    "rna_or_transcript",
    "disease_or_condition",
    "drug_or_treatment",
    "organism",
    "cell_or_cell_type",
    "anatomy_or_location",
    "biological_process",
    "molecular_function",
    "number_or_quantity",
    "percentage_or_rate",
    "date_or_time",
    "method_resource_or_tool",
    "structure_or_composition",
    "other",
}

SFT_INSTRUCTION = """You are a biomedical expert. Your task is to extract the single most relevant factoid answer from the provided PubMed resources. Relevant information is marked with [BS] and [ES].
Rules:
- Select exactly one short biomedical expression from the provided resources, preferably within [BS] and [ES], and preserve its source surface whenever possible.
- Do NOT translate, abbreviate, expand, normalize, paraphrase, combine, or infer an answer.
- Return exactly one expression only.
- Do NOT provide alternate answers, explanations, or extra text.
- Maintain this format strictly: [BE] extracted expression [EE]"""

GENERATION_SYSTEM_PROMPT = """
You create rigorously grounded biomedical factoid questions. Return JSON only.

Given one PubMed resource, first decide whether it explicitly supports at
least two different single-answer BioASQ-style factoid relations. If it does,
create exactly two questions. Work answer-first: independently select two
different short answer spans, then write one question for each span.

Requirements:
- Each answer must be copied exactly as one continuous span from text inside a
  supplied [BS]...[ES] region. Preserve punctuation, hyphens, capitalization,
  Unicode characters, spacing, and word order.
- Select different answer spans and genuinely different evidence relations for
  the two questions. Do not create two paraphrases of the same relation.
- If the resource cannot support two such questions, set eligible=false and
  return no items. Do not force a second question from a list-only resource.
- Each answer must denote one factoid target. Reject enumerations of three or
  more separate entities, values, features, drugs, genes, or other items.
- Questions must ask biomedical facts directly. Never ask what is mentioned in
  a title, abstract, article, paper, resource, snippet, passage, or supplied text.
- Do not leak the answer by repeating all of its content words in a different
  order, and do not write tautological questions.
- The resource must explicitly state the relation needed to answer the question.
  Do not require outside biomedical knowledge or inference across missing facts.
- The question must be natural and self-contained. It must uniquely request the
  selected answer, preserve every essential qualifier, and not contain the answer.
- Prefer errors relevant to biomedical extraction: entity type, relation direction,
  numerical attribution, disease/population qualifiers, molecular specificity,
  biological process/function, acronym expansion, and answer boundaries.
- Do not write yes/no, list, opinion, or multi-part questions.
- The answer should normally contain 1 to 12 words and must not be a whole
  explanatory sentence.
- support_quote must be an exact continuous quote from the [BS]...[ES] text and
  must contain the answer exactly.

Return exactly one of these forms:
{"eligible":false,"eligibility_basis":"why two distinct factoid relations are unavailable","items":[]}

or:
{"eligible":true,"eligibility_basis":"why two distinct factoid relations are explicitly supported","items":[
  {
    "question":"... ?",
    "answer":"exact continuous source span",
    "answer_type":"one allowed answer type",
    "required_relation":"short description",
    "essential_qualifiers":["..."],
    "support_quote":"exact continuous source quote",
    "basis":"one concise sentence explaining direct answerability"
  },
  {"question":"... ?", "answer":"...", "answer_type":"...",
   "required_relation":"...", "essential_qualifiers":[],
   "support_quote":"...", "basis":"..."}
]}
""".strip()

VERIFICATION_SYSTEM_PROMPT = """
You are a blinded biomedical factoid QA verifier. Return JSON only.

For each supplied question, use only the supplied PubMed resources and extract
the single shortest continuous source span that completely answers it. Preserve
the exact source surface, including punctuation and hyphens. You are not given
the intended answer.

Set status="accepted" only when:
- one resource explicitly states the requested relation;
- exactly one answer is supported by the complete evidence pack;
- the extracted answer preserves every essential qualifier; and
- the answer is an exact continuous span in the cited resource.

Set status="review" for ambiguity, multiple valid answers, missing relations,
outside-knowledge requirements, non-factoid questions, or unsupported answers.

Return exactly this form:
{"items":[
  {
    "synthetic_question_id":"supplied ID",
    "status":"accepted or review",
    "extracted_answer":"exact span or empty string",
    "resource_id":"supplied resource number or empty string",
    "unique_answer":true,
    "explicit_relation":true,
    "basis":"one concise sentence"
  },
  {"synthetic_question_id":"supplied ID","status":"accepted or review",
   "extracted_answer":"...","resource_id":"...","unique_answer":true,
   "explicit_relation":true,"basis":"..."}
]}
""".strip()


def digest(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_questions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else value.get("questions")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected list or object containing questions")
    return [row for row in rows if isinstance(row, dict) and row.get("type") == "factoid"]


def weak_surface(value: str) -> str:
    # Deliberately preserve punctuation and hyphens.
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def question_tokens(value: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", value.casefold()))


def jaccard(left: str, right: str) -> float:
    a, b = question_tokens(left), question_tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


METADATA_QUESTION_RE = re.compile(
    r"\b(title|abstract|article|paper|resource|snippet|passage|supplied text)\b",
    flags=re.I,
)


def is_list_like_answer(value: str) -> bool:
    """Flag clear enumerations while retaining ordinary compound factoid spans."""
    comma_count = value.count(",")
    semicolon_count = value.count(";")
    return semicolon_count >= 2 or comma_count >= 3 or (
        comma_count >= 2 and bool(re.search(r"\b(?:and|or)\b", value, flags=re.I))
    )


def answer_tokens_leaked(question: str, answer: str) -> bool:
    """Catch reordered/near-tautological leakage missed by substring matching."""
    answer_parts = question_tokens(answer)
    return bool(answer_parts) and answer_parts <= question_tokens(question)


def answer_type_mismatch(answer_type: str, answer: str) -> bool:
    if answer_type != "biological_process":
        return False
    return bool(
        re.search(
            r"\b(imaging|assay|analysis|method|approach|technique|scan|microscopy)\b",
            answer,
            flags=re.I,
        )
    )


def normalized_span_contains(container: str, span: str) -> bool:
    """Containment with punctuation preserved and alphanumeric boundaries guarded."""
    container_key = weak_surface(container)
    span_key = weak_surface(span)
    if not span_key:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(span_key)}(?!\w)", container_key))


def pubmed_id_from_text(value: str) -> str | None:
    match = re.search(r"PubMed\s+ID:\s*([0-9]+)", value, flags=re.I)
    return match.group(1) if match else None


def pubmed_id_from_url(value: str) -> str | None:
    match = re.search(r"/pubmed/([0-9]+)", str(value), flags=re.I)
    return match.group(1) if match else None


def marked_spans(value: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"\[BS\](.*?)\[ES\]", value, flags=re.I | re.S)
        if match.group(1).strip()
    ]


def resources_from_prepared_row(row: dict[str, Any]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for key, value in row.items():
        match = re.fullmatch(r"input_(\d+)", str(key))
        if match and int(match.group(1)) >= 2 and str(value or "").strip():
            result.append((int(match.group(1)) - 1, str(value).strip()))
    return sorted(result)


def protected_pmids(dev_source: Path, test_dir: Path) -> set[str]:
    values: set[str] = set()
    for row in read_questions(dev_source):
        for _, text in resources_from_prepared_row(row):
            if pmid := pubmed_id_from_text(text):
                values.add(pmid)
    for path in sorted(test_dir.glob("*_golden.json")):
        for row in read_questions(path):
            for document in row.get("documents") or []:
                if pmid := pubmed_id_from_url(document):
                    values.add(pmid)
            for snippet in row.get("snippets") or []:
                if isinstance(snippet, dict) and (pmid := pubmed_id_from_url(snippet.get("document", ""))):
                    values.add(pmid)
    return values


def build_source_pool(source: Path, protected: set[str]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    pool: list[dict[str, Any]] = []
    for row in read_questions(source):
        for resource_id, text in resources_from_prepared_row(row):
            spans = marked_spans(text)
            pmid = pubmed_id_from_text(text)
            if not pmid or pmid in protected or not spans:
                continue
            text_hash = digest(text)
            key = (pmid, text_hash)
            if key in seen:
                continue
            seen.add(key)
            marked_text = " ".join(spans)
            if len(marked_text) < 80 or len(marked_text) > 6000:
                continue
            pool.append(
                {
                    "pool_id": f"pool_{len(pool):05d}",
                    "source_question_id": str(row["id"]),
                    "source_question": str(row.get("input_1") or "").strip(),
                    "source_resource_id": resource_id,
                    "pubmed_id": pmid,
                    "resource_text": text,
                    "marked_spans": spans,
                    "text_sha256": text_hash,
                }
            )
    return pool


def retrieve_distractors(
    pool: list[dict[str, Any]], selected_indices: list[int], count: int
) -> dict[int, list[int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    texts = [" ".join(row["marked_spans"]) for row in pool]
    matrix = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=80000,
        sublinear_tf=True,
    ).fit_transform(texts)
    result: dict[int, list[int]] = {}
    for index in selected_indices:
        scores = linear_kernel(matrix[index], matrix).ravel()
        ranked = scores.argsort()[::-1]
        source = pool[index]
        choices: list[int] = []
        used_pmids = {source["pubmed_id"]}
        for candidate_index in ranked:
            candidate_index = int(candidate_index)
            candidate = pool[candidate_index]
            if candidate_index == index:
                continue
            if candidate["pubmed_id"] in used_pmids:
                continue
            if candidate["source_question_id"] == source["source_question_id"]:
                continue
            choices.append(candidate_index)
            used_pmids.add(candidate["pubmed_id"])
            if len(choices) >= count:
                break
        if len(choices) < count:
            raise RuntimeError(f"Insufficient distractor candidates for {source['pool_id']}")
        result[index] = choices
    return result


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    protected = protected_pmids(Path(args.dev_source), Path(args.test_dir))
    pool = build_source_pool(Path(args.source), protected)
    by_question: dict[str, list[int]] = {}
    for index, row in enumerate(pool):
        by_question.setdefault(row["source_question_id"], []).append(index)
    rng = random.Random(args.seed)
    question_ids = list(by_question)
    rng.shuffle(question_ids)
    selected_indices: list[int] = []
    selected_pmids: set[str] = set()
    for question_id in question_ids:
        candidates = list(by_question[question_id])
        rng.shuffle(candidates)
        chosen = next(
            (idx for idx in candidates if pool[idx]["pubmed_id"] not in selected_pmids),
            None,
        )
        if chosen is None:
            continue
        selected_indices.append(chosen)
        selected_pmids.add(pool[chosen]["pubmed_id"])
        if len(selected_indices) >= args.source_count:
            break
    if len(selected_indices) != args.source_count:
        raise RuntimeError(f"Selected {len(selected_indices)} sources, expected {args.source_count}")

    # Source-disjoint split and balanced evidence arm within each split.
    rng.shuffle(selected_indices)
    train_count = args.source_count - args.validation_source_count
    split_indices = {
        "train": selected_indices[:train_count],
        "validation": selected_indices[train_count:],
    }
    retrieval = retrieve_distractors(pool, selected_indices, args.retrieval_candidate_count)
    manifest: list[dict[str, Any]] = []
    for split, indices in split_indices.items():
        rng.shuffle(indices)
        for split_position, index in enumerate(indices):
            arm = "clean" if split_position % 2 == 0 else "distractor"
            source = pool[index]
            manifest.append(
                {
                    "synthetic_source_id": f"synthetic_source_{len(manifest):04d}",
                    "split": split,
                    "evidence_arm": arm,
                    "target": source,
                    "distractor_candidate_pool_ids": [
                        pool[candidate_index]["pool_id"]
                        for candidate_index in retrieval[index]
                    ],
                }
            )
    pool_path = output_root / "eligible_source_pool.jsonl"
    manifest_path = output_root / "source_manifest.jsonl"
    write_jsonl(pool_path, pool)
    write_jsonl(manifest_path, manifest)
    summary = {
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(args.source).resolve()),
        "dev_source": str(Path(args.dev_source).resolve()),
        "test_dir": str(Path(args.test_dir).resolve()),
        "protected_pmid_count": len(protected),
        "eligible_unique_resource_count": len(pool),
        "selected_source_count": len(manifest),
        "expected_raw_question_count": len(manifest) * args.questions_per_source,
        "split_counts": dict(Counter(row["split"] for row in manifest)),
        "evidence_arm_counts": dict(Counter(row["evidence_arm"] for row in manifest)),
        "split_arm_counts": dict(Counter(f"{row['split']}:{row['evidence_arm']}" for row in manifest)),
        "unique_selected_pmid_count": len({row["target"]["pubmed_id"] for row in manifest}),
        "selected_protected_pmid_overlap": len(
            {row["target"]["pubmed_id"] for row in manifest} & protected
        ),
        "seed": args.seed,
        "questions_per_source": args.questions_per_source,
        "distractor_count": args.distractor_count,
        "source_pool_jsonl": str(pool_path.resolve()),
        "source_manifest_jsonl": str(manifest_path.resolve()),
    }
    write_json(output_root / "prepare_summary.json", summary)
    return summary


def generation_user_prompt(source: dict[str, Any], feedback: str = "") -> str:
    allowed = ", ".join(sorted(ANSWER_TYPES))
    return f"""Allowed answer_type values: {allowed}

Target resource:
Resource 1 (PubMed {source['target']['pubmed_id']}):
{source['target']['resource_text']}

Generate exactly two distinct answer-first factoid questions from this resource.
{feedback}"""


def validate_generated_items(
    items: Any, source: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(items, list) or len(items) != QUESTIONS_PER_SOURCE:
        raise ValueError(f"items must contain exactly {QUESTIONS_PER_SOURCE} records")
    expected = {
        "question",
        "answer",
        "answer_type",
        "required_relation",
        "essential_qualifiers",
        "support_quote",
        "basis",
    }
    marked_text = "\n".join(source["target"]["marked_spans"])
    answers: list[str] = []
    questions: list[str] = []
    relations: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(f"item {index} must contain exactly {sorted(expected)}")
        for key in ("question", "answer", "required_relation", "support_quote", "basis"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"item {index} {key} must be a non-empty string")
            item[key] = item[key].strip()
        if item["answer_type"] not in ANSWER_TYPES:
            raise ValueError(f"item {index} has invalid answer_type")
        if not isinstance(item.get("essential_qualifiers"), list) or any(
            not isinstance(x, str) or not x.strip() for x in item["essential_qualifiers"]
        ):
            raise ValueError(f"item {index} essential_qualifiers must be a string list")
        if item["answer"] not in marked_text:
            raise ValueError(f"item {index} answer is not an exact marked source span")
        if item["support_quote"] not in marked_text or item["answer"] not in item["support_quote"]:
            # Repair only audit metadata when the answer itself is already exact.
            item["support_quote"] = next(
                span for span in source["target"]["marked_spans"] if item["answer"] in span
            )
        if METADATA_QUESTION_RE.search(item["question"]):
            raise ValueError(f"item {index} asks about document metadata rather than a biomedical fact")
        if weak_surface(item["answer"]) in weak_surface(item["question"]):
            raise ValueError(f"item {index} leaks its answer in the question")
        if answer_tokens_leaked(item["question"], item["answer"]):
            raise ValueError(f"item {index} repeats all answer tokens in the question")
        if not item["question"].endswith("?"):
            raise ValueError(f"item {index} question must end with ?")
        if len(item["answer"].split()) > 12 or len(item["answer"]) > 180:
            raise ValueError(f"item {index} answer is too long")
        if is_list_like_answer(item["answer"]):
            raise ValueError(f"item {index} answer is a list rather than one factoid target")
        if answer_type_mismatch(item["answer_type"], item["answer"]):
            raise ValueError(f"item {index} answer conflicts with its answer_type")
        if len(re.sub(r"[^A-Za-z0-9]", "", item["answer"])) < 2:
            raise ValueError(f"item {index} answer is not a meaningful expression")
        answers.append(weak_surface(item["answer"]))
        questions.append(item["question"])
        relations.append(item["required_relation"])
    if len(set(answers)) != QUESTIONS_PER_SOURCE:
        raise ValueError("the two generated answers must be different spans")
    if len({weak_surface(value) for value in relations}) != QUESTIONS_PER_SOURCE:
        raise ValueError("the two generated questions must target different relations")
    if jaccard(relations[0], relations[1]) > 0.60:
        raise ValueError("the two required relations are too similar")
    if jaccard(questions[0], questions[1]) > 0.75:
        raise ValueError("the two generated questions are too similar")
    return items


def validate_generated(value: Any, source: dict[str, Any]) -> dict[str, Any]:
    expected = {"eligible", "eligibility_basis", "items"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"response must contain exactly {sorted(expected)}")
    if not isinstance(value.get("eligible"), bool):
        raise ValueError("eligible must be a boolean")
    if not isinstance(value.get("eligibility_basis"), str) or not value["eligibility_basis"].strip():
        raise ValueError("eligibility_basis must be a non-empty string")
    value["eligibility_basis"] = value["eligibility_basis"].strip()
    if not value["eligible"]:
        if value.get("items") != []:
            raise ValueError("ineligible resources must return an empty items list")
        return value
    validate_generated_items(value.get("items"), source)
    return value


def select_evidence_pack(
    source: dict[str, Any], items: list[dict[str, Any]], pool_by_id: dict[str, dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], str]:
    target = source["target"]
    resources = [{"kind": "target", **target}]
    if source["evidence_arm"] == "distractor":
        answer_keys = [weak_surface(item["answer"]) for item in items]
        for pool_id in source["distractor_candidate_pool_ids"]:
            candidate = pool_by_id[pool_id]
            candidate_key = weak_surface(" ".join(candidate["marked_spans"]))
            if any(answer_key and answer_key in candidate_key for answer_key in answer_keys):
                continue
            resources.append({"kind": "distractor", **candidate})
            if len(resources) == args.distractor_count + 1:
                break
        if len(resources) != args.distractor_count + 1:
            raise ValueError(f"Could not find {args.distractor_count} answer-free distractors")
    rng = random.Random(f"{args.seed}:{source['synthetic_source_id']}")
    rng.shuffle(resources)
    packed = []
    target_resource_id = ""
    for index, resource in enumerate(resources, 1):
        packed.append(
            {
                "resource_id": str(index),
                "kind": resource["kind"],
                "pool_id": resource["pool_id"],
                "pubmed_id": resource["pubmed_id"],
                "text": resource["resource_text"],
            }
        )
        if resource["kind"] == "target":
            target_resource_id = str(index)
    return packed, target_resource_id


def verification_user_prompt(
    source: dict[str, Any], items: list[dict[str, Any]], evidence_pack: list[dict[str, Any]], feedback: str = ""
) -> str:
    questions = "\n".join(
        f"- {source['synthetic_source_id']}::q{index}: {item['question']}"
        for index, item in enumerate(items, 1)
    )
    resources = "\n\n".join(
        f"Resource {resource['resource_id']} (PubMed {resource['pubmed_id']}):\n{resource['text']}"
        for resource in evidence_pack
    )
    return f"""Questions:
{questions}

PubMed resources:

{resources}

Return one verification item for every supplied question ID in the same order.
{feedback}"""


def validate_verification(
    value: Any, source: dict[str, Any], items: list[dict[str, Any]], evidence_pack: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"items"}:
        raise ValueError("verification response must contain only items")
    records = value.get("items")
    if not isinstance(records, list) or len(records) != len(items):
        raise ValueError("verification item count does not match question count")
    expected = {
        "synthetic_question_id",
        "status",
        "extracted_answer",
        "resource_id",
        "unique_answer",
        "explicit_relation",
        "basis",
    }
    resource_by_id = {resource["resource_id"]: resource for resource in evidence_pack}
    wanted_ids = [f"{source['synthetic_source_id']}::q{i}" for i in range(1, len(items) + 1)]
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError(f"verification item {index} has wrong keys")
        if record.get("synthetic_question_id") != wanted_ids[index]:
            raise ValueError(f"verification item {index} has wrong question ID")
        if record.get("status") not in {"accepted", "review"}:
            raise ValueError(f"verification item {index} has invalid status")
        if not isinstance(record.get("extracted_answer"), str):
            raise ValueError(f"verification item {index} extracted_answer must be string")
        resource_id = str(record.get("resource_id") or "")
        if resource_id and resource_id not in resource_by_id:
            # The evidence labels show both a compact resource number and PMID.
            # Accept a PMID citation only when it maps unambiguously to one
            # supplied resource, then canonicalize it to the resource number.
            matching_ids = [
                candidate_id
                for candidate_id, resource in resource_by_id.items()
                if str(resource.get("pubmed_id") or "") == resource_id
            ]
            if len(matching_ids) == 1:
                resource_id = matching_ids[0]
                record["resource_id"] = resource_id
            else:
                raise ValueError(f"verification item {index} cites unknown resource")
        if not isinstance(record.get("unique_answer"), bool) or not isinstance(record.get("explicit_relation"), bool):
            raise ValueError(f"verification item {index} boolean checks are invalid")
        if not isinstance(record.get("basis"), str) or not record["basis"].strip():
            raise ValueError(f"verification item {index} basis is empty")
        if record["status"] == "accepted":
            if not resource_id or not record["extracted_answer"].strip():
                raise ValueError(f"accepted verification item {index} lacks answer/resource")
            if record["extracted_answer"] not in resource_by_id[resource_id]["text"]:
                raise ValueError(f"verification answer {index} is not an exact cited-resource span")
    return value


def read_api_key(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    value = next((line for line in lines if line and not line.startswith("#")), "")
    if value.startswith("OPENAI_API_KEY="):
        value = value.split("=", 1)[1].strip()
    value = value.strip().strip('"').strip("'")
    if not value:
        raise ValueError(f"No API key found in {path}")
    return value


class CachedJsonCaller:
    def __init__(
        self,
        args: argparse.Namespace,
        stage: str,
        system_prompt: str,
        rubric_version: str,
        model: str,
    ):
        self.args = args
        self.stage = stage
        self.system_prompt = system_prompt
        self.rubric_version = rubric_version
        self.model = model
        self.cache_dir = Path(args.output_root) / "cache" / stage
        self.error_dir = Path(args.output_root) / "errors" / stage
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)
        self.new_api_calls = 0
        self.rate_limit_hits = 0

    def cache_key(self, record_id: str, prompt: str) -> str:
        return digest(
            {
                "stage": self.stage,
                "record_id": record_id,
                "model": self.model,
                "rubric_version": self.rubric_version,
                "system_prompt": self.system_prompt,
                "user_prompt": prompt,
            }
        )

    def call_api(self, prompt: str, api_key: str) -> dict[str, Any]:
        import requests

        response = requests.post(
            self.args.endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.args.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.args.timeout_seconds,
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

    @staticmethod
    def is_rate_limit(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        return bool(
            (response is not None and getattr(response, "status_code", None) == 429)
            or "429" in str(exc)
        )

    def run_one(
        self,
        record_id: str,
        prompt_builder: Callable[[str], str],
        validator: Callable[[Any], dict[str, Any]],
        api_key: str,
    ) -> tuple[dict[str, Any], str]:
        base_prompt = prompt_builder("")
        cache_path = self.cache_dir / f"{self.cache_key(record_id, base_prompt)}.json"
        if cache_path.exists():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            validator(value)
            return value, "cache"
        if self.args.max_new_calls is not None and self.new_api_calls >= self.args.max_new_calls:
            raise RuntimeError("max_new_calls reached")
        error_path = self.error_dir / f"{self.cache_key(record_id, base_prompt)}.json"
        feedback = ""
        if error_path.exists():
            previous = json.loads(error_path.read_text(encoding="utf-8"))
            if previous.get("error"):
                feedback = (
                    f"A previous run failed validation: {previous['error']}. "
                    "Start with a corrected complete JSON object."
                )
        validation_attempt = 0
        rate_limit_attempt = 0
        last_error: Exception | None = None
        last_value: Any = None
        while validation_attempt <= self.args.max_retries:
            prompt = prompt_builder(feedback)
            try:
                if self.args.request_delay_seconds and self.new_api_calls:
                    time.sleep(self.args.request_delay_seconds)
                self.new_api_calls += 1
                value = self.call_api(prompt, api_key)
                last_value = value
                validator(value)
                cache_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
                return value, "api"
            except Exception as exc:
                last_error = exc
                if self.is_rate_limit(exc):
                    self.rate_limit_hits += 1
                    if rate_limit_attempt >= self.args.rate_limit_max_retries:
                        break
                    delay = min(
                        self.args.rate_limit_max_sleep_seconds,
                        self.args.rate_limit_initial_sleep_seconds * (2**rate_limit_attempt),
                    )
                    time.sleep(delay + random.uniform(0, min(5.0, delay * 0.1)))
                    rate_limit_attempt += 1
                    continue
                validation_attempt += 1
                feedback = (
                    f"Previous response failed validation: {exc}. Return a corrected complete JSON object."
                )
                if validation_attempt <= self.args.max_retries:
                    time.sleep(2 ** (validation_attempt - 1))
        write_json(
            error_path,
            {"record_id": record_id, "error": str(last_error), "last_value": last_value},
        )
        raise RuntimeError(f"{record_id} failed after retries: {last_error}")


def selected_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(args.output_root) / "source_manifest.jsonl")
    if not rows:
        raise FileNotFoundError("source_manifest.jsonl is missing; run --phase prepare first")
    return rows if args.source_limit == 0 else rows[: args.source_limit]


def run_generation(args: argparse.Namespace, *, dry_run: bool = False) -> dict[str, Any]:
    sources = selected_manifest(args)
    if dry_run:
        print(generation_user_prompt(sources[0]))
        return {"status": "dry_run", "source_count": len(sources)}
    api_key = read_api_key(Path(args.api_key_file))
    caller = CachedJsonCaller(
        args,
        "generation",
        GENERATION_SYSTEM_PROMPT,
        GENERATION_RUBRIC_VERSION,
        args.generator_model,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, source in enumerate(sources, 1):
        try:
            value, origin = caller.run_one(
                source["synthetic_source_id"],
                lambda feedback, source=source: generation_user_prompt(source, feedback),
                lambda value, source=source: validate_generated(value, source),
                api_key,
            )
            rows.append(
                {
                    **source,
                    "generation_eligible": value["eligible"],
                    "generation_eligibility_basis": value["eligibility_basis"],
                    "generated_items": value["items"],
                    "generation_origin": origin,
                }
            )
        except Exception as exc:
            failures.append({"synthetic_source_id": source["synthetic_source_id"], "error": str(exc)})
            if "max_new_calls reached" in str(exc):
                break
        if index == 1 or index % args.progress_every == 0 or index == len(sources):
            print(f"generation {index}/{len(sources)} | cached/complete {len(rows)} | new calls {caller.new_api_calls}")
    path = Path(args.output_root) / "generated_sources.jsonl"
    existing = {row["synthetic_source_id"]: row for row in read_jsonl(path)}
    existing.update({row["synthetic_source_id"]: row for row in rows})
    write_jsonl(path, [existing[key] for key in sorted(existing)])
    write_json(Path(args.output_root) / "generation_failures.json", failures)
    summary = {
        "status": "complete" if len(existing) == len(sources) else "partial",
        "selected_source_count": len(sources),
        "generated_source_count": len(existing),
        "eligible_source_count": sum(
            bool(row.get("generation_eligible", True) and row.get("generated_items"))
            for row in existing.values()
        ),
        "ineligible_source_count": sum(
            row.get("generation_eligible") is False for row in existing.values()
        ),
        "ineligible_source_ids": sorted(
            row["synthetic_source_id"]
            for row in existing.values()
            if row.get("generation_eligible") is False
        ),
        "generated_question_count": sum(len(row["generated_items"]) for row in existing.values()),
        "new_api_calls": caller.new_api_calls,
        "rate_limit_hits": caller.rate_limit_hits,
        "failure_count": len(failures),
        "model": args.generator_model,
        "rubric_version": GENERATION_RUBRIC_VERSION,
    }
    write_json(Path(args.output_root) / "generation_summary.json", summary)
    return summary


def run_verification(args: argparse.Namespace, *, dry_run: bool = False) -> dict[str, Any]:
    expected_sources = selected_manifest(args)
    expected_ids = {row["synthetic_source_id"] for row in expected_sources}
    generated_all = [
        row
        for row in read_jsonl(Path(args.output_root) / "generated_sources.jsonl")
        if row["synthetic_source_id"] in expected_ids
    ]
    generated_ids = {row["synthetic_source_id"] for row in generated_all}
    missing_ids = sorted(expected_ids - generated_ids)
    ineligible_ids = sorted(
        row["synthetic_source_id"]
        for row in generated_all
        if row.get("generation_eligible") is False or not row.get("generated_items")
    )
    generated = [
        row
        for row in generated_all
        if row.get("generation_eligible", True) and row.get("generated_items")
    ]
    if not generated_all:
        raise FileNotFoundError("No generated sources; run --phase generate first")
    if missing_ids or ineligible_ids:
        print(
            f"verification will process {len(generated)}/{len(expected_sources)} eligible sources; "
            f"{len(missing_ids)} generation failures and {len(ineligible_ids)} ineligible sources "
            "will count as rejected slots"
        )
    if not generated:
        summary = {
            "status": "complete",
            "selected_source_count": len(expected_sources),
            "generated_source_count": len(generated_all),
            "eligible_generated_source_count": 0,
            "ineligible_source_count": len(ineligible_ids),
            "ineligible_source_ids": ineligible_ids,
            "missing_generation_source_count": len(missing_ids),
            "missing_generation_source_ids": missing_ids,
            "verified_source_count": 0,
            "new_api_calls": 0,
            "rate_limit_hits": 0,
            "failure_count": 0,
            "model": args.verifier_model,
            "rubric_version": VERIFICATION_RUBRIC_VERSION,
        }
        write_json(Path(args.output_root) / "verification_summary.json", summary)
        return summary
    pool_by_id = {
        row["pool_id"]: row
        for row in read_jsonl(Path(args.output_root) / "eligible_source_pool.jsonl")
    }
    packed_sources = []
    for source in generated:
        pack, target_id = select_evidence_pack(source, source["generated_items"], pool_by_id, args)
        packed_sources.append({**source, "evidence_pack": pack, "target_resource_id": target_id})
    if dry_run:
        print(verification_user_prompt(
            packed_sources[0], packed_sources[0]["generated_items"], packed_sources[0]["evidence_pack"]
        ))
        return {"status": "dry_run", "source_count": len(packed_sources)}
    api_key = read_api_key(Path(args.api_key_file))
    caller = CachedJsonCaller(
        args,
        "verification",
        VERIFICATION_SYSTEM_PROMPT,
        VERIFICATION_RUBRIC_VERSION,
        args.verifier_model,
    )
    output: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, source in enumerate(packed_sources, 1):
        try:
            value, origin = caller.run_one(
                source["synthetic_source_id"],
                lambda feedback, source=source: verification_user_prompt(
                    source, source["generated_items"], source["evidence_pack"], feedback
                ),
                lambda value, source=source: validate_verification(
                    value, source, source["generated_items"], source["evidence_pack"]
                ),
                api_key,
            )
            output.append({**source, "verification_items": value["items"], "verification_origin": origin})
        except Exception as exc:
            failures.append({"synthetic_source_id": source["synthetic_source_id"], "error": str(exc)})
            if "max_new_calls reached" in str(exc):
                break
        if index == 1 or index % args.progress_every == 0 or index == len(packed_sources):
            print(f"verification {index}/{len(packed_sources)} | cached/complete {len(output)} | new calls {caller.new_api_calls}")
    path = Path(args.output_root) / "verified_sources.jsonl"
    existing = {row["synthetic_source_id"]: row for row in read_jsonl(path)}
    existing.update({row["synthetic_source_id"]: row for row in output})
    write_jsonl(path, [existing[key] for key in sorted(existing)])
    write_json(Path(args.output_root) / "verification_failures.json", failures)
    summary = {
        "status": "complete" if len(existing) == len(packed_sources) else "partial",
        "selected_source_count": len(expected_sources),
        "generated_source_count": len(generated_all),
        "eligible_generated_source_count": len(packed_sources),
        "ineligible_source_count": len(ineligible_ids),
        "ineligible_source_ids": ineligible_ids,
        "missing_generation_source_count": len(missing_ids),
        "missing_generation_source_ids": missing_ids,
        "verified_source_count": len(existing),
        "new_api_calls": caller.new_api_calls,
        "rate_limit_hits": caller.rate_limit_hits,
        "failure_count": len(failures),
        "model": args.verifier_model,
        "rubric_version": VERIFICATION_RUBRIC_VERSION,
    }
    write_json(Path(args.output_root) / "verification_summary.json", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    sources = read_jsonl(Path(args.output_root) / "verified_sources.jsonl")
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    generation_invalid_sources: dict[str, str] = {}
    for source in sources:
        try:
            validate_generated_items(source.get("generated_items"), source)
            generation_error = ""
        except Exception as exc:
            generation_error = str(exc)
            generation_invalid_sources[source["synthetic_source_id"]] = generation_error
        by_id = {item["synthetic_question_id"]: item for item in source["verification_items"]}
        for index, generated in enumerate(source["generated_items"], 1):
            question_id = f"{source['synthetic_source_id']}::q{index}"
            verified = by_id[question_id]
            generated_key = weak_surface(generated["answer"])
            verified_key = weak_surface(verified["extracted_answer"])
            if generated_key and generated_key == verified_key:
                match_type = "exact"
            elif normalized_span_contains(verified["extracted_answer"], generated["answer"]):
                # The blinded verifier may return a longer clause. Accept only
                # when the intended exact source span is wholly inside it; the
                # reverse direction could omit an essential qualifier.
                match_type = "verifier_superspan"
            else:
                match_type = "mismatch"
            answer_match = match_type in {"exact", "verifier_superspan"}
            target_match = str(verified["resource_id"]) == str(source["target_resource_id"])
            keep = bool(
                not generation_error
                and verified["status"] == "accepted"
                and verified["unique_answer"]
                and verified["explicit_relation"]
                and answer_match
                and target_match
            )
            row = {
                "synthetic_question_id": question_id,
                "synthetic_source_id": source["synthetic_source_id"],
                "split": source["split"],
                "evidence_arm": source["evidence_arm"],
                "question": generated["question"],
                "answer": generated["answer"],
                "output": f"[BE]{generated['answer']}[EE]",
                "answer_type": generated["answer_type"],
                "required_relation": generated["required_relation"],
                "essential_qualifiers": generated["essential_qualifiers"],
                "support_quote": generated["support_quote"],
                "generation_basis": generated["basis"],
                "generation_validation_passed": not bool(generation_error),
                "generation_validation_error": generation_error,
                "target_resource_id": source["target_resource_id"],
                "target_pubmed_id": source["target"]["pubmed_id"],
                "evidence_pack": source["evidence_pack"],
                "verification": verified,
                "verification_exact_answer_match": match_type == "exact",
                "verification_answer_match": answer_match,
                "verification_match_type": match_type,
                "verification_target_resource_match": target_match,
                "accepted": keep,
            }
            (accepted if keep else review).append(row)

    output_root = Path(args.output_root)
    write_jsonl(output_root / "synthetic_accepted_all.jsonl", accepted)
    write_jsonl(output_root / "synthetic_review.jsonl", review)
    for split in ("train", "validation"):
        split_rows = [row for row in accepted if row["split"] == split]
        write_jsonl(output_root / f"synthetic_{split}.jsonl", split_rows)
        prepared = []
        for row in split_rows:
            item = {
                "id": row["synthetic_question_id"],
                "type": "factoid",
                "instruction": SFT_INSTRUCTION,
                "input_1": row["question"],
                "output": row["output"],
            }
            for resource_index, resource in enumerate(row["evidence_pack"], 2):
                item[f"input_{resource_index}"] = resource["text"]
            prepared.append(item)
        write_json(output_root / f"synthetic_{split}_prepared.json", prepared)

    csv_path = output_root / "synthetic_review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "synthetic_question_id", "split", "evidence_arm", "question", "answer",
            "target_pubmed_id", "generation_validation_passed",
            "generation_validation_error", "verification_match_type",
            "verification_answer_match", "verification_target_resource_match", "verification",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in review:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list)) else row.get(key, "")
                for key in fields
            })
    all_rows = accepted + review
    summary = {
        "status": "finalized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verified_source_count": len(sources),
        "verified_question_count": len(all_rows),
        "accepted_question_count": len(accepted),
        "review_question_count": len(review),
        "acceptance_rate": len(accepted) / len(all_rows) if all_rows else 0.0,
        "generation_invalid_source_count": len(generation_invalid_sources),
        "generation_invalid_sources": generation_invalid_sources,
        "verification_match_type_counts": dict(
            Counter(row["verification_match_type"] for row in all_rows)
        ),
        "accepted_match_type_counts": dict(
            Counter(row["verification_match_type"] for row in accepted)
        ),
        "accepted_split_counts": dict(Counter(row["split"] for row in accepted)),
        "accepted_arm_counts": dict(Counter(row["evidence_arm"] for row in accepted)),
        "accepted_answer_type_counts": dict(Counter(row["answer_type"] for row in accepted)),
        "source_overlap_between_splits": len(
            {row["synthetic_source_id"] for row in accepted if row["split"] == "train"}
            & {row["synthetic_source_id"] for row in accepted if row["split"] == "validation"}
        ),
        "normalization": "NFKC + casefold + whitespace collapse; punctuation and hyphens preserved",
        "answer_agreement_rule": (
            "exact weak-surface match or intended answer wholly contained in a longer "
            "verifier span; reverse containment is rejected"
        ),
        "real_dev_unchanged": True,
        "real_test_unchanged": True,
    }
    write_json(output_root / "final_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "generate", "verify", "finalize", "all"], default="prepare")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dev-source", type=Path, default=DEFAULT_DEV_SOURCE)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument(
        "--model",
        default=None,
        help="Backward-compatible override that sets both generator and verifier models",
    )
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--source-count", type=int, default=SOURCE_COUNT)
    parser.add_argument("--validation-source-count", type=int, default=VALIDATION_SOURCE_COUNT)
    parser.add_argument("--questions-per-source", type=int, default=QUESTIONS_PER_SOURCE)
    parser.add_argument("--distractor-count", type=int, default=DISTRACTOR_COUNT)
    parser.add_argument("--retrieval-candidate-count", type=int, default=RETRIEVAL_CANDIDATE_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source-limit", type=int, default=0, help="API phases only; 0 uses all prepared sources")
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--rate-limit-initial-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--rate-limit-max-sleep-seconds", type=float, default=600.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print one API prompt without making calls")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model:
        args.generator_model = args.model
        args.verifier_model = args.model
    if args.questions_per_source != QUESTIONS_PER_SOURCE:
        raise ValueError("This rubric requires exactly two questions per eligible source")
    if not 0 < args.validation_source_count < args.source_count:
        raise ValueError("validation-source-count must be between 1 and source-count-1")
    if args.source_count % 2 or args.validation_source_count % 2:
        raise ValueError("source and validation counts must be even to balance evidence arms")
    if args.source_limit < 0:
        raise ValueError("source-limit must be non-negative")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    summaries = []
    if args.phase in {"prepare", "all"}:
        summaries.append(prepare_manifest(args))
    if args.phase in {"generate", "all"}:
        summaries.append(run_generation(args, dry_run=args.dry_run))
    if args.phase in {"verify", "all"}:
        summaries.append(run_verification(args, dry_run=args.dry_run))
    if args.phase in {"finalize", "all"} and not args.dry_run:
        summaries.append(finalize(args))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
