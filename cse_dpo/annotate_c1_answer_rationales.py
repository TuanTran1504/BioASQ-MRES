#!/usr/bin/env python
"""Annotate grounded mistaken rationales for strict-equivalence Stage-1 C1 answers.

The script joins the current Stage-1 C3>C1 pairs to the accepted gold-answer
rationale bank. GPT-4.1-mini writes a plausible but flawed rationale for each C1
answer: its evidence sentence must remain faithful to a supplied snippet, while
its selection sentence must commit the labeled C1 error.

Valid responses are cached. The script also writes matched reason-first
positive/negative records, but it does not alter a training notebook or start
training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cse_dpo.annotate_gold_answer_rationales import (
    read_source,
    resources_from_row,
)
from cse_dpo.candidate_bank_class_judge import (
    candidate_is_extractive,
    extract_snippets,
    extractive_evidence_ids,
)


ROOT = Path(__file__).resolve().parents[1]
STRICT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/class_judgments/"
    "candidate_banks_train_full_fresh_c3_c1_v2_strict_equivalence_gold_c3_gold_supported_1130/"
    "merged"
)
DEFAULT_POSITIVE_RATIONALES = (
    ROOT
    / "Artifacts/cse_dpo/gold_answer_rationales/"
    "gpt41mini_gold_supported_1130_v2/gold_answer_rationales.jsonl"
)
DEFAULT_STAGE1_PAIRS = (
    STRICT_ROOT
    / "staged_curriculum_pairs/dpo_stage1_concept_learning_all_pairs.jsonl"
)
DEFAULT_JUDGMENTS = STRICT_ROOT / "candidate_class_judgments.jsonl"
DEFAULT_PREPARED_SOURCE = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "evidence_grounded_single_answer_qwen25_05b/train_prepared.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/c1_answer_rationales/"
    "gpt41mini_strict_equivalence_v2_positive1111_rubric_v3"
)
DEFAULT_API_KEY_FILE = ROOT / "open_ai_api.txt"

RUBRIC_VERSION = "c1-grounded-mistaken-rationale-v3"
VALID_STATUSES = {"accepted", "review"}

SYSTEM_PROMPT = r"""
You are constructing rejected rationales for biomedical factoid preference
training. Return JSON only.

You receive:
- a factoid question;
- a rejected C1 answer;
- the C1 relation and error labels assigned by a previous strict-equivalence judge;
- PubMed snippets.

Write a plausible rationale that could lead a model to select the supplied C1
answer. The rationale must expose the selection behavior represented by the error
label without inventing a false statement about the snippet.

For status="accepted", reason must contain exactly this logical structure:
1. Evidence sentence: cite one strongest supplied snippet, or two only when
   necessary, and state accurately what that evidence says about the C1 answer or
   the nearby concept.
2. Mistaken-selection sentence: map that evidence to the question's requested
   slot in the specific flawed way indicated by relation_type and error_type.

The evidence statement must be true to the supplied text. The inference may
select the wrong entity, relation, scope, qualifier, value, population, entity
type, part, or explanation. The reason must sound like a model sincerely
justifying its answer. It must never diagnose, reveal, acknowledge, or correct its
own flaw. Put all audit discussion of the supplied labels only in basis.

Rules:
- Use only supplied snippets and cite literal IDs as "Snippet 1.1" or
  "Snippets 1.1 and 2.1".
- The reason must literally name every ID in evidence_ids.
- Preserve the supplied C1 answer. Do not replace or normalize it.
- Mention the supplied C1 answer in the reason.
- Do not explicitly contrast it with an accepted answer.
- Do not mention an accepted alias in reason unless that complete alias is a
  continuous part of the supplied C1 answer.
- Do not use annotation or self-correction language in reason, including: C1,
  gold answer, accepted answer, chosen answer, rejected answer, negative answer,
  error, mistake, mistaken, wrong, incorrect, broader, narrower, omit, missing,
  extra qualifier, confusion, instead of, rather than, without, solely, or only.
- The words "gold standard", "preferred treatment", and similar biomedical claims
  are allowed when supported by the snippet.
- Keep the reason between 25 and 110 words.
- If no grounded, plausible rationale can be written without fabricating evidence,
  return status="review".

Output exactly these five keys:
- status: accepted or review
- evidence_ids: JSON list of supplied snippet IDs
- evidence_claim: a concise accurate statement of what the cited evidence says
- reason: the two-sentence rationale, or a concise explanation for review
- basis: an audit note explaining how the rationale embodies the supplied error
  label, or why it needs review

Example 1 — wrong answer type:
Question: Which gene is affected by the reported mutation?
C1 answer: c.436delC
relation_type: wrong_entity
error_type: wrong_answer_type
Snippet 1.1: A homozygous c.436delC variant was identified in DCAF17.

Output:
{"status":"accepted","evidence_ids":["1.1"],"evidence_claim":"Snippet 1.1 identifies c.436delC as the reported variant.","reason":"Snippet 1.1 states that c.436delC is the genetic variant identified in the report. Because the question requests the affected genetic entity, c.436delC is selected as that entity.","basis":"The reason accurately cites the variant and then commits the supplied answer-type/entity selection error."}

Example 2 — broader answer, with no self-diagnosis:
Question: Orteronel was developed for treatment of which cancer?
C1 answer: prostate cancer
relation_type: broader
error_type: broader
Snippet 5.1: Orteronel for the treatment of prostate cancer.

Output:
{"status":"accepted","evidence_ids":["5.1"],"evidence_claim":"Snippet 5.1 describes orteronel as a treatment for prostate cancer.","reason":"Snippet 5.1 describes orteronel as a treatment developed for prostate cancer. The question asks which cancer orteronel was developed to treat, so prostate cancer is selected as the disease indication.","basis":"The reason sincerely selects the supplied broader disease term; the broader-label diagnosis appears only in this audit field."}

Example 3 — missing qualifier, with no accepted-answer quotation:
Question: What is the cause of the syndrome?
C1 answer: deficiency of paternally expressed genes
relation_type: missing_qualifier
error_type: missing_qualifier
Snippet 1.1: The syndrome is caused by deficiency of paternally expressed genes in chromosome 15q11-q13.

Output:
{"status":"accepted","evidence_ids":["1.1"],"evidence_claim":"Snippet 1.1 attributes the syndrome to deficient paternal gene expression at a specified chromosomal region.","reason":"Snippet 1.1 attributes the syndrome to a deficiency of paternally expressed genes at a defined chromosomal region. The question asks for the cause, so deficiency of paternally expressed genes is selected as the causal mechanism.","basis":"The reason selects the supplied expression while leaving the missing-location diagnosis exclusively in this audit field."}
""".strip()


def digest(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def prepare_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive_rows = read_jsonl(args.positive_rationales)
    positives = {
        str(row["question_id"]): row
        for row in positive_rows
        if row.get("status") == "accepted"
    }

    source_rows = {str(row["id"]): row for row in read_source(args.prepared_source)}
    judgment_rows = read_jsonl(args.judgments)
    judgments = {
        (str(row["response_id"]), str(row.get("source_model") or "")): row
        for row in judgment_rows
    }
    stage1_pairs = read_jsonl(args.stage1_pairs)

    records: list[dict[str, Any]] = []
    missing_positive = 0
    nonextractive_c1_skipped = 0
    duplicate_pairs = 0
    seen: set[tuple[str, str]] = set()
    per_question: Counter[str] = Counter()

    for pair in stage1_pairs:
        question_id = str(pair["question_id"])
        positive = positives.get(question_id)
        if positive is None:
            missing_positive += 1
            continue

        candidate = str(pair["rejected_candidate"]).strip()
        dedupe_key = (question_id, normalize_text(candidate))
        if dedupe_key in seen:
            duplicate_pairs += 1
            continue
        if args.max_pairs_per_question and per_question[question_id] >= args.max_pairs_per_question:
            continue

        judgment_key = (
            str(pair["rejected_response_id"]),
            str(pair.get("rejected_source_model") or ""),
        )
        judgment = judgments.get(judgment_key)
        if judgment is None:
            raise ValueError(f"{pair['pair_id']}: rejected judgment was not found")
        if judgment.get("class") != "C1":
            raise ValueError(
                f"{pair['pair_id']}: rejected judgment is {judgment.get('class')}, not C1"
            )
        if normalize_text(judgment.get("candidate")) != normalize_text(candidate):
            raise ValueError(f"{pair['pair_id']}: rejected candidate/judgment mismatch")

        source = source_rows.get(question_id)
        if source is None:
            raise ValueError(f"{pair['pair_id']}: prepared source question was not found")
        snippets = extract_snippets(resources_from_row(source))
        if not snippets:
            raise ValueError(f"{pair['pair_id']}: no source snippets")

        valid_ids = {str(snippet["snippet_id"]) for snippet in snippets}
        judged_ids = [
            str(value)
            for value in judgment.get("evidence_ids", [])
            if str(value) in valid_ids
        ]
        extractive_ids = extractive_evidence_ids(candidate, snippets)
        if not extractive_ids and not args.include_nonextractive_c1:
            nonextractive_c1_skipped += 1
            continue
        positive_ids = [
            str(value)
            for value in positive.get("evidence_ids", [])
            if str(value) in valid_ids
        ]
        if extractive_ids:
            accepted_aliases = list(positive.get("accepted_aliases", []))
            candidate_only_ids = [
                snippet_id
                for snippet_id in extractive_ids
                if not any(
                    contains_token_phrase(
                        next(
                            snippet["text"]
                            for snippet in snippets
                            if snippet["snippet_id"] == snippet_id
                        ),
                        alias,
                    )
                    and not contains_token_phrase(candidate, alias)
                    for alias in accepted_aliases
                )
            ]
            ranked_ids = list(dict.fromkeys(candidate_only_ids + extractive_ids))
        else:
            ranked_ids = list(dict.fromkeys(judged_ids + positive_ids))
        if ranked_ids:
            prompt_ids = set(ranked_ids[: min(args.max_prompt_snippets, 4)])
            prompt_snippets = [
                snippet for snippet in snippets if snippet["snippet_id"] in prompt_ids
            ]
        else:
            prompt_snippets = snippets[: args.max_prompt_snippets]

        records.append(
            {
                "pair_id": str(pair["pair_id"]),
                "question_id": question_id,
                "question": str(pair["question"]),
                "source_prompt": str(pair["prompt"]),
                "positive_answer": str(positive["chosen_answer"]),
                "positive_output": str(positive["chosen_output"]),
                "positive_reason": str(positive["reason"]),
                "positive_evidence_ids": list(positive.get("evidence_ids", [])),
                "accepted_aliases": list(positive.get("accepted_aliases", [])),
                "c1_answer": candidate,
                "c1_output": str(pair["rejected"]),
                "relation_type": str(judgment.get("relation_type") or ""),
                "error_type": str(judgment.get("error_type") or ""),
                "c1_basis": str(judgment.get("basis") or ""),
                "c1_evidence_support": str(judgment.get("evidence_support") or ""),
                "c1_judged_evidence_ids": judged_ids,
                "c1_extractive_evidence_ids": extractive_ids,
                "c1_source_model": str(pair.get("rejected_source_model") or ""),
                "c1_response_id": str(pair["rejected_response_id"]),
                "snippets": snippets,
                "prompt_snippets": prompt_snippets,
            }
        )
        seen.add(dedupe_key)
        per_question[question_id] += 1

    if args.pair_offset:
        records = records[args.pair_offset :]
    if args.pair_limit:
        records = records[: args.pair_limit]

    preparation_summary = {
        "positive_rows": len(positive_rows),
        "accepted_positive_questions": len(positives),
        "stage1_source_pairs": len(stage1_pairs),
        "eligible_unique_pairs_before_slice": len(seen),
        "selected_pairs": len(records),
        "selected_questions": len({record["question_id"] for record in records}),
        "stage1_pairs_without_accepted_positive": missing_positive,
        "nonextractive_c1_pairs_skipped": nonextractive_c1_skipped,
        "include_nonextractive_c1": args.include_nonextractive_c1,
        "duplicate_stage1_pairs_removed": duplicate_pairs,
        "pairs_per_question": dict(Counter(per_question.values())),
        "c1_evidence_support_counts": dict(
            Counter(record["c1_evidence_support"] for record in records)
        ),
        "c1_error_type_counts": dict(Counter(record["error_type"] for record in records)),
        "c1_relation_type_counts": dict(
            Counter(record["relation_type"] for record in records)
        ),
    }
    return records, preparation_summary


def sentence_like_count(value: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", value.strip()))


def phrase_tokens(value: Any) -> list[str]:
    return re.findall(r"[\w]+(?:[-'][\w]+)*", str(value or "").casefold())


def contains_token_phrase(haystack: Any, needle: Any) -> bool:
    haystack_tokens = phrase_tokens(haystack)
    needle_tokens = phrase_tokens(needle)
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    width = len(needle_tokens)
    return any(
        haystack_tokens[index : index + width] == needle_tokens
        for index in range(len(haystack_tokens) - width + 1)
    )


def candidate_token_coverage(candidate: str, reason: str) -> float:
    candidate_tokens = set(phrase_tokens(candidate))
    reason_tokens = set(phrase_tokens(reason))
    if not candidate_tokens:
        return 0.0
    return len(candidate_tokens & reason_tokens) / len(candidate_tokens)


def validate_annotation(value: Any, record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("annotation is not a JSON object")
    expected = {"status", "evidence_ids", "evidence_claim", "reason", "basis"}
    if set(value) != expected:
        raise ValueError(f"response keys must be exactly {sorted(expected)}")
    if value.get("status") not in VALID_STATUSES:
        raise ValueError("status must be accepted or review")
    if not isinstance(value.get("evidence_ids"), list):
        raise ValueError("evidence_ids must be a list")
    valid_ids = {str(snippet["snippet_id"]) for snippet in record["prompt_snippets"]}
    if any(str(item) not in valid_ids for item in value["evidence_ids"]):
        raise ValueError("evidence_ids contains an ID not supplied in the prompt")
    for field in ("evidence_claim", "reason", "basis"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"{field} must be a non-empty string")

    if value["status"] == "accepted":
        if not value["evidence_ids"]:
            raise ValueError("accepted annotation must cite at least one snippet")
        word_count = len(value["reason"].split())
        if not 25 <= word_count <= 110:
            raise ValueError("accepted reason must contain 25 to 110 words")
        if sentence_like_count(value["reason"]) < 2:
            raise ValueError("accepted reason must contain at least two sentences")
        reason_casefold = value["reason"].casefold()
        missing_ids = [
            str(evidence_id)
            for evidence_id in value["evidence_ids"]
            if str(evidence_id).casefold() not in reason_casefold
        ]
        if missing_ids:
            raise ValueError(f"reason does not name cited snippets: {missing_ids}")
        candidate_tokens = phrase_tokens(record["c1_answer"])
        exact_candidate_mention = contains_token_phrase(
            value["reason"], record["c1_answer"]
        )
        if len(candidate_tokens) < 8 and not exact_candidate_mention:
            raise ValueError("reason does not explicitly mention the supplied C1 answer")
        if len(candidate_tokens) >= 8 and candidate_token_coverage(
            record["c1_answer"], value["reason"]
        ) < 0.60:
            raise ValueError("reason does not sufficiently identify the long C1 answer")

        cited_ids = {str(item) for item in value["evidence_ids"]}
        extractive_ids = {str(item) for item in record["c1_extractive_evidence_ids"]}
        if extractive_ids and not (cited_ids & extractive_ids):
            raise ValueError(
                "reason must cite a supplied snippet that contains the C1 answer"
            )

        for alias in record["accepted_aliases"]:
            if (
                len(phrase_tokens(alias)) > 0
                and contains_token_phrase(value["reason"], alias)
                and not contains_token_phrase(record["c1_answer"], alias)
            ):
                raise ValueError(
                    f"reason introduces accepted alias outside the C1 answer: {alias!r}"
                )

        banned = re.compile(
            r"\b(C1|gold answer|accepted answer|chosen answer|rejected answer|"
            r"negative answer|incorrect|wrong|mistak\w*|error|broader|narrower|"
            r"omit\w*|missing|extra qualifier|confus\w*|instead of|rather than|"
            r"without|solely|only)\b",
            flags=re.I,
        )
        if banned.search(value["reason"]):
            raise ValueError("reason contains annotation or self-correction language")

    return value


def build_user_prompt(record: dict[str, Any], feedback: str = "") -> str:
    snippets = "\n".join(
        f"Snippet {snippet['snippet_id']} (PubMed {snippet['pubmed_id']}): "
        f"{snippet['text']}"
        for snippet in record["prompt_snippets"]
    )
    return f"""Question: {record['question']}

Supplied C1 answer: {record['c1_answer']}
relation_type: {record['relation_type']}
error_type: {record['error_type']}

Supplied snippets:
{snippets}

Return exactly one JSON object with the five required keys. Use one strongest
snippet when possible. Keep its evidence claim accurate, then make the
mistaken-selection sentence embody the supplied relation_type/error_type without
diagnosing or correcting itself. Treat relation_type and error_type as private
guidance: never name or explain them in reason. The reason must sound sincerely
confident. Terms such as broader, narrower, omits, missing, mistaken, confusion,
rather than, instead of, without, solely, and only are forbidden in reason. Put
all diagnosis only in basis.
{feedback}"""


class C1RationaleAnnotator:
    def __init__(self, records: list[dict[str, Any]], args: argparse.Namespace):
        self.records = records
        self.args = args
        self.output_root = Path(args.output_root)
        self.cache_dir = self.output_root / "cache"
        self.error_dir = self.output_root / "errors"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)
        self.new_api_calls = 0
        self.rate_limit_hits = 0

    def cache_key(self, record: dict[str, Any]) -> str:
        return digest(
            {
                "pair_id": record["pair_id"],
                "question": record["question"],
                "c1_answer": record["c1_answer"],
                "relation_type": record["relation_type"],
                "error_type": record["error_type"],
                "c1_basis": record["c1_basis"],
                "prompt_snippets": record["prompt_snippets"],
                "model": self.args.model,
                "rubric_version": RUBRIC_VERSION,
                "system_prompt": SYSTEM_PROMPT,
            }
        )

    def call_api(self, prompt: str, api_key: str) -> dict[str, Any]:
        import requests

        response = requests.post(
            self.args.endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.args.model,
                "temperature": 0,
                "max_tokens": 700,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.args.timeout_seconds,
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

    @staticmethod
    def is_rate_limit_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        return "429" in str(exc)

    @staticmethod
    def cited_snippets(
        record: dict[str, Any], evidence_ids: list[Any]
    ) -> list[dict[str, Any]]:
        wanted = {str(item) for item in evidence_ids}
        return [
            dict(snippet)
            for snippet in record["prompt_snippets"]
            if snippet["snippet_id"] in wanted
        ]

    def base_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: record[key]
            for key in (
                "pair_id",
                "question_id",
                "question",
                "positive_answer",
                "positive_output",
                "positive_reason",
                "positive_evidence_ids",
                "accepted_aliases",
                "c1_answer",
                "c1_output",
                "relation_type",
                "error_type",
                "c1_basis",
                "c1_evidence_support",
                "c1_judged_evidence_ids",
                "c1_extractive_evidence_ids",
                "c1_source_model",
                "c1_response_id",
            )
        } | {
            "judge_model": self.args.model,
            "rubric_version": RUBRIC_VERSION,
        }

    def annotate_one(self, record: dict[str, Any], api_key: str) -> dict[str, Any]:
        key = self.cache_key(record)
        cache_path = self.cache_dir / f"{key}.json"
        base = self.base_record(record)

        if cache_path.exists():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            validate_annotation(value, record)
            return {
                **base,
                **value,
                "supporting_snippets": self.cited_snippets(
                    record, value["evidence_ids"]
                ),
                "origin": "cache",
            }

        if self.args.max_new_calls is not None and self.new_api_calls >= self.args.max_new_calls:
            return {
                **base,
                "status": "deferred",
                "evidence_ids": [],
                "evidence_claim": "",
                "reason": "Deferred because max_new_calls was reached.",
                "basis": "No API call was made.",
                "supporting_snippets": [],
                "origin": "deferred",
            }

        feedback = ""
        validation_attempt = 0
        rate_limit_attempt = 0
        last_error: Exception | None = None
        last_value: Any = None

        while validation_attempt <= self.args.max_retries:
            try:
                if self.args.request_delay_seconds > 0 and self.new_api_calls > 0:
                    time.sleep(self.args.request_delay_seconds)
                self.new_api_calls += 1
                value = self.call_api(build_user_prompt(record, feedback), api_key)
                last_value = value
                validate_annotation(value, record)
                cache_path.write_text(
                    json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return {
                    **base,
                    **value,
                    "supporting_snippets": self.cited_snippets(
                        record, value["evidence_ids"]
                    ),
                    "origin": "api",
                }
            except Exception as exc:
                last_error = exc
                if self.is_rate_limit_error(exc):
                    self.rate_limit_hits += 1
                    if rate_limit_attempt >= self.args.rate_limit_max_retries:
                        raise RuntimeError(
                            f"Rate limit persisted for {record['pair_id']}: {exc}"
                        ) from exc
                    sleep_seconds = min(
                        self.args.rate_limit_max_sleep_seconds,
                        self.args.rate_limit_initial_sleep_seconds
                        * (2**rate_limit_attempt),
                    )
                    sleep_seconds += random.uniform(0, min(5.0, sleep_seconds * 0.1))
                    print(
                        f"Rate limited on {record['pair_id']}; sleeping "
                        f"{sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)
                    rate_limit_attempt += 1
                    continue

                validation_attempt += 1
                feedback = (
                    f"Previous response failed validation: {exc}. Return a corrected "
                    "complete five-key JSON object. The reason must cite its IDs, mention "
                    "the supplied C1 answer, and must not admit or correct the selection error."
                )
                if validation_attempt <= self.args.max_retries:
                    time.sleep(2 ** (validation_attempt - 1))

        self.error_dir.joinpath(f"{key}.json").write_text(
            json.dumps(
                {
                    **base,
                    "error": str(last_error),
                    "last_invalid_response": last_value,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            **base,
            "status": "review",
            "evidence_ids": [],
            "evidence_claim": "",
            "reason": f"Automatic annotation failed validation: {last_error}",
            "basis": "Manual review is required.",
            "supporting_snippets": [],
            "origin": "error",
        }

    def run(self) -> list[dict[str, Any]]:
        key_file = Path(self.args.api_key_file)
        if not key_file.exists():
            raise FileNotFoundError(key_file)
        api_key = key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError(f"API key file is empty: {key_file}")

        output: list[dict[str, Any]] = []
        for index, record in enumerate(self.records, 1):
            output.append(self.annotate_one(record, api_key))
            if (
                index == 1
                or index % self.args.progress_every == 0
                or index == len(self.records)
            ):
                print(
                    f"{index} / {len(self.records)} | new API calls: "
                    f"{self.new_api_calls} | rate limits: {self.rate_limit_hits}"
                )
        return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "question_id",
        "question",
        "positive_answer",
        "positive_reason",
        "c1_answer",
        "relation_type",
        "error_type",
        "c1_evidence_support",
        "status",
        "evidence_ids",
        "evidence_claim",
        "reason",
        "basis",
        "supporting_snippets",
        "c1_source_model",
        "origin",
        "judge_model",
        "rubric_version",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field, "")
                    for field in fieldnames
                }
            )


def build_matched_pairs(
    annotations: list[dict[str, Any]], source_by_pair: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in annotations:
        if row.get("status") != "accepted":
            continue
        source = source_by_pair[row["pair_id"]]
        output.append(
            {
                "pair_id": row["pair_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "source_prompt": source["source_prompt"],
                "positive_answer": row["positive_answer"],
                "positive_reason": row["positive_reason"],
                "c1_answer": row["c1_answer"],
                "c1_reason": row["reason"],
                "relation_type": row["relation_type"],
                "error_type": row["error_type"],
                "c1_evidence_support": row["c1_evidence_support"],
                "c1_source_model": row["c1_source_model"],
                "chosen_completion": (
                    f"Reason: {row['positive_reason']}\n"
                    f"Answer: {row['positive_output']}"
                ),
                "rejected_completion": (
                    f"Reason: {row['reason']}\nAnswer: {row['c1_output']}"
                ),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-rationales", type=Path, default=DEFAULT_POSITIVE_RATIONALES
    )
    parser.add_argument("--stage1-pairs", type=Path, default=DEFAULT_STAGE1_PAIRS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--prepared-source", type=Path, default=DEFAULT_PREPARED_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument(
        "--endpoint", default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=0,
        help="0 annotates every eligible pair after pair-offset.",
    )
    parser.add_argument(
        "--max-pairs-per-question",
        type=int,
        default=0,
        help="0 keeps every unique Stage-1 C1 pair.",
    )
    parser.add_argument("--max-prompt-snippets", type=int, default=12)
    parser.add_argument(
        "--include-nonextractive-c1",
        action="store_true",
        help=(
            "Also annotate C1 answers absent from every snippet. By default these "
            "are excluded because a grounded mistaken rationale cannot support them."
        ),
    )
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--rate-limit-initial-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--rate-limit-max-sleep-seconds", type=float, default=600.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate joins and print prompts without calling the API or writing outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pair_offset < 0 or args.pair_limit < 0:
        raise ValueError("pair-offset and pair-limit must be non-negative")
    if args.max_pairs_per_question < 0 or args.max_prompt_snippets < 1:
        raise ValueError("invalid pair cap or prompt-snippet limit")
    if args.max_new_calls is not None and args.max_new_calls < 0:
        raise ValueError("max-new-calls must be non-negative")

    records, preparation_summary = prepare_records(args)
    print(
        json.dumps(
            {
                **preparation_summary,
                "pair_offset": args.pair_offset,
                "pair_limit": args.pair_limit,
                "max_pairs_per_question": args.max_pairs_per_question,
                "model": args.model,
                "output_root": str(args.output_root),
                "rubric_version": RUBRIC_VERSION,
                "rubric_sha256": digest(SYSTEM_PROMPT),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        for record in records[: min(2, len(records))]:
            print("\n" + "=" * 80)
            print(build_user_prompt(record))
        return

    annotator = C1RationaleAnnotator(records, args)
    annotations = annotator.run()
    output_root = Path(args.output_root)
    write_jsonl(output_root / "c1_answer_rationales.jsonl", annotations)
    write_csv(output_root / "c1_answer_rationales.csv", annotations)
    review = [row for row in annotations if row.get("status") != "accepted"]
    write_jsonl(output_root / "c1_answer_rationales_review.jsonl", review)

    source_by_pair = {record["pair_id"]: record for record in records}
    matched_pairs = build_matched_pairs(annotations, source_by_pair)
    write_jsonl(output_root / "matched_reason_first_pairs.jsonl", matched_pairs)

    summary = {
        "status": "complete" if not review else "needs_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **preparation_summary,
        "annotation_count": len(annotations),
        "matched_pair_count": len(matched_pairs),
        "status_counts": dict(Counter(row.get("status") for row in annotations)),
        "origin_counts": dict(Counter(row.get("origin") for row in annotations)),
        "accepted_error_type_counts": dict(
            Counter(
                row.get("error_type")
                for row in annotations
                if row.get("status") == "accepted"
            )
        ),
        "accepted_relation_type_counts": dict(
            Counter(
                row.get("relation_type")
                for row in annotations
                if row.get("status") == "accepted"
            )
        ),
        "new_api_calls": annotator.new_api_calls,
        "rate_limit_hits": annotator.rate_limit_hits,
        "judge_model": args.model,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": digest(SYSTEM_PROMPT),
        "annotations_jsonl": str(output_root / "c1_answer_rationales.jsonl"),
        "annotations_csv": str(output_root / "c1_answer_rationales.csv"),
        "review_jsonl": str(output_root / "c1_answer_rationales_review.jsonl"),
        "matched_pairs_jsonl": str(output_root / "matched_reason_first_pairs.jsonl"),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
