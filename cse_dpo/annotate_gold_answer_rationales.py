#!/usr/bin/env python
"""Annotate evidence-grounded rationales for the 1,130 supported gold answers.

The script does not construct DPO pairs.  It creates a reusable positive-rationale
bank with one canonical, snippet-supported gold answer per training question.  A
later pass can join these records to C1 candidates and annotate rejected-answer
rationales.

The workflow is resumable.  Every valid GPT response is cached by the complete
question, answer, evidence, model, and rubric.  Re-running the same command reuses
the cache and makes no duplicate API call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cse_dpo.candidate_bank_class_judge import (
    candidate_is_extractive,
    extract_snippets,
    extractive_evidence_ids,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "evidence_grounded_single_answer_qwen25_05b/train_prepared.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/gold_answer_rationales/"
    "gpt41mini_gold_supported_1130_v2"
)
DEFAULT_API_KEY_FILE = ROOT / "open_ai_api.txt"

RUBRIC_VERSION = "gold-answer-grounded-rationale-v2"
VALID_STATUSES = {"accepted", "review"}
VALID_ANSWER_TYPES = {
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
    "relation_or_association",
    "number_or_quantity",
    "percentage_or_rate",
    "date_or_time",
    "method_resource_or_tool",
    "physical_property",
    "structure_or_composition",
    "other",
}

SYSTEM_PROMPT = r"""
You are a biomedical evidence-rationale annotator. Return JSON only.

Your task is to explain why a supplied accepted factoid answer answers the exact
question, using only the supplied PubMed snippets. The accepted answer has already
been selected; do not replace, normalize, expand, shorten, or question it.

For an accepted annotation, write a detailed but concise two-sentence reason:
1. The first sentence must cite the supporting snippet ID or IDs and state the
   specific evidence claim that supports the answer.
2. The second sentence must identify what the question asks for and explain why the
   supplied answer fills that requested entity, relation, value, population, scope,
   or other answer slot.

Grounding rules:
- Use only the supplied snippets. Do not add external biomedical knowledge.
- Select exactly one strongest supporting snippet whenever possible. Use two only
  when one snippet is insufficient, and never cite more than two.
- Cite only supplied snippet IDs. The first sentence of the reason must begin with
  "Snippet 1.1 states ..." or "Snippets 1.1 and 2.1 state ...", and it must
  literally name every ID listed in evidence_ids.
- At least one cited snippet must contain the supplied answer as a continuous span,
  allowing only harmless case and typography differences.
- State the concrete relation expressed by the evidence. Avoid vague claims such as
  "the answer is discussed", "it is the main entity", or "it is relevant".
- Explain why the answer matches the question. Do not merely repeat the answer.
- Do not use preference labels or dataset language such as gold, chosen, preferred,
  rejected, positive, negative, correct answer, or incorrect answer.
- Keep the reason between 25 and 100 words.

Use status="review" when the snippets do not clearly support the supplied answer or
do not establish why it answers the question. For review records, briefly explain
the problem in the reason and do not invent support.

Output exactly these six keys:
- status: accepted or review
- requested_answer_type: one of the allowed values supplied in the user prompt
- evidence_ids: a JSON list of supplied snippet IDs
- evidence_claim: one concise sentence paraphrasing the decisive evidence
- reason: the two-sentence grounded rationale for accepted records, or the review
  explanation for review records
- basis: one short sentence explaining why the annotation was accepted or sent for
  review

Example:
Question: Which bacteria was EcoRI isolated from?
Accepted answer: Escherichia coli
Snippet 1.1: The restriction endonuclease EcoRI was isolated from E. coli.

Output:
{"status":"accepted","requested_answer_type":"organism","evidence_ids":["1.1"],"evidence_claim":"EcoRI was isolated from E. coli.","reason":"Snippet 1.1 states that the restriction endonuclease EcoRI was isolated from E. coli. Because the question asks for the source bacterium from which EcoRI was isolated, Escherichia coli fills the requested organism slot.","basis":"The cited snippet directly states the source organism and matches the question relation."}
""".strip()


def digest(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_source(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list or an object containing questions")
    factoid_rows = [row for row in rows if isinstance(row, dict) and row.get("type") == "factoid"]
    question_ids = [str(row.get("id") or "").strip() for row in factoid_rows]
    if any(not question_id for question_id in question_ids):
        raise ValueError(f"{path}: at least one factoid row has no id")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"{path}: factoid question IDs are not unique")
    return factoid_rows


def parse_answer_aliases(value: Any) -> list[str]:
    value = str(value or "").strip()
    pattern = re.compile(r"\[BE\]\s*(.*?)\s*\[EE\]", flags=re.I | re.S)
    aliases = [match.group(1).strip() for match in pattern.finditer(value)]
    remainder = pattern.sub("", value).strip()
    if not aliases or remainder:
        raise ValueError(f"Expected one or more [BE] answer [EE] outputs, got {value!r}")
    if any(not alias for alias in aliases):
        raise ValueError("Gold answer alias is empty")
    return list(dict.fromkeys(aliases))


def resources_from_row(row: dict[str, Any]) -> list[str]:
    indexed_resources: list[tuple[int, str]] = []
    for key, value in row.items():
        match = re.fullmatch(r"input_(\d+)", str(key))
        if not match or int(match.group(1)) < 2:
            continue
        text = str(value or "").strip()
        if text:
            indexed_resources.append((int(match.group(1)), text))
    return [text for _, text in sorted(indexed_resources)]


def prepare_record(row: dict[str, Any]) -> dict[str, Any]:
    accepted_aliases = parse_answer_aliases(row.get("output"))
    snippets = extract_snippets(resources_from_row(row))
    if not snippets:
        raise ValueError(f"{row.get('id')}: no snippets")
    supported = [
        (alias, extractive_evidence_ids(alias, snippets))
        for alias in accepted_aliases
        if extractive_evidence_ids(alias, snippets)
    ]
    if not supported:
        raise ValueError(
            f"{row.get('id')}: none of the accepted aliases is extractive from the snippets"
        )
    answer, evidence_ids = supported[0]
    return {
        "question_id": str(row["id"]),
        "question": str(row.get("input_1") or "").strip(),
        "chosen_answer": answer,
        "chosen_output": f"[BE]{answer}[EE]",
        "accepted_aliases": accepted_aliases,
        "snippets": snippets,
        "deterministic_support_ids": evidence_ids,
        "source_path": str(DEFAULT_SOURCE),
    }


def sentence_like_count(value: str) -> int:
    # This is deliberately permissive around biomedical abbreviations and decimal
    # values.  The prompt asks for two sentences; semantic validation remains with
    # the judge and manual audit.
    return len(re.findall(r"[.!?](?:\s|$)", value.strip()))


def validate_annotation(value: Any, record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("annotation is not a JSON object")
    expected = {
        "status",
        "requested_answer_type",
        "evidence_ids",
        "evidence_claim",
        "reason",
        "basis",
    }
    if set(value) != expected:
        raise ValueError(f"response keys must be exactly {sorted(expected)}")
    if value.get("status") not in VALID_STATUSES:
        raise ValueError("status must be accepted or review")
    if value.get("requested_answer_type") not in VALID_ANSWER_TYPES:
        raise ValueError("invalid requested_answer_type")
    if not isinstance(value.get("evidence_ids"), list):
        raise ValueError("evidence_ids must be a list")
    valid_ids = {snippet["snippet_id"] for snippet in record["snippets"]}
    if any(str(item) not in valid_ids for item in value["evidence_ids"]):
        raise ValueError("evidence_ids contains an ID not supplied in the prompt")
    for field in ("evidence_claim", "reason", "basis"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"{field} must be a non-empty string")

    if value["status"] == "accepted":
        if not value["evidence_ids"]:
            raise ValueError("accepted annotation must cite at least one snippet")
        if len(value["evidence_ids"]) > 2:
            raise ValueError("accepted annotation must cite no more than two snippets")
        cited = [
            snippet
            for snippet in record["snippets"]
            if snippet["snippet_id"] in {str(item) for item in value["evidence_ids"]}
        ]
        if not candidate_is_extractive(record["chosen_answer"], cited):
            raise ValueError(
                "no cited snippet contains the accepted answer; cite only one or two "
                f"of these exact-span support IDs: {record['deterministic_support_ids']}"
            )
        reason_words = value["reason"].split()
        if not 25 <= len(reason_words) <= 100:
            raise ValueError("accepted reason must contain 25 to 100 words")
        if sentence_like_count(value["reason"]) < 2:
            raise ValueError("accepted reason must contain two sentences")
        reason_casefold = value["reason"].casefold()
        missing_mentions = [
            evidence_id
            for evidence_id in value["evidence_ids"]
            if str(evidence_id).casefold() not in reason_casefold
        ]
        if missing_mentions:
            raise ValueError(f"reason does not name cited snippets: {missing_mentions}")
        if "snippet" not in reason_casefold:
            raise ValueError("accepted reason must explicitly refer to its snippet evidence")

    banned = re.compile(
        r"\b(gold|chosen|preferred|rejected|positive answer|negative answer|incorrect answer)\b",
        flags=re.I,
    )
    if banned.search(value["reason"]):
        raise ValueError("reason contains preference or dataset-label language")
    return value


def build_user_prompt(record: dict[str, Any], feedback: str = "") -> str:
    snippets = "\n".join(
        f"Snippet {snippet['snippet_id']} (PubMed {snippet['pubmed_id']}): "
        f"{snippet['text']}"
        for snippet in record["snippets"]
    )
    answer_types = ", ".join(sorted(VALID_ANSWER_TYPES))
    eligible_support_ids = ", ".join(record["deterministic_support_ids"])
    return f"""Question: {record['question']}

Accepted answer: {record['chosen_answer']}

Supplied snippets:
{snippets}

Return exactly one JSON object with the six required keys.
Allowed requested_answer_type values: {answer_types}.
Exact-span support IDs: {eligible_support_ids}.
Cite only IDs from this exact-span support list.
Choose exactly one strongest supporting snippet unless two are genuinely needed.
Never cite more than two snippets. The first sentence of reason must start with
"Snippet <ID> states ..." or "Snippets <ID> and <ID> state ...", and must contain
every literal ID listed in evidence_ids. The second sentence must explain what the
question asks and why the supplied answer fills that exact slot.
{feedback}"""


class GoldRationaleAnnotator:
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
                "question_id": record["question_id"],
                "question": record["question"],
                "chosen_answer": record["chosen_answer"],
                "accepted_aliases": record["accepted_aliases"],
                "snippets": record["snippets"],
                "model": self.args.model,
                "rubric_version": RUBRIC_VERSION,
                "system_prompt": SYSTEM_PROMPT,
            }
        )

    def call_api(self, prompt: str, api_key: str) -> dict[str, Any]:
        import requests

        payload = {
            "model": self.args.model,
            "temperature": 0,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(
            self.args.endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
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
            for snippet in record["snippets"]
            if snippet["snippet_id"] in wanted
        ]

    def annotate_one(self, record: dict[str, Any], api_key: str) -> dict[str, Any]:
        key = self.cache_key(record)
        cache_path = self.cache_dir / f"{key}.json"
        base = {
            "question_id": record["question_id"],
            "question": record["question"],
            "chosen_answer": record["chosen_answer"],
            "chosen_output": record["chosen_output"],
            "accepted_aliases": record["accepted_aliases"],
            "deterministic_support_ids": record["deterministic_support_ids"],
            "source_path": record["source_path"],
            "judge_model": self.args.model,
            "rubric_version": RUBRIC_VERSION,
        }
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            validate_annotation(cached, record)
            return {
                **base,
                **cached,
                "supporting_snippets": self.cited_snippets(
                    record, cached["evidence_ids"]
                ),
                "origin": "cache",
            }

        if self.args.max_new_calls is not None and self.new_api_calls >= self.args.max_new_calls:
            return {
                **base,
                "status": "deferred",
                "requested_answer_type": None,
                "evidence_ids": [],
                "supporting_snippets": [],
                "evidence_claim": "",
                "reason": "Deferred because max_new_calls was reached.",
                "basis": "No API call was made.",
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
                            f"Rate limit persisted for {record['question_id']}: {exc}"
                        ) from exc
                    sleep_seconds = min(
                        self.args.rate_limit_max_sleep_seconds,
                        self.args.rate_limit_initial_sleep_seconds * (2**rate_limit_attempt),
                    )
                    sleep_seconds += random.uniform(0, min(5.0, sleep_seconds * 0.1))
                    print(
                        f"Rate limited on {record['question_id']}; sleeping "
                        f"{sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)
                    rate_limit_attempt += 1
                    continue
                validation_attempt += 1
                feedback = (
                    f"Previous response failed validation: {exc}. Return a corrected complete "
                    "six-key JSON object. Preserve the supplied answer and cite only supplied "
                    "snippet IDs."
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
            "requested_answer_type": "other",
            "evidence_ids": [],
            "supporting_snippets": [],
            "evidence_claim": "",
            "reason": f"Automatic annotation failed validation: {last_error}",
            "basis": "Manual review is required.",
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
            if index == 1 or index % self.args.progress_every == 0 or index == len(self.records):
                print(
                    f"{index} / {len(self.records)} | new API calls: {self.new_api_calls} "
                    f"| rate limits: {self.rate_limit_hits}"
                )
        return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "question",
        "chosen_answer",
        "chosen_output",
        "accepted_aliases",
        "status",
        "requested_answer_type",
        "evidence_ids",
        "supporting_snippets",
        "deterministic_support_ids",
        "evidence_claim",
        "reason",
        "basis",
        "origin",
        "judge_model",
        "rubric_version",
        "source_path",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument(
        "--question-limit",
        type=int,
        default=0,
        help="0 annotates every question after question-offset.",
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
        help="Validate inputs and print example prompts without calling the API or writing outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.question_offset < 0 or args.question_limit < 0:
        raise ValueError("question-offset and question-limit must be non-negative")
    if args.max_new_calls is not None and args.max_new_calls < 0:
        raise ValueError("max-new-calls must be non-negative")
    rows = read_source(args.source)
    if args.question_offset == 0 and args.question_limit == 0 and len(rows) != 1130:
        raise ValueError(f"Expected the full supported source to contain 1,130 rows, got {len(rows)}")
    stop = None if args.question_limit == 0 else args.question_offset + args.question_limit
    selected = rows[args.question_offset:stop]
    records = [prepare_record(row) for row in selected]
    for record in records:
        record["source_path"] = str(args.source)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "source_question_count": len(rows),
                "selected_question_count": len(records),
                "question_offset": args.question_offset,
                "question_limit": args.question_limit,
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

    annotator = GoldRationaleAnnotator(records, args)
    annotations = annotator.run()
    output_root = Path(args.output_root)
    write_jsonl(output_root / "gold_answer_rationales.jsonl", annotations)
    write_csv(output_root / "gold_answer_rationales.csv", annotations)
    review = [row for row in annotations if row.get("status") != "accepted"]
    write_jsonl(output_root / "gold_answer_rationales_review.jsonl", review)
    summary = {
        "status": "complete" if not review else "needs_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_question_count": len(rows),
        "selected_question_count": len(records),
        "annotation_count": len(annotations),
        "status_counts": dict(Counter(row.get("status") for row in annotations)),
        "answer_type_counts": dict(
            Counter(row.get("requested_answer_type") for row in annotations)
        ),
        "origin_counts": dict(Counter(row.get("origin") for row in annotations)),
        "new_api_calls": annotator.new_api_calls,
        "rate_limit_hits": annotator.rate_limit_hits,
        "judge_model": args.model,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": digest(SYSTEM_PROMPT),
        "annotations_jsonl": str(output_root / "gold_answer_rationales.jsonl"),
        "annotations_csv": str(output_root / "gold_answer_rationales.csv"),
        "review_jsonl": str(output_root / "gold_answer_rationales_review.jsonl"),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
