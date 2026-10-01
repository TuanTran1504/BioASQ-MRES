"""Class-judge strict-extractive candidate-bank answers and build preference pairs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any

PAIR_CLASS_RANK = {"C1": 1, "C2": 2, "C3": 3}
VALID_LLM_CLASSES = {"C2", "C1", "UNCERTAIN"}
LEGACY_CLASS_MAP = {"C0": "C1", "C1": "C1", "C2": "C2", "C3": "C3", "UNCERTAIN": "UNCERTAIN"}
VALID_SUPPORT = {"supported", "unsupported", "contradicted", "insufficient"}
JUDGE_RUBRIC_VERSION = "strict-equivalence-fewshot-v2.3-defined-fields-json"
EQUIVALENT_RELATION_TYPES = {
    "synonym",
    "abbreviation_expansion",
    "nomenclature_variant",
    "spelling_or_inflection",
    "numerically_equivalent",
    "harmless_formatting",
}
NON_EQUIVALENT_RELATION_TYPES = {
    "broader",
    "narrower",
    "part_whole",
    "wrong_entity",
    "wrong_relation",
    "wrong_value",
    "wrong_population",
    "missing_qualifier",
    "extra_qualifier",
    "unsupported_explanation",
    "non_answer",
    "other_non_equivalent",
}
VALID_RELATION_TYPES = EQUIVALENT_RELATION_TYPES | NON_EQUIVALENT_RELATION_TYPES | {"uncertain"}
VALID_ERROR_TYPES = NON_EQUIVALENT_RELATION_TYPES | {"none", "insufficient_information"}

JUDGE_SYSTEM = r"""
You are a biomedical factoid answer evaluator. Return JSON only.

Class definitions:
- C2: equivalent-correct. The candidate is not an exact accepted alias, but it denotes the same answer and can replace the gold answer without changing the truth of the answer. It must preserve the same biomedical entity, relation, value, population, scope, and every essential qualifier. Valid synonyms, abbreviation expansions, nomenclature variants, spelling/inflection variants, mathematically equivalent numerical forms, and harmless formatting variants belong here.
- C1: not equivalent. Use C1 whenever the candidate changes or omits any essential meaning. This includes related-but-different concepts, broader or narrower concepts, subtype/supertype changes, part-versus-whole answers, wrong member, wrong entity, wrong relation, wrong value, wrong population, missing qualifiers, extra unsupported qualifiers, explanations instead of the requested answer, contradictions, hallucinations, and non-answers.
- UNCERTAIN: the supplied information is insufficient for a confident C2/C1 decision.

Decision rule:
1. First decide whether the candidate and a gold alias are strictly substitutable as the answer to this exact question.
2. A candidate being true, mentioned, supported by a snippet, or medically related is not enough for C2.
3. If one answer is broader, narrower, a component, a container, a subtype, a supertype, or differs in an essential qualifier or value, label C1 even when both phrases occur in the snippets.
4. Numerical variants are C2 only when they express the same value or bound. For example, ">30" and "more than 30" are equivalent; "fewer than 100" and "at most several hundred" are not.
5. When genuinely uncertain about equivalence, use UNCERTAIN rather than stretching C2.

Output-field definitions:
- class: C2 for a strictly substitutable non-exact answer; C1 for a non-equivalent answer; UNCERTAIN when the supplied information cannot establish equivalence.
- semantic_correct: true for C2, false for C1, and null for UNCERTAIN.
- basis: one short sentence stating the decisive reason for the classification. Do not provide hidden chain-of-thought.
- evidence_support assesses the candidate against the supplied snippets independently of answer equivalence:
  - supported: at least one supplied snippet directly supports the candidate.
  - unsupported: no supplied snippet supports a substantive claim made by the candidate.
  - contradicted: at least one supplied snippet directly conflicts with the candidate.
  - insufficient: the snippets are too incomplete or ambiguous to assess support.
- evidence_ids: list only the IDs of supplied snippets that directly support or contradict the candidate. Use [] when no snippet can be cited. Never invent an ID.
- relation_type describes the candidate-to-gold relationship:
  - synonym: a substitutable synonym.
  - abbreviation_expansion: an abbreviation and its expansion.
  - nomenclature_variant: an alternative scientific or biomedical name for the same entity.
  - spelling_or_inflection: a harmless spelling or grammatical-number variation.
  - numerically_equivalent: the same numerical value, range, or bound.
  - harmless_formatting: a typography, spacing, or capitalization difference with unchanged meaning.
  - broader: the candidate covers a wider concept than the gold answer.
  - narrower: the candidate covers a more specific concept than the gold answer.
  - part_whole: the candidate gives a component for the whole, or the whole for a component.
  - wrong_entity: a different disease, drug, gene, protein, process, location, or other entity.
  - wrong_relation: relevant entities are connected by the wrong relationship.
  - wrong_value: an incorrect number, amount, date, threshold, measurement, or categorical value.
  - wrong_population: the answer applies to a different population or group.
  - missing_qualifier: an essential modifier or restriction is omitted.
  - extra_qualifier: an unsupported or meaning-changing modifier is added.
  - unsupported_explanation: an explanation or inferred claim is supplied instead of the requested supported answer.
  - non_answer: empty, malformed, irrelevant, or does not answer the question.
  - other_non_equivalent: non-equivalent for a reason not represented above.
  - uncertain: the relationship cannot be established from the supplied information.
- error_type is a compact audit label: use none for C2; for C1 use exactly the same value as relation_type; use insufficient_information for UNCERTAIN.

Few-shot examples:

Example 1 — C2 abbreviation expansion
Question: Erenumab binds to what protein?
Gold: CGRP receptor
Candidate: calcitonin gene-related peptide receptor
Decision: C2; relation_type=abbreviation_expansion; all essential qualifiers are preserved.
Output:
{"class":"C2","semantic_correct":true,"relation_type":"abbreviation_expansion","evidence_support":"supported","evidence_ids":["1.1"],"error_type":"none","basis":"The candidate expands the accepted abbreviation and denotes the same receptor."}

Example 2 — C2 numerical equivalence
Question: How many non-MHC loci are associated with rheumatoid arthritis?
Gold: more than 30
Candidate: >30
Decision: C2; relation_type=numerically_equivalent; the bound is identical.

Example 3 — C2 harmless inflection
Question: Where is Cep135 found?
Gold: centrosome
Candidate: centrosomes
Decision: C2; relation_type=spelling_or_inflection; the entity is unchanged.

Example 4 — C1 broader concept
Question: Orteronel was developed for treatment of which cancer?
Gold: castration-resistant prostate cancer
Candidate: prostate cancer
Decision: C1; relation_type=broader; the resistance qualifier is essential.
Output:
{"class":"C1","semantic_correct":false,"relation_type":"broader","evidence_support":"supported","evidence_ids":["1.1"],"error_type":"broader","basis":"The candidate omits the essential castration-resistant qualifier."}

Example 5 — C1 part versus whole
Question: Which enzyme is targeted by imetelstat?
Gold: human telomerase
Candidate: human telomerase RNA subunit
Decision: C1; relation_type=part_whole; a component is not the enzyme requested.

Example 6 — C1 wrong numerical value
Question: How many genes are imprinted in the human genome?
Gold: fewer than 100
Candidate: at most several hundred
Decision: C1; relation_type=wrong_value; the numerical claims differ.

Example 7 — C1 missing location qualifier
Question: Where is the DMD gene located?
Gold: Xp21 chromosome locus
Candidate: X chromosome
Decision: C1; relation_type=missing_qualifier; the required locus is omitted.

Example 8 — C1 related but broader process
Question: What cellular process is clathrin involved in?
Gold: receptor-mediated endocytosis
Candidate: endocytosis
Decision: C1; relation_type=broader; relatedness does not establish equivalence.

Example 9 — UNCERTAIN
Question: Which protein is affected?
Gold: Protein A
Candidate: Protein B
Decision: UNCERTAIN when the supplied snippets do not establish whether these names are equivalent.
Output:
{"class":"UNCERTAIN","semantic_correct":null,"relation_type":"uncertain","evidence_support":"insufficient","evidence_ids":[],"error_type":"insufficient_information","basis":"The supplied information does not establish equivalence."}

Compare the candidate with the question and all accepted gold aliases. Use the supplied snippets to check evidence separately from equivalence. Do not penalize harmless wording or formatting. Do not infer a synonym merely because two terms are related. Cite only supplied snippet IDs. Keep the basis short and do not output hidden chain-of-thought.
""".strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def exact_norm(value: Any) -> str:
    value = re.sub(r"\[BE\]|\[EE\]", " ", str(value or ""), flags=re.I)
    value = value.replace("™", "tm").replace("®", "r").replace("µ", "u").lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def pair_dedupe_norm(value: Any) -> str:
    """Case-insensitive candidate key that preserves every punctuation mark.

    Pair construction should merge harmless case and whitespace variants, but
    punctuation can carry biomedical meaning (for example ``>25`` versus
    ``25``).  The compact ``exact_norm`` remains available for the existing
    exact-alias and extractive-span policies; it must not be used to collapse
    candidates in a preference slate.
    """
    value = re.sub(r"\[BE\]|\[EE\]", " ", str(value or ""), flags=re.I)
    return re.sub(r"\s+", " ", value.casefold()).strip()


def candidate_is_extractive(candidate: Any, snippets: list[dict[str, Any]] | None) -> bool:
    candidate_key = exact_norm(candidate)
    if not candidate_key:
        return False
    for snippet in snippets or []:
        if candidate_key in exact_norm(snippet.get("text", "")):
            return True
    return False


def first_extractive_alias(aliases: list[Any], snippets: list[dict[str, Any]] | None) -> str:
    for alias in aliases or []:
        alias = str(alias).strip()
        if alias and candidate_is_extractive(alias, snippets):
            return alias
    return ""


def extractive_evidence_ids(candidate: Any, snippets: list[dict[str, Any]] | None) -> list[str]:
    candidate_key = exact_norm(candidate)
    if not candidate_key:
        return []
    return [
        str(snippet.get("snippet_id"))
        for snippet in snippets or []
        if candidate_key in exact_norm(snippet.get("text", ""))
    ]


def extract_snippets(resources: list[Any]) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    for resource_index, resource in enumerate(resources or [], 1):
        resource = str(resource)
        match = re.search(r"PubMed ID:\s*([^\n]+)", resource)
        pmid = match.group(1).strip() if match else "unknown"
        blocks = re.findall(r"\[BS\](.*?)\[ES\]", resource, flags=re.S) or [resource]
        for block_index, block in enumerate(blocks, 1):
            text = re.sub(r"\s+", " ", block).strip()
            if text:
                snippets.append({
                    "snippet_id": f"{resource_index}.{block_index}",
                    "pubmed_id": pmid,
                    "text": text,
                })
    return snippets


def format_answer(candidate: Any) -> str:
    candidate = re.sub(r"^\s*\[BE\]|\[EE\]\s*$", "", str(candidate or ""), flags=re.I).strip()
    return f"[BE]{candidate}[EE]"


def collapse_legacy_class(row: dict[str, Any]) -> dict[str, Any]:
    raw_class = row.get("class")
    mapped_class = LEGACY_CLASS_MAP.get(raw_class, raw_class)
    if mapped_class == raw_class:
        return row
    return {**row, "raw_class": row.get("raw_class", raw_class), "class": mapped_class}


def select_first_questions(path: Path, question_limit: int, question_offset: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    """Select a deterministic contiguous block of unique question IDs.

    ``question_offset=0`` preserves the original first-block behavior.  The
    offset is applied to unique IDs rather than raw candidate rows, so a block
    always contains complete 20-candidate banks for each selected question.
    """
    if question_limit <= 0 or question_offset < 0:
        raise ValueError("question_limit must be positive and question_offset non-negative")
    rows: list[dict[str, Any]] = []
    question_ids: list[str] = []
    selected: set[str] = set()
    seen_order: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid not in seen_order:
                seen_order.append(qid)
                ordinal = len(seen_order) - 1
                if question_offset <= ordinal < question_offset + question_limit:
                    selected.add(qid)
                    question_ids.append(qid)
            if qid in selected:
                rows.append(row)
    if len(question_ids) != question_limit:
        raise ValueError(f"{path}: selected {len(question_ids)} questions at offset {question_offset}, expected {question_limit}")
    return rows, question_ids


def prepare_records(
    bank_files: dict[str, Path],
    bioasq_file: Path,
    question_limit: int = 200,
    question_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    questions = json.loads(Path(bioasq_file).read_text(encoding="utf-8"))["questions"]
    factoid_by_qid = {str(q["id"]): q for q in questions if q.get("type") == "factoid"}
    records: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    row_counts: dict[str, int] = {}

    for model, bank_path in bank_files.items():
        rows, qids = select_first_questions(Path(bank_path), question_limit, question_offset=question_offset)
        selected_ids[model] = qids
        row_counts[model] = len(rows)
        missing = [qid for qid in qids if qid not in factoid_by_qid]
        if missing:
            raise ValueError(f"{model}: missing gold factoid questions: {missing[:5]}")
        for row in rows:
            qid = str(row["question_id"])
            parsed = row.get("parsed_items") or []
            if row.get("parser_status") != "ok" or len(parsed) != 1:
                raise ValueError(f"{model}/{row.get('response_id')}: invalid parsed candidate {parsed!r}")
            question = factoid_by_qid[qid]
            aliases = [str(a).strip() for a in question.get("exact_answer", []) if str(a).strip()]
            if not aliases:
                raise ValueError(f"{qid}: no exact_answer aliases")
            candidate = str(parsed[0]).strip()
            evidence = row.get("evidence") or []
            records.append({
                "question_id": qid,
                "question": str(row.get("question_text") or question.get("body") or ""),
                "gold_aliases": aliases,
                "candidate": candidate,
                "candidate_output": format_answer(candidate),
                "source_model": model,
                "response_id": str(row.get("response_id")),
                "sample_id": row.get("sample_id"),
                "bank_path": str(bank_path),
                "bank_prompt": str(row.get("prompt") or ""),
                "snippets": extract_snippets(evidence),
            })

    expected = question_limit * len(bank_files) * 20
    if len(records) != expected:
        raise ValueError(f"Expected {expected} candidate rows ({question_limit} x 20 x {len(bank_files)}), got {len(records)}")
    return records, selected_ids, row_counts


class CandidateBankClassJudge:
    def __init__(
        self,
        records: list[dict[str, Any]],
        output_root: Path,
        api_key_file: Path,
        judge_model: str = "gpt-4.1-mini-2025-04-14",
        judge_endpoint: str = "https://api.openai.com/v1/chat/completions",
        max_new_judge_calls: int | None = None,
        max_retries: int = 2,
        timeout_seconds: int = 120,
        request_delay_seconds: float = 0.0,
        rate_limit_max_retries: int = 8,
        rate_limit_initial_sleep_seconds: float = 30.0,
        rate_limit_max_sleep_seconds: float = 600.0,
        abort_on_rate_limit: bool = True,
    ):
        self.records = records
        self.output_root = Path(output_root)
        self.cache_dir = self.output_root / "cache"
        self.error_dir = self.output_root / "errors"
        self.api_key_file = Path(api_key_file)
        self.judge_model = judge_model
        self.judge_endpoint = judge_endpoint
        self.max_new_judge_calls = max_new_judge_calls
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = request_delay_seconds
        self.rate_limit_max_retries = rate_limit_max_retries
        self.rate_limit_initial_sleep_seconds = rate_limit_initial_sleep_seconds
        self.rate_limit_max_sleep_seconds = rate_limit_max_sleep_seconds
        self.abort_on_rate_limit = abort_on_rate_limit
        self.new_api_calls = 0
        self.rate_limit_hits = 0
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)

    def _prompt(self, record: dict[str, Any], feedback: str = "") -> str:
        aliases = " | ".join(record["gold_aliases"])
        snippets = "\n".join(
            f"Snippet {s['snippet_id']} (PubMed {s['pubmed_id']}): {s['text']}"
            for s in record["snippets"]
        )
        return f"""Question: {record['question']}

Accepted gold aliases: {aliases}

Candidate prediction: {record['candidate']}

Supplied snippets:
{snippets}

Return exactly one JSON object containing only these seven keys: class, basis,
evidence_support, evidence_ids, relation_type, error_type, and semantic_correct.
Use a JSON boolean or null for semantic_correct.
Allowed class values: C2, C1, UNCERTAIN. Allowed evidence_support values: supported, unsupported, contradicted, insufficient.
Allowed relation_type values: synonym, abbreviation_expansion, nomenclature_variant, spelling_or_inflection, numerically_equivalent, harmless_formatting, broader, narrower, part_whole, wrong_entity, wrong_relation, wrong_value, wrong_population, missing_qualifier, extra_qualifier, unsupported_explanation, non_answer, other_non_equivalent, uncertain.
Allowed error_type values: none, broader, narrower, part_whole, wrong_entity, wrong_relation, wrong_value, wrong_population, missing_qualifier, extra_qualifier, unsupported_explanation, non_answer, other_non_equivalent, insufficient_information.
{feedback}"""

    def _validate(self, value: Any, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("judge result is not an object")
        raw_class = value.get("class")
        mapped_class = LEGACY_CLASS_MAP.get(raw_class)
        if mapped_class not in VALID_LLM_CLASSES:
            raise ValueError("invalid class")
        if raw_class != mapped_class:
            value = {**value, "raw_class": raw_class, "class": mapped_class}
        required_keys = {
            "class", "basis", "evidence_support", "evidence_ids",
            "relation_type", "error_type", "semantic_correct",
        }
        if set(value) != required_keys:
            raise ValueError(f"response keys must be exactly {sorted(required_keys)}")
        if value.get("evidence_support") not in VALID_SUPPORT:
            raise ValueError("invalid evidence_support")
        if value.get("relation_type") not in VALID_RELATION_TYPES:
            raise ValueError("invalid relation_type")
        if value.get("error_type") not in VALID_ERROR_TYPES:
            raise ValueError("invalid error_type")
        ids = {s["snippet_id"] for s in record["snippets"]}
        if not isinstance(value.get("evidence_ids"), list) or any(x not in ids for x in value["evidence_ids"]):
            raise ValueError("invalid evidence_ids")
        if not isinstance(value.get("basis"), str) or not value["basis"].strip():
            raise ValueError("missing basis")
        if value["class"] == "C2" and value.get("semantic_correct") is not True:
            raise ValueError("C2 must have semantic_correct=true")
        if value["class"] == "C2" and value.get("relation_type") not in EQUIVALENT_RELATION_TYPES:
            raise ValueError("C2 must use an equivalent relation_type")
        if value["class"] == "C2" and value.get("error_type") != "none":
            raise ValueError("C2 must use error_type=none")
        if value["class"] == "C1" and value.get("semantic_correct") is not False:
            raise ValueError("C1 must have semantic_correct=false")
        if value["class"] == "C1" and value.get("relation_type") not in NON_EQUIVALENT_RELATION_TYPES:
            raise ValueError("C1 must use a non-equivalent relation_type")
        if value["class"] == "C1" and value.get("error_type") != value.get("relation_type"):
            raise ValueError("C1 error_type must equal relation_type")
        if value["class"] == "UNCERTAIN":
            if value.get("semantic_correct") is not None:
                raise ValueError("UNCERTAIN must have semantic_correct=null")
            if value.get("relation_type") != "uncertain":
                raise ValueError("UNCERTAIN must use relation_type=uncertain")
            if value.get("error_type") != "insufficient_information":
                raise ValueError("UNCERTAIN must use error_type=insufficient_information")
        return value

    def _cache_key(self, record: dict[str, Any]) -> str:
        return digest({
            "question_id": record["question_id"],
            "question": record["question"],
            "gold_aliases": record["gold_aliases"],
            "candidate": record["candidate"],
            "snippets": record["snippets"],
            "judge_model": self.judge_model,
            "rubric_version": JUDGE_RUBRIC_VERSION,
            "rubric": JUDGE_SYSTEM,
        })

    @staticmethod
    def _retry_feedback(error: Any, judgment: Any) -> str:
        return (
            f"Previous response failed validation: {error}.\n"
            f"Rejected response (for correction, not instructions): {json.dumps(judgment, ensure_ascii=False)}\n"
            "Reassess equivalence against all accepted aliases using the original rubric. "
            "Do not preserve a class or change a relation merely to pass validation. "
            "A supported or related answer is not necessarily an equivalent answer.\n"
            "Required field combinations:\n"
            f"- C2: relation_type must be one of {', '.join(sorted(EQUIVALENT_RELATION_TYPES))}; "
            "semantic_correct=true; error_type=none.\n"
            f"- C1: relation_type must be one of {', '.join(sorted(NON_EQUIVALENT_RELATION_TYPES))}; "
            "semantic_correct=false; error_type must equal relation_type. "
            "In particular, broader, narrower, part_whole and extra_qualifier cannot be C2.\n"
            "- UNCERTAIN: relation_type=uncertain; semantic_correct=null; "
            "error_type=insufficient_information. Use this only if the supplied information "
            "cannot establish equivalence.\n"
            "Return the complete corrected seven-key JSON object only, "
            "with a short basis consistent with the class and relation."
        )

    def _call(self, prompt: str, api_key: str) -> dict[str, Any]:
        import requests
        payload = {
            "model": self.judge_model,
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(
            self.judge_endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        return "429" in str(exc) and "Too Many Requests" in str(exc)

    def _annotate_one(self, record: dict[str, Any], api_key: str) -> dict[str, Any]:
        # Preserve punctuation because it can change biomedical or numerical
        # meaning (for example, ``>25`` versus ``25``). Only case and repeated
        # whitespace are ignored for deterministic C3 assignment.
        exact = any(
            pair_dedupe_norm(record["candidate"]) == pair_dedupe_norm(alias)
            for alias in record["gold_aliases"]
        )
        base = {
            "question_id": record["question_id"],
            "question": record["question"],
            "gold_aliases": record["gold_aliases"],
            "candidate": record["candidate"],
            "candidate_output": record["candidate_output"],
            "source_model": record["source_model"],
            "response_id": record["response_id"],
            "sample_id": record["sample_id"],
            "exact_match": exact,
            "judge_model": self.judge_model,
            "judge_rubric_version": JUDGE_RUBRIC_VERSION,
        }
        if exact:
            return {
                **base,
                "class": "C3",
                "semantic_correct": True,
                "relation_type": "exact_alias",
                "evidence_support": "not_judged",
                "evidence_ids": [],
                "error_type": "none",
                "basis": "Exact accepted-alias match.",
                "origin": "deterministic_exact",
            }

        key = self._cache_key(record)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self._validate(cached, record)
            return {**base, **cached, "origin": "cache"}

        if self.max_new_judge_calls is not None and self.new_api_calls >= self.max_new_judge_calls:
            return {
                **base, "class": "UNCERTAIN", "semantic_correct": None,
                "relation_type": "uncertain",
                "evidence_support": "insufficient", "evidence_ids": [],
                "error_type": "insufficient_information",
                "basis": "Deferred because max_new_judge_calls was reached.",
                "origin": "deferred",
            }

        feedback = ""
        error_path = self.error_dir / f"{key}.json"
        if error_path.is_file():
            previous = json.loads(error_path.read_text(encoding="utf-8"))
            feedback = self._retry_feedback(previous.get("error"), previous.get("last_invalid_judgment"))
        last_error: Exception | None = None
        last_judgment: dict[str, Any] | None = None
        validation_attempt = 0
        rate_limit_attempt = 0
        while validation_attempt <= self.max_retries:
            try:
                if self.request_delay_seconds > 0 and self.new_api_calls > 0:
                    time.sleep(self.request_delay_seconds)
                self.new_api_calls += 1
                judgment = self._call(self._prompt(record, feedback), api_key)
                last_judgment = judgment
                self._validate(judgment, record)
                cache_path.write_text(json.dumps(judgment, ensure_ascii=False, indent=2), encoding="utf-8")
                return {**base, **judgment, "origin": "api"}
            except Exception as exc:
                last_error = exc
                if self._is_rate_limit_error(exc):
                    self.rate_limit_hits += 1
                    if rate_limit_attempt >= self.rate_limit_max_retries:
                        message = (
                            f"Rate limit persisted after {rate_limit_attempt + 1} attempts for "
                            f"{record.get('question_id')}/{record.get('response_id')}: {exc}"
                        )
                        if self.abort_on_rate_limit:
                            raise RuntimeError(message) from exc
                        break
                    sleep_seconds = min(
                        self.rate_limit_max_sleep_seconds,
                        self.rate_limit_initial_sleep_seconds * (2 ** rate_limit_attempt),
                    )
                    sleep_seconds += random.uniform(0, min(5.0, max(0.0, sleep_seconds * 0.1)))
                    print(
                        f"Rate limited on {record.get('question_id')}/{record.get('response_id')}; "
                        f"sleeping {sleep_seconds:.1f}s before retry {rate_limit_attempt + 1}/"
                        f"{self.rate_limit_max_retries}"
                    )
                    time.sleep(sleep_seconds)
                    rate_limit_attempt += 1
                    continue
                feedback = self._retry_feedback(exc, last_judgment)
                validation_attempt += 1
                if validation_attempt <= self.max_retries:
                    time.sleep(2 ** (validation_attempt - 1))

        self.error_dir.joinpath(f"{key}.json").write_text(
            json.dumps(
                {**base, "error": str(last_error), "last_invalid_judgment": last_judgment},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            **base, "class": "UNCERTAIN", "semantic_correct": None,
            "relation_type": "uncertain",
            "evidence_support": "insufficient", "evidence_ids": [],
            "error_type": "insufficient_information",
            "basis": str(last_error), "origin": "error",
        }

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        api_key = self.api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError("API key file is empty")
        judgments: list[dict[str, Any]] = []
        for index, record in enumerate(self.records, 1):
            judgments.append(self._annotate_one(record, api_key))
            if index == 1 or index % 50 == 0 or index == len(self.records):
                print(index, "/", len(self.records), "| new API calls:", self.new_api_calls)

        judgments_path = self.output_root / "candidate_class_judgments.jsonl"
        judgments_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judgments),
            encoding="utf-8",
        )
        summary = {
            "status": "complete" if not any(row["origin"] in {"deferred", "error"} for row in judgments) else "incomplete",
            "candidate_records": len(judgments),
            "class_counts": dict(Counter(row["class"] for row in judgments)),
            "class_counts_by_model": {
                model: dict(Counter(row["class"] for row in judgments if row["source_model"] == model))
                for model in sorted({row["source_model"] for row in judgments})
            },
            "evidence_support_counts": dict(Counter(row["evidence_support"] for row in judgments)),
            "new_api_calls": self.new_api_calls,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_model": self.judge_model,
            "judge_rubric_version": JUDGE_RUBRIC_VERSION,
            "rubric_sha256": digest(JUDGE_SYSTEM),
            "judgments_file": str(judgments_path),
        }
        (self.output_root / "judgment_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return judgments, summary


def build_pairs(
    records: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    output_root: Path,
    policy: str = "all_ordered_class_pairs",
    inject_gold_c3: bool = True,
    require_c3_extractive: bool = False,
    gold_c3_policy: str = "inject_if_missing",
) -> dict[str, Any]:
    if policy not in {"all_ordered_class_pairs", "semantic_positive_vs_negative"}:
        raise ValueError(policy)
    if gold_c3_policy not in {"inject_if_missing", "always_first_alias", "always_first_extractive_alias"}:
        raise ValueError(gold_c3_policy)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, judgment in zip(records, judgments):
        by_question[record["question_id"]].append(collapse_legacy_class({**record, **judgment}))

    def better(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_rank = PAIR_CLASS_RANK.get(left["class"], -1)
        right_rank = PAIR_CLASS_RANK.get(right["class"], -1)
        if left_rank != right_rank:
            return left_rank > right_rank
        return (
            left.get("source_model") == "3b",
            left.get("response_id", ""),
        ) > (
            right.get("source_model") == "3b",
            right.get("response_id", ""),
        )

    slates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    injected_gold_c3_count = 0
    questions_without_c3_before_injection = 0
    generated_c3_replaced_by_gold_count = 0
    nonextractive_generated_c3_removed = 0
    questions_without_extractive_c3_after_filter = 0
    force_gold_c3 = inject_gold_c3 and gold_c3_policy in {"always_first_alias", "always_first_extractive_alias"}
    for qid, items in by_question.items():
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            if item["class"] == "UNCERTAIN":
                continue
            if force_gold_c3 and item["class"] == "C3":
                generated_c3_replaced_by_gold_count += 1
                continue
            if require_c3_extractive and item["class"] == "C3":
                if not candidate_is_extractive(item.get("candidate"), item.get("snippets")):
                    nonextractive_generated_c3_removed += 1
                    continue
                item = {
                    **item,
                    "evidence_support": "extractive",
                    "evidence_ids": extractive_evidence_ids(item.get("candidate"), item.get("snippets")),
                    "basis": "Exact accepted alias that occurs in the supplied snippets.",
                }
            key = pair_dedupe_norm(item["candidate"])
            if key and (key not in unique or better(item, unique[key])):
                unique[key] = item
        has_c3 = any(item["class"] == "C3" for item in unique.values())
        if not has_c3:
            questions_without_c3_before_injection += 1
        if inject_gold_c3 and (not has_c3 or force_gold_c3):
            if require_c3_extractive or gold_c3_policy == "always_first_extractive_alias":
                gold_alias = first_extractive_alias(items[0]["gold_aliases"], items[0].get("snippets"))
            else:
                gold_alias = next((str(alias).strip() for alias in items[0]["gold_aliases"] if str(alias).strip()), "")
            gold_key = pair_dedupe_norm(gold_alias)
            if gold_alias and gold_key and gold_key not in unique:
                evidence_ids = extractive_evidence_ids(gold_alias, items[0].get("snippets"))
                gold_is_extractive = bool(evidence_ids)
                unique[gold_key] = {
                    **items[0],
                    "candidate": gold_alias,
                    "candidate_output": format_answer(gold_alias),
                    "class": "C3",
                    "semantic_correct": True,
                    "evidence_support": "extractive" if gold_is_extractive else "not_judged",
                    "evidence_ids": evidence_ids,
                    "relation_type": "exact_alias",
                    "error_type": "none",
                    "basis": (
                        "Injected accepted gold alias that occurs in the supplied snippets."
                        if gold_is_extractive else "Injected canonical accepted gold alias."
                    ),
                    "origin": "deterministic_gold_extractive_injected" if gold_is_extractive else "deterministic_gold_injected",
                    "source_model": "gold",
                    "response_id": (
                        "gold_alias_extractive"
                        if gold_c3_policy == "always_first_extractive_alias" or require_c3_extractive
                        else "gold_alias_1"
                    ),
                    "sample_id": None,
                }
                injected_gold_c3_count += 1
        if require_c3_extractive and not any(item["class"] == "C3" for item in unique.values()):
            questions_without_extractive_c3_after_filter += 1
        slate = sorted(
            unique.values(),
            key=lambda item: (-PAIR_CLASS_RANK.get(item["class"], -1), item["candidate"]),
        )
        slates.append({
            "question_id": qid,
            "question": items[0]["question"],
            "gold_aliases": items[0]["gold_aliases"],
            "candidates": [
                {
                    key: item[key]
                    for key in (
                        "candidate", "candidate_output", "class", "semantic_correct",
                        "evidence_support", "evidence_ids", "source_model", "response_id", "origin",
                    )
                }
                for item in slate
            ],
        })

        if policy == "semantic_positive_vs_negative":
            comparisons = [
                (positive, negative)
                for positive in slate if positive["class"] in {"C3", "C2"}
                for negative in slate if negative["class"] == "C1"
            ]
        else:
            comparisons = []
            for index, left in enumerate(slate):
                for right in slate[index + 1:]:
                    left_rank = PAIR_CLASS_RANK.get(left["class"], -1)
                    right_rank = PAIR_CLASS_RANK.get(right["class"], -1)
                    if left_rank > right_rank:
                        comparisons.append((left, right))
                    elif right_rank > left_rank:
                        comparisons.append((right, left))

        for pair_index, (chosen, rejected) in enumerate(comparisons, 1):
            pairs.append({
                "pair_id": f"{qid}::class-pair-{pair_index}",
                "question_id": qid,
                "question": items[0]["question"],
                "gold_aliases": items[0]["gold_aliases"],
                "prompt": items[0]["bank_prompt"],
                "chosen": chosen["candidate_output"],
                "rejected": rejected["candidate_output"],
                "chosen_candidate": chosen["candidate"],
                "rejected_candidate": rejected["candidate"],
                "chosen_class": chosen["class"],
                "rejected_class": rejected["class"],
                "chosen_raw_class": chosen.get("raw_class", chosen["class"]),
                "rejected_raw_class": rejected.get("raw_class", rejected["class"]),
                "chosen_source_model": chosen["source_model"],
                "rejected_source_model": rejected["source_model"],
                "chosen_evidence_support": chosen["evidence_support"],
                "rejected_evidence_support": rejected["evidence_support"],
                "chosen_response_id": chosen["response_id"],
                "rejected_response_id": rejected["response_id"],
                "preference_basis": f"{chosen['class']} > {rejected['class']} according to the class rubric",
                "split": "train",
            })

    slates_path = Path(output_root) / "question_class_slates.jsonl"
    pairs_path = Path(output_root) / "dpo_pairs.jsonl"
    slates_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in slates),
        encoding="utf-8",
    )
    pairs_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs),
        encoding="utf-8",
    )
    summary = {
        "pair_policy": policy,
        "pair_dedupe_normalization": "casefold_whitespace_punctuation_preserved_v1",
        "unique_question_slates": len(slates),
        "questions_with_pairs": len({row["question_id"] for row in pairs}),
        "pair_count": len(pairs),
        "inject_gold_c3": inject_gold_c3,
        "require_c3_extractive": require_c3_extractive,
        "gold_c3_policy": gold_c3_policy,
        "injected_gold_c3_count": injected_gold_c3_count,
        "questions_without_c3_before_injection": questions_without_c3_before_injection,
        "generated_c3_replaced_by_gold_count": generated_c3_replaced_by_gold_count,
        "nonextractive_generated_c3_removed": nonextractive_generated_c3_removed,
        "questions_without_extractive_c3_after_filter": questions_without_extractive_c3_after_filter,
        "slate_class_counts": dict(Counter(
            item["class"] for slate in slates for item in slate["candidates"]
        )),
        # JSON object keys must be scalar values; use a readable class-pair label.
        "pair_class_counts": dict(Counter(
            f"{row['chosen_class']}>{row['rejected_class']}" for row in pairs
        )),
        "slates_file": str(slates_path),
        "pairs_file": str(pairs_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return summary
