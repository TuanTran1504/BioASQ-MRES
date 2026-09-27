#!/usr/bin/env python
"""Compare direct GPT factoid extraction with structured evidence reasoning.

This is a paired evaluation on the same BioASQ dev questions.  Neither prompt
contains the gold answer.  Each response is cached independently, so a pilot can
be extended to all 160 questions without repeating completed API calls.

The experiment tests inference-time reasoning only.  It does not train a model
and it does not establish that a smaller local model will learn the same process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cse_dpo.generated_bioasq_eval import load_gold_examples
from src.utility.bioasq_format import parse_prediction_items
from src.utility.bioasq_official import evaluate_with_bioasq_java


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "single_answer_full_resources_qwen25_05b/eval_prepared.json"
)
V1_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/inference_strategy_comparisons/"
    "gpt41mini_direct_vs_structured_reasoning_dev160_v1"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/inference_strategy_comparisons/"
    "gpt41mini_direct_vs_answer_shape_reasoning_dev160_v2"
)
DEFAULT_API_KEY_FILE = ROOT / "open_ai_api.txt"
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
EXPERIMENT_VERSION_V1 = "gpt-direct-vs-structured-evidence-reasoning-v1"
EXPERIMENT_VERSION_V2 = "gpt-direct-vs-answer-shape-reasoning-v2"


DIRECT_SYSTEM_PROMPT = """
You are a biomedical factoid answer extractor.

Answer the question using only the supplied PubMed resources. Select exactly one
short biomedical expression that occurs as a continuous span in the supplied
resources, preferably inside [BS] and [ES]. Preserve the source wording and all
essential qualifiers. Do not translate, expand, normalize, paraphrase, combine,
or infer an answer. Return exactly this format and no explanation:
[BE] extracted expression [EE]
""".strip()


REASONING_SYSTEM_PROMPT_V1 = """
You are a biomedical factoid answer extractor using structured evidence-guided
reasoning. Return JSON only.

Analyze the supplied question and PubMed resources in this order:
1. Identify the requested answer type and the exact relation or value requested.
2. List the essential constraints, including population, condition, location,
   subtype, measurement, and other qualifiers when present.
3. Identify 1 to 5 plausible candidate expressions from the supplied resources.
   Every candidate must be copied as a continuous source span.
4. For every candidate, cite its resource number, check its entity/value type,
   check whether the surrounding evidence expresses the requested relation, and
   mark it accept or reject.
5. Select the one candidate that satisfies all question constraints. Preserve its
   source surface exactly. Do not use outside biomedical knowledge.

Return exactly these keys:
- requested_answer_type: short string
- required_relation: short string
- essential_constraints: JSON list of short strings
- candidates: JSON list of 1 to 5 objects, each containing exactly text,
  resource_id, candidate_type, relation_check, constraint_check, and decision
- selected_answer: exactly one candidate text
- selection_reason: one concise evidence-based sentence

Candidate decision must be "accept" or "reject". Exactly one candidate must be
accepted, and it must equal selected_answer. Do not include [BE] or [EE] inside
selected_answer.
""".strip()


REASONING_SYSTEM_PROMPT_V2 = """
You are a biomedical factoid answer extractor using structured evidence-guided
reasoning. Return JSON only.

Analyze the supplied question and PubMed resources in this order:
1. Predict answer_shape as exactly one of entity_span, value_span, process_span,
   or definition_clause.
   - entity_span: a named entity, acronym expansion, organism, disease, protein,
     gene, drug, treatment, location, structure, or other noun phrase.
   - value_span: a number, percentage, date, duration, count, or measurement.
   - process_span: a named biological process, function, activity, or mechanism.
   - definition_clause: a clause or sentence genuinely required to define or
     describe the subject. Use this only when a shorter name, category, acronym
     expansion, entity phrase, value, or process cannot answer the question.
2. Identify the requested answer type, required relation, and every essential
   qualifier in the question.
3. Find 1 to 5 plausible continuous spans in the resources. Preserve punctuation,
   hyphens, capitalization, and source wording exactly.
4. For each candidate, check its type, relation, constraints, and boundaries.
   Reject a whole sentence or surrounding explanation whenever a shorter contained
   span preserves the requested answer and all essential qualifiers.
5. Select the shortest candidate that completely answers the question. Do not
   remove essential modifiers merely to make it shorter. Do not use outside
   biomedical knowledge.

Boundary rules:
- For entity_span, value_span, and process_span, return only the minimal answer
  phrase. Exclude lead-in subjects, reporting verbs, explanations, citations, and
  trailing commentary.
- An acronym expansion is normally entity_span, not definition_clause.
- For definition_clause, copy only the decisive definitional clause. A full
  sentence is allowed only for this shape and only when the entire sentence is
  needed to answer the question.
- If no supplied resource supports any answer, return no accepted candidate and
  set selected_answer to the empty string.

Return exactly these keys:
- answer_shape: entity_span, value_span, process_span, or definition_clause
- requested_answer_type: short string
- required_relation: short string
- essential_constraints: JSON list of short strings
- candidates: JSON list of 0 to 5 objects, each containing exactly text,
  resource_id, candidate_type, relation_check, constraint_check, boundary_check,
  and decision
- selected_answer: the accepted candidate text, or an empty string when unsupported
- selection_reason: one concise evidence-based sentence

Candidate decision must be "accept" or "reject". When selected_answer is not
empty, exactly one candidate must be accepted and equal selected_answer. Do not
include [BE] or [EE] inside selected_answer.
""".strip()


REASONING_SYSTEM_PROMPTS = {
    "v1": REASONING_SYSTEM_PROMPT_V1,
    "v2": REASONING_SYSTEM_PROMPT_V2,
}
ANSWER_SHAPES = {"entity_span", "value_span", "process_span", "definition_clause"}
STRUCTURED_KEYS_V1 = {
    "requested_answer_type",
    "required_relation",
    "essential_constraints",
    "candidates",
    "selected_answer",
    "selection_reason",
}
STRUCTURED_KEYS_V2 = STRUCTURED_KEYS_V1 | {"answer_shape"}
CANDIDATE_KEYS_V1 = {
    "text",
    "resource_id",
    "candidate_type",
    "relation_check",
    "constraint_check",
    "decision",
}
CANDIDATE_KEYS_V2 = CANDIDATE_KEYS_V1 | {"boundary_check"}


def digest(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    value = next((line for line in lines if line and not line.startswith("#")), "")
    if value.startswith("OPENAI_API_KEY="):
        value = value.split("=", 1)[1].strip()
    value = value.strip().strip(chr(34)).strip(chr(39))
    if not value:
        raise ValueError(f"No API key found in {path}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("questions")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list or an object containing questions")
    rows = [row for row in rows if isinstance(row, dict) and row.get("type") == "factoid"]
    ids = [str(row.get("id") or "").strip() for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: factoid IDs must be present and unique")
    return rows


def resources_from_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    resources: list[tuple[int, str]] = []
    for key, value in row.items():
        match = re.fullmatch(r"input_(\d+)", str(key))
        if not match or int(match.group(1)) < 2:
            continue
        text = str(value or "").strip()
        if text:
            resources.append((int(match.group(1)) - 1, text))
    return [(str(number), text) for number, text in sorted(resources)]




def build_resource_id_aliases(resources: list[tuple[str, str]]) -> dict[str, str]:
    """Map displayed resource labels and PubMed identifiers to resource numbers."""
    aliases: dict[str, str] = {}
    for resource_id, text in resources:
        aliases[resource_id] = resource_id
        aliases[f"resource {resource_id}".casefold()] = resource_id
        for match in re.finditer(r"PubMed\s+ID:\s*([0-9]+)", text, flags=re.I):
            pubmed_id = match.group(1)
            aliases[pubmed_id] = resource_id
            aliases[f"pmid {pubmed_id}".casefold()] = resource_id
            aliases[f"pubmed {pubmed_id}".casefold()] = resource_id
    return aliases


def canonical_resource_id(value: Any, aliases: dict[str, str]) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        raw = str(value)
    elif isinstance(value, float) and value.is_integer():
        raw = str(int(value))
    else:
        raw = str(value or "").strip()
    folded = re.sub(r"\s+", " ", raw).casefold().strip()
    if folded in aliases:
        return aliases[folded]
    # Be permissive about labels such as "Resource 3 (PMID 123)" while still
    # requiring the extracted identifier to correspond to a supplied resource.
    resource_match = re.search(r"\bresource\s*#?\s*([0-9]+)\b", folded)
    if resource_match and resource_match.group(1) in aliases:
        return aliases[resource_match.group(1)]
    pubmed_match = re.search(r"\b(?:pmid|pubmed(?:\s+id)?)\s*[:#]?\s*([0-9]+)\b", folded)
    if pubmed_match and pubmed_match.group(1) in aliases:
        return aliases[pubmed_match.group(1)]
    return raw

def build_user_prompt(row: dict[str, Any]) -> str:
    resources = "\n\n".join(
        f"Resource {resource_id}:\n{text}"
        for resource_id, text in resources_from_row(row)
    )
    return f"Question: {str(row.get('input_1') or '').strip()}\n\nPubMed resources:\n\n{resources}"


def validate_direct(text: str) -> str:
    """Parse one direct answer and canonicalize harmless format variation.

    Chat models occasionally prepend ``Answer:`` or add text outside an otherwise
    valid tag pair.  Formatting is not the behavior under comparison, so retain
    the single parsed answer surface and store it in canonical BioASQ tags.
    """
    items = parse_prediction_items(text, "factoid")
    if len(items) != 1:
        raise ValueError(f"direct response must contain exactly one answer, got {len(items)}")
    answer = str(items[0] or "").strip()
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.I).strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {chr(34), chr(39)}:
        answer = answer[1:-1].strip()
    if not answer:
        raise ValueError("direct response contains an empty answer")
    return f"[BE]{answer}[EE]"


def validate_structured(
    value: Any,
    valid_resource_ids: set[str],
    *,
    prompt_version: str = "v2",
    resource_id_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    structured_keys = STRUCTURED_KEYS_V2 if prompt_version == "v2" else STRUCTURED_KEYS_V1
    candidate_keys = CANDIDATE_KEYS_V2 if prompt_version == "v2" else CANDIDATE_KEYS_V1
    aliases = dict(resource_id_aliases or {})
    for resource_id in valid_resource_ids:
        aliases.setdefault(str(resource_id).casefold(), str(resource_id))
    if not isinstance(value, dict) or set(value) != structured_keys:
        raise ValueError(f"structured response keys must be exactly {sorted(structured_keys)}")
    if prompt_version == "v2" and value.get("answer_shape") not in ANSWER_SHAPES:
        raise ValueError(f"answer_shape must be one of {sorted(ANSWER_SHAPES)}")
    for key in ("requested_answer_type", "required_relation", "selection_reason"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(value.get("selected_answer"), str):
        raise ValueError("selected_answer must be a string")
    if "[BE]" in value["selected_answer"] or "[EE]" in value["selected_answer"]:
        raise ValueError("selected_answer must not contain output tags")
    constraints = value.get("essential_constraints")
    if not isinstance(constraints, list) or any(not isinstance(x, str) for x in constraints):
        raise ValueError("essential_constraints must be a list of strings")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 0 <= len(candidates) <= 5:
        raise ValueError("candidates must contain 0 to 5 objects")
    accepted: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != candidate_keys:
            raise ValueError(f"candidate keys must be exactly {sorted(candidate_keys)}")
        # JSON models commonly emit a resource number as either 3 or "3".
        # Resource IDs are labels rather than quantities, so canonicalize both
        # representations before applying the strict schema checks and caching.
        resource_id = candidate.get("resource_id")
        candidate["resource_id"] = canonical_resource_id(resource_id, aliases)
        for key in candidate_keys - {"decision"}:
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ValueError(f"candidate {key} must be a non-empty string")
        if str(candidate["resource_id"]) not in valid_resource_ids:
            raise ValueError(f"unknown resource_id {candidate['resource_id']!r}")
        if candidate.get("decision") not in {"accept", "reject"}:
            raise ValueError("candidate decision must be accept or reject")
        if candidate["decision"] == "accept":
            accepted.append(candidate["text"].strip())
    selected_answer = value["selected_answer"].strip()
    if selected_answer:
        if accepted != [selected_answer]:
            raise ValueError("exactly one accepted candidate must equal selected_answer")
    elif accepted:
        raise ValueError("an abstention cannot contain an accepted candidate")
    return value


def weak_surface(value: str) -> str:
    # Deliberately preserve punctuation and hyphens; normalize only Unicode,
    # casing, and whitespace for the extractivity diagnostic.
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def is_weakly_extractive(answer: str, resources: list[tuple[str, str]]) -> bool:
    needle = weak_surface(answer)
    return bool(needle) and any(needle in weak_surface(text) for _, text in resources)


class PairedGenerator:
    def __init__(self, args: argparse.Namespace, api_key: str):
        self.args = args
        self.api_key = api_key
        self.reasoning_prompt = REASONING_SYSTEM_PROMPTS[args.reasoning_prompt_version]
        self.experiment_version = (
            EXPERIMENT_VERSION_V2
            if args.reasoning_prompt_version == "v2"
            else EXPERIMENT_VERSION_V1
        )
        self.output_root = Path(args.output_root)
        self.cache_root = self.output_root / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.new_api_calls = 0
        self.rate_limit_hits = 0

    def cache_path(self, row: dict[str, Any], arm: str, system_prompt: str) -> Path:
        key = digest({
            "experiment_version": (
                EXPERIMENT_VERSION_V1 if arm == "direct" else self.experiment_version
            ),
            "question_id": row["id"],
            "user_prompt": build_user_prompt(row),
            "arm": arm,
            "system_prompt": system_prompt,
            "model": self.args.model,
            "temperature": 0,
        })
        path = self.cache_root / arm / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def is_rate_limit_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        return "429" in str(exc)

    def request(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": self.args.model,
            "temperature": 0,
            "max_tokens": self.args.reasoning_max_tokens if json_mode else self.args.direct_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            self.args.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.args.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "").strip()

    def generate_arm(self, row: dict[str, Any], arm: str) -> dict[str, Any]:
        structured = arm == "structured_reasoning"
        system_prompt = self.reasoning_prompt if structured else DIRECT_SYSTEM_PROMPT
        cache_path = self.cache_path(row, arm, system_prompt)
        resource_pairs = resources_from_row(row)
        resource_ids = {resource_id for resource_id, _ in resource_pairs}
        resource_aliases = build_resource_id_aliases(resource_pairs)
        cache_origin = "cache"
        candidate_cache_path = cache_path
        if not structured and not cache_path.exists() and self.args.reuse_direct_cache_root:
            reusable = Path(self.args.reuse_direct_cache_root) / cache_path.name
            if reusable.exists():
                candidate_cache_path = reusable
                cache_origin = "reused_direct_cache"
        if candidate_cache_path.exists():
            cached = json.loads(candidate_cache_path.read_text(encoding="utf-8"))
            if structured:
                validate_structured(
                    cached["response"], resource_ids,
                    prompt_version=self.args.reasoning_prompt_version,
                    resource_id_aliases=resource_aliases,
                )
            else:
                cached["response"] = validate_direct(cached["response"])
                if candidate_cache_path != cache_path:
                    cache_path.write_text(
                        json.dumps(cached, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
            return {**cached, "origin": cache_origin}

        feedback = ""
        validation_attempt = 0
        rate_limit_attempt = 0
        while validation_attempt <= self.args.max_validation_retries:
            try:
                if self.args.request_delay_seconds > 0 and self.new_api_calls > 0:
                    time.sleep(self.args.request_delay_seconds)
                prompt = build_user_prompt(row)
                if feedback:
                    prompt += f"\n\nCorrection required: {feedback}"
                self.new_api_calls += 1
                raw = self.request(system_prompt, prompt, json_mode=structured)
                response: Any = json.loads(raw) if structured else raw
                if structured:
                    response = validate_structured(
                        response, resource_ids,
                        prompt_version=self.args.reasoning_prompt_version,
                        resource_id_aliases=resource_aliases,
                    )
                else:
                    response = validate_direct(response)
                cached = {
                    "question_id": row["id"],
                    "arm": arm,
                    "model": self.args.model,
                    "response": response,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                cache_path.write_text(
                    json.dumps(cached, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return {**cached, "origin": "api"}
            except Exception as exc:
                if self.is_rate_limit_error(exc):
                    self.rate_limit_hits += 1
                    if rate_limit_attempt >= self.args.rate_limit_max_retries:
                        raise
                    delay = min(
                        self.args.rate_limit_max_sleep_seconds,
                        self.args.rate_limit_initial_sleep_seconds * (2**rate_limit_attempt),
                    )
                    delay += random.uniform(0, min(5.0, delay * 0.1))
                    print(f"Rate limited; sleeping {delay:.1f}s")
                    time.sleep(delay)
                    rate_limit_attempt += 1
                    continue
                validation_attempt += 1
                if validation_attempt > self.args.max_validation_retries:
                    raise RuntimeError(
                        f"{row['id']} {arm} failed after validation retries: {exc}"
                    ) from exc
                feedback = f"{exc}. Return a complete response in the required format."

        raise AssertionError("unreachable")


def score_arm(
    arm: str,
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    prediction_rows = [
        {
            "question_id": row["question_id"],
            "question_type": "factoid",
            "body": examples[row["question_id"]].body,
            "source_path": examples[row["question_id"]].source_path,
            "prediction": row[f"{arm}_prediction"],
        }
        for row in rows
    ]
    official = evaluate_with_bioasq_java(
        prediction_rows=prediction_rows,
        examples_by_key={(qid, "factoid"): example for qid, example in examples.items()},
        model_label=f"gpt41mini-{arm}",
        model_dir=output_root / arm,
        args=argparse.Namespace(
            bioasq_java_jar=str(
                ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
            ),
            bioasq_java_heap="512m",
            bioasq_java_version=5,
        ),
        include_per_question=True,
    )
    metrics = official["aggregate"]["by_type"]["factoid"]["metrics"]
    by_id = {row["question_id"]: row for row in official["per_question"]}
    return metrics, by_id


def paired_bootstrap_delta(
    rows: list[dict[str, Any]], key_a: str, key_b: str, *, seed: int = 3407, draws: int = 10000
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas = [float(row[key_b]) - float(row[key_a]) for row in rows]
    if not deltas:
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    means = []
    for _ in range(draws):
        means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    means.sort()
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "ci95_low": means[int(0.025 * draws)],
        "ci95_high": means[min(draws - 1, int(0.975 * draws))],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat_rows.append({
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        })
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_review_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = ["# GPT direct versus structured-reasoning review", ""]
    for index, row in enumerate(rows, 1):
        trace = row["structured_trace"]
        parts.extend([
            f"## {index}. {row['question_id']}",
            "",
            f"**Question:** {row['question']}",
            "",
            f"**Gold (shown only after generation):** `{row['gold_output']}`",
            "",
            f"**Direct:** `{row['direct_prediction']}` — MRR {row['direct_mrr']}",
            "",
            f"**Reasoning answer:** `{row['structured_reasoning_prediction']}` — MRR {row['structured_reasoning_mrr']}",
            "",
            f"**Answer shape:** {trace.get('answer_shape', '(v1 did not predict shape)')}",
            "",
            f"**Requested type:** {trace['requested_answer_type']}",
            "",
            f"**Required relation:** {trace['required_relation']}",
            "",
            f"**Constraints:** {', '.join(trace['essential_constraints']) or '(none)' }",
            "",
            "**Candidates:**",
            "",
        ])
        for candidate in trace["candidates"]:
            parts.append(
                f"- `{candidate['text']}` (Resource {candidate['resource_id']}, "
                f"{candidate['decision']}): {candidate['relation_check']} "
                f"{candidate['constraint_check']} "
                f"{candidate.get('boundary_check', '')}".rstrip()
            )
        parts.extend(["", f"**Selection reason:** {trace['selection_reason']}", ""])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument(
        "--reasoning-prompt-version", choices=sorted(REASONING_SYSTEM_PROMPTS), default="v2"
    )
    parser.add_argument(
        "--reuse-direct-cache-root",
        type=Path,
        default=V1_OUTPUT_ROOT / "cache/direct",
        help="Optional cache directory for reusing the unchanged direct baseline.",
    )
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--question-limit", type=int, default=0, help="0 means all remaining questions")
    parser.add_argument("--direct-max-tokens", type=int, default=80)
    parser.add_argument("--reasoning-max-tokens", type=int, default=1200)
    parser.add_argument("--max-validation-retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--rate-limit-initial-sleep-seconds", type=float, default=15.0)
    parser.add_argument("--rate-limit-max-sleep-seconds", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.question_offset < 0 or args.question_limit < 0:
        raise ValueError("question offset and limit must be non-negative")
    all_rows = load_rows(args.source)
    stop = None if args.question_limit == 0 else args.question_offset + args.question_limit
    selected = all_rows[args.question_offset:stop]
    if not selected:
        raise ValueError("No questions selected")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reasoning_prompt = REASONING_SYSTEM_PROMPTS[args.reasoning_prompt_version]
    experiment_version = (
        EXPERIMENT_VERSION_V2
        if args.reasoning_prompt_version == "v2"
        else EXPERIMENT_VERSION_V1
    )
    manifest = {
        "experiment_version": experiment_version,
        "reasoning_prompt_version": args.reasoning_prompt_version,
        "source": str(args.source),
        "source_question_count": len(all_rows),
        "selected_question_count": len(selected),
        "question_offset": args.question_offset,
        "question_limit": args.question_limit,
        "model": args.model,
        "temperature": 0,
        "direct_max_tokens": args.direct_max_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "direct_prompt_sha256": digest(DIRECT_SYSTEM_PROMPT),
        "reasoning_prompt_sha256": digest(reasoning_prompt),
        "reuse_direct_cache_root": str(args.reuse_direct_cache_root),
        "gold_visible_to_model": False,
        "paired_same_input": True,
        "scoring_backend": "bioasq_java",
    }
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        print("\nDIRECT SYSTEM PROMPT\n" + DIRECT_SYSTEM_PROMPT)
        print("\nSTRUCTURED SYSTEM PROMPT\n" + reasoning_prompt)
        print("\nSHARED USER PROMPT\n" + build_user_prompt(selected[0]))
        return

    api_key = read_api_key(args.api_key_file)
    generator = PairedGenerator(args, api_key)
    raw_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(selected, 1):
        direct = generator.generate_arm(source_row, "direct")
        structured = generator.generate_arm(source_row, "structured_reasoning")
        trace = structured["response"]
        direct_prediction = direct["response"]
        selected_answer = trace["selected_answer"].strip()
        resource_pairs = resources_from_row(source_row)
        direct_items = parse_prediction_items(direct_prediction, "factoid")
        gold_items = parse_prediction_items(str(source_row.get("output") or ""), "factoid")
        raw_rows.append({
            "question_id": source_row["id"],
            "question": str(source_row.get("input_1") or "").strip(),
            "gold_output": source_row.get("output"),
            "direct_prediction": direct_prediction,
            "structured_reasoning_prediction": (
                f"[BE]{selected_answer}[EE]" if selected_answer else ""
            ),
            "structured_trace": trace,
            "structured_abstained": not bool(selected_answer),
            "gold_weakly_extractive": any(
                is_weakly_extractive(item, resource_pairs) for item in gold_items
            ),
            "direct_weakly_extractive": (
                len(direct_items) == 1 and is_weakly_extractive(direct_items[0], resource_pairs)
            ),
            "structured_weakly_extractive": is_weakly_extractive(selected_answer, resource_pairs),
            "direct_answer_word_count": (
                len(direct_items[0].split()) if len(direct_items) == 1 else 0
            ),
            "structured_answer_word_count": len(selected_answer.split()),
            "gold_min_answer_word_count": min(
                (len(item.split()) for item in gold_items), default=0
            ),
            "direct_origin": direct["origin"],
            "structured_origin": structured["origin"],
        })
        if index == 1 or index % args.progress_every == 0 or index == len(selected):
            print(
                f"{index} / {len(selected)} | new API calls: {generator.new_api_calls} "
                f"| rate limits: {generator.rate_limit_hits}"
            )

    examples_all = load_gold_examples([args.source])
    selected_ids = {row["question_id"] for row in raw_rows}
    examples = {qid: example for qid, example in examples_all.items() if qid in selected_ids}
    direct_metrics, direct_by_id = score_arm("direct", raw_rows, examples, output_root)
    reasoning_metrics, reasoning_by_id = score_arm(
        "structured_reasoning", raw_rows, examples, output_root
    )
    for row in raw_rows:
        qid = row["question_id"]
        for arm, scores in (
            ("direct", direct_by_id[qid]),
            ("structured_reasoning", reasoning_by_id[qid]),
        ):
            row[f"{arm}_mrr"] = float(scores.get("mrr", 0.0))
            row[f"{arm}_strict_accuracy"] = float(scores.get("strict_accuracy", 0.0))
            row[f"{arm}_lenient_accuracy"] = float(scores.get("lenient_accuracy", 0.0))
        direct_ok = row["direct_mrr"] > 0
        reasoning_ok = row["structured_reasoning_mrr"] > 0
        row["transition"] = (
            "gain" if not direct_ok and reasoning_ok
            else "loss" if direct_ok and not reasoning_ok
            else "both_correct" if direct_ok and reasoning_ok
            else "both_wrong"
        )
        row["mrr_delta"] = row["structured_reasoning_mrr"] - row["direct_mrr"]

    bootstrap = paired_bootstrap_delta(raw_rows, "direct_mrr", "structured_reasoning_mrr")
    transition_counts = Counter(row["transition"] for row in raw_rows)
    support_subset_metrics: dict[str, Any] = {}
    for subset_name, expected_support in (
        ("gold_weakly_extractive", True),
        ("gold_not_weakly_extractive", False),
    ):
        subset_rows = [
            row for row in raw_rows
            if bool(row["gold_weakly_extractive"]) is expected_support
        ]
        if not subset_rows:
            support_subset_metrics[subset_name] = {"question_count": 0}
            continue
        subset_direct, _ = score_arm(
            "direct", subset_rows, examples, output_root / "support_subsets" / subset_name
        )
        subset_reasoning, _ = score_arm(
            "structured_reasoning",
            subset_rows,
            examples,
            output_root / "support_subsets" / subset_name,
        )
        support_subset_metrics[subset_name] = {
            "question_count": len(subset_rows),
            "direct": subset_direct,
            "structured_reasoning": subset_reasoning,
            "structured_minus_direct_mrr": (
                float(subset_reasoning["mrr"]) - float(subset_direct["mrr"])
            ),
            "transition_counts": dict(Counter(row["transition"] for row in subset_rows)),
        }
    summary = {
        **manifest,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "direct": direct_metrics,
        "structured_reasoning": reasoning_metrics,
        "structured_minus_direct": {
            "mrr": float(reasoning_metrics["mrr"]) - float(direct_metrics["mrr"]),
            "strict_accuracy": (
                float(reasoning_metrics["strict_accuracy"])
                - float(direct_metrics["strict_accuracy"])
            ),
            "lenient_accuracy": (
                float(reasoning_metrics["lenient_accuracy"])
                - float(direct_metrics["lenient_accuracy"])
            ),
        },
        "paired_mrr_bootstrap": bootstrap,
        "transition_counts": dict(transition_counts),
        "support_subset_metrics": support_subset_metrics,
        "answer_shape_counts": dict(Counter(
            row["structured_trace"].get("answer_shape", "not_recorded")
            for row in raw_rows
        )),
        "answer_length_words": {
            "direct_mean": sum(row["direct_answer_word_count"] for row in raw_rows) / len(raw_rows),
            "structured_mean": sum(row["structured_answer_word_count"] for row in raw_rows) / len(raw_rows),
            "gold_min_alias_mean": sum(row["gold_min_answer_word_count"] for row in raw_rows) / len(raw_rows),
            "structured_longer_than_direct_count": sum(
                row["structured_answer_word_count"] > row["direct_answer_word_count"]
                for row in raw_rows
            ),
            "structured_longer_than_gold_min_count": sum(
                row["structured_answer_word_count"] > row["gold_min_answer_word_count"]
                for row in raw_rows
            ),
        },
        "direct_weakly_extractive_count": sum(bool(row["direct_weakly_extractive"]) for row in raw_rows),
        "structured_weakly_extractive_count": sum(
            bool(row["structured_weakly_extractive"]) for row in raw_rows
        ),
        "structured_abstention_count": sum(
            bool(row["structured_abstained"]) for row in raw_rows
        ),
        "gold_weakly_extractive_count": sum(
            bool(row["gold_weakly_extractive"]) for row in raw_rows
        ),
        "gold_not_weakly_extractive_count": sum(
            not bool(row["gold_weakly_extractive"]) for row in raw_rows
        ),
        "new_api_calls": generator.new_api_calls,
        "rate_limit_hits": generator.rate_limit_hits,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "per_question_comparison.json", raw_rows)
    write_jsonl(output_root / "per_question_comparison.jsonl", raw_rows)
    write_csv(output_root / "per_question_comparison.csv", raw_rows)
    write_review_markdown(output_root / "reasoning_review.md", raw_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
