from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.bioasq_format import normalize_for_match
from src.utility.data import build_resources, clean_text, load_prepared_records
from src.utility.eval_dataset import render_manual_chat

from .common import load_json_records, read_json, safe_ratio, write_jsonl
from .normalize_set_answers import (
    normalize_answer_surface,
    normalized_surface_variants,
    parse_list_output,
)
from .schemas import (
    CandidateBankRecord,
    GoldGroup,
    LABEL_GOLD_MATCHED,
    LABEL_METRIC_NEGATIVE,
    MatchResult,
    MatchedCandidate,
    MatchedResponse,
    ParsedListOutput,
    QuestionExample,
    ResponseMetrics,
    SemanticListItem,
    to_jsonable,
)

SHORT_LONG_PATTERN = re.compile(
    r"(?P<long>[A-Za-z][A-Za-z0-9α-ωΑ-Ω/\- ,]{2,100}?)\s*\((?P<short>[A-Za-zα-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω/\-]{1,20})\)"
)
LONG_SHORT_PATTERN = re.compile(
    r"(?P<short>[A-Za-zα-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω/\-]{1,20})\s*\((?P<long>[A-Za-z][A-Za-z0-9α-ωΑ-Ω/\- ,]{2,100}?)\)"
)
TRAILING_GENERIC_SUFFIXES = (
    " gene",
    " genes",
)
TRAILING_FORMULATION_SUFFIXES = (
    " hydrochloride",
    " hydrobromide",
    " monohydrate",
    " dihydrate",
    " trihydrate",
    " hcl",
)
TRAILING_ANATOMY_SUFFIXES = (
    " gland",
)
LEADING_DESCRIPTOR_PREFIXES = (
    "human ",
    "mouse ",
    "murine ",
    "rat ",
    "zebrafish ",
    "long ncrna ",
    "lncrna ",
    "long noncoding rna ",
    "long non coding rna ",
    "noncoding rna ",
    "non coding rna ",
)
LEADING_FORMULATION_PREFIXES = (
    "sr ",
    "xr ",
    "er ",
    "cr ",
    "dr ",
    "extended release ",
    "sustained release ",
    "immediate release ",
    "delayed release ",
)
SPELLING_TOKEN_EQUIVALENTS = {
    "anaemia": "anemia",
    "apnoea": "apnea",
    "apnoeic": "apneic",
    "diarrhoea": "diarrhea",
    "haematopoietic": "hematopoietic",
    "haemoglobin": "hemoglobin",
    "inhibithors": "inhibitors",
    "leukaemia": "leukemia",
    "oesophageal": "esophageal",
    "oestrogen": "estrogen",
    "paediatric": "pediatric",
    "proceedure": "procedure",
    "sulphate": "sulfate",
    "tumour": "tumor",
    "tumours": "tumors",
}
MATCH_TYPE_EXACT_NORMALIZED = "exact_normalized"
MATCH_TYPE_GOLD_ALIAS = "gold_alias"
MATCH_TYPE_EXPLICIT_ALIAS = "explicit_alias"
MATCH_TYPE_EVIDENCE_ACRONYM = "evidence_acronym"
MATCH_TYPE_CONTROLLED_SPELLING = "controlled_spelling_variant"
MATCH_TYPE_CONTROLLED_FORMULATION = "controlled_formulation_variant"
MATCH_TYPE_CONTROLLED_ROMAN_NUMERAL = "controlled_roman_numeral"
MATCH_TYPE_CONTROLLED_MORPHOLOGY = "controlled_morphology"
MATCH_TYPE_NO_MATCH = "no_match"
MATCH_TYPE_PRIORITY = {
    MATCH_TYPE_NO_MATCH: 0,
    MATCH_TYPE_CONTROLLED_MORPHOLOGY: 1,
    MATCH_TYPE_CONTROLLED_ROMAN_NUMERAL: 2,
    MATCH_TYPE_CONTROLLED_FORMULATION: 3,
    MATCH_TYPE_CONTROLLED_SPELLING: 4,
    MATCH_TYPE_EVIDENCE_ACRONYM: 5,
    MATCH_TYPE_EXPLICIT_ALIAS: 6,
    MATCH_TYPE_GOLD_ALIAS: 7,
    MATCH_TYPE_EXACT_NORMALIZED: 8,
}
MATCH_TYPE_CONFIDENCE = {
    MATCH_TYPE_NO_MATCH: 0.0,
    MATCH_TYPE_CONTROLLED_MORPHOLOGY: 0.82,
    MATCH_TYPE_CONTROLLED_ROMAN_NUMERAL: 0.88,
    MATCH_TYPE_CONTROLLED_FORMULATION: 0.9,
    MATCH_TYPE_CONTROLLED_SPELLING: 0.91,
    MATCH_TYPE_EVIDENCE_ACRONYM: 0.92,
    MATCH_TYPE_EXPLICIT_ALIAS: 0.95,
    MATCH_TYPE_GOLD_ALIAS: 0.97,
    MATCH_TYPE_EXACT_NORMALIZED: 1.0,
}
MATCH_CARDINALITY_BONUS = 1_000_000
ROMAN_NUMERAL_TABLE = (
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
)
NO_MATCH_RESULT = MatchResult(
    matched=False,
    match_type=MATCH_TYPE_NO_MATCH,
    confidence=0.0,
)


@dataclass(frozen=True)
class SurfaceMatchProfile:
    base_variants: frozenset[str]
    explicit_alias_variants: frozenset[str]
    evidence_alias_variants: frozenset[str]
    spelling_variants: frozenset[str]
    formulation_variants: frozenset[str]
    roman_variants: frozenset[str]
    morphology_variants: frozenset[str]
    canonical_variants: frozenset[str]


def _match_result(match_type: str) -> MatchResult:
    return MatchResult(
        matched=match_type != MATCH_TYPE_NO_MATCH,
        match_type=match_type,
        confidence=MATCH_TYPE_CONFIDENCE[match_type],
    )


def _match_sort_key(result: MatchResult) -> tuple[int, float]:
    return (
        MATCH_TYPE_PRIORITY.get(result.match_type, 0),
        result.confidence,
    )


def _better_match_result(left: MatchResult, right: MatchResult) -> bool:
    return _match_sort_key(left) > _match_sort_key(right)


def _best_match_result(results: Iterable[MatchResult]) -> MatchResult:
    best = NO_MATCH_RESULT
    for result in results:
        if _better_match_result(result, best):
            best = result
    return best


def _question_evidence_key(question: QuestionExample) -> tuple[str, ...]:
    return tuple(clean_text(resource) for resource in question.evidence if clean_text(resource))


def _question_gold_signature(question: QuestionExample) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple((group.group_id, group.aliases) for group in question.gold_groups)


def _official_exact_normalize(text: str) -> str:
    return normalize_for_match(clean_text(text))


def build_prompt_text(question: QuestionExample) -> str:
    user_parts = [f"Question: {question.question_text}"]
    evidence = [resource for resource in question.evidence if clean_text(resource)]
    if evidence:
        user_parts.append("PubMed resources:")
        for index, resource in enumerate(evidence, start=1):
            user_parts.append(f"Resource {index}:\n{resource}")

    return render_manual_chat(
        messages=[
            {"role": "system", "content": question.instruction},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        chat_template=None,
    ) + "Answer:"


def _normalized_key_variants(normalized: str) -> tuple[str, ...]:
    if not normalized:
        return ()
    variants = [normalized]
    collapsed = normalized.replace(" ", "")
    if collapsed and collapsed != normalized:
        variants.append(collapsed)
    return tuple(dict.fromkeys(variants))


def _int_to_roman(value: int) -> str:
    remaining = value
    parts: list[str] = []
    for numeral_value, numeral_text in ROMAN_NUMERAL_TABLE:
        while remaining >= numeral_value:
            parts.append(numeral_text)
            remaining -= numeral_value
    return "".join(parts)


def _roman_to_int(token: str) -> int | None:
    lowered = clean_text(token).lower()
    if not lowered or any(char not in {"i", "v", "x", "l"} for char in lowered):
        return None

    total = 0
    previous = 0
    roman_values = {"i": 1, "v": 5, "x": 10, "l": 50}
    for char in reversed(lowered):
        value = roman_values[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value

    if total <= 0 or total > 50:
        return None
    if _int_to_roman(total) != lowered:
        return None
    return total


def _iter_roman_normalized_variants(normalized: str) -> tuple[str, ...]:
    tokens = [token for token in clean_text(normalized).split() if token]
    if not tokens:
        return ()

    derived: list[str] = []
    for index, token in enumerate(tokens):
        value = _roman_to_int(token)
        if value is None:
            continue
        updated_tokens = list(tokens)
        updated_tokens[index] = str(value)
        candidate = clean_text(" ".join(updated_tokens))
        if candidate:
            derived.append(candidate)
    return tuple(dict.fromkeys(derived))


def _iter_spelling_normalized_variants(normalized: str) -> tuple[str, ...]:
    tokens = [token for token in clean_text(normalized).split() if token]
    if not tokens:
        return ()

    derived: set[str] = set()
    replaced_any = False
    fully_replaced_tokens = list(tokens)
    for index, token in enumerate(tokens):
        replacement = SPELLING_TOKEN_EQUIVALENTS.get(token)
        if replacement is None:
            continue
        replaced_any = True
        updated_tokens = list(tokens)
        updated_tokens[index] = replacement
        candidate = clean_text(" ".join(updated_tokens))
        if candidate:
            derived.add(candidate)
        fully_replaced_tokens[index] = replacement

    if replaced_any:
        candidate = clean_text(" ".join(fully_replaced_tokens))
        if candidate:
            derived.add(candidate)

    return tuple(sorted(derived))


def _iter_formulation_normalized_variants(normalized: str) -> tuple[str, ...]:
    cleaned = clean_text(normalized)
    if not cleaned:
        return ()

    derived: list[str] = []
    for prefix in LEADING_FORMULATION_PREFIXES:
        if cleaned.startswith(prefix):
            base = clean_text(cleaned[len(prefix) :])
            if base:
                derived.append(base)

    for suffix in TRAILING_FORMULATION_SUFFIXES + TRAILING_ANATOMY_SUFFIXES:
        if cleaned.endswith(suffix):
            base = clean_text(cleaned[: -len(suffix)])
            if base:
                derived.append(base)

    return tuple(dict.fromkeys(derived))


def _iter_inflection_normalized_variants(normalized: str) -> tuple[str, ...]:
    tokens = [token for token in clean_text(normalized).split() if token]
    if not tokens:
        return ()

    derived: set[str] = set()
    for index, token in enumerate(tokens):
        replacement: str | None = None
        if token.endswith("ies") and len(token) > 4:
            replacement = f"{token[:-3]}y"
        elif token.endswith("es") and len(token) > 4:
            replacement = token[:-2]
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            replacement = token[:-1]
        if not replacement or replacement == token:
            continue
        updated_tokens = list(tokens)
        updated_tokens[index] = replacement
        candidate = clean_text(" ".join(updated_tokens))
        if candidate:
            derived.add(candidate)
    return tuple(sorted(derived))


def _looks_like_abbreviation(value: str) -> bool:
    cleaned = clean_text(value)
    if not cleaned or len(cleaned) > 20 or " " in cleaned:
        return False
    letters = [char for char in cleaned if char.isalpha()]
    greek_letters = any("α" <= char <= "ω" or "Α" <= char <= "Ω" for char in cleaned)
    uppercase_letters = sum(1 for char in cleaned if char.isupper())
    return bool(letters) and (
        cleaned.isupper()
        or any(char.isdigit() for char in cleaned)
        or "-" in cleaned
        or greek_letters
        or uppercase_letters >= max(1, len(letters) // 2)
    )


def _iter_alias_pairs(text: str) -> Iterable[tuple[str, str]]:
    cleaned = clean_text(text)
    if not cleaned:
        return ()

    pairs: list[tuple[str, str]] = []
    for pattern in (SHORT_LONG_PATTERN, LONG_SHORT_PATTERN):
        for match in pattern.finditer(cleaned):
            short_form = clean_text(match.group("short"))
            long_form = clean_text(match.group("long"))
            if not _looks_like_abbreviation(short_form):
                continue
            pairs.append((short_form, long_form))
    return tuple(pairs)


def _iter_explicit_alias_forms(text: str) -> tuple[str, ...]:
    cleaned = clean_text(text)
    if not cleaned:
        return ()

    forms: list[str] = []
    without_parentheses = clean_text(re.sub(r"\([^)]*\)", " ", cleaned))
    if without_parentheses and without_parentheses != cleaned:
        forms.append(without_parentheses)

    for short_form, long_form in _iter_alias_pairs(cleaned):
        forms.append(short_form)
        forms.append(long_form)
        numbered_long_form = _derive_numbered_long_form(short_form, long_form)
        if numbered_long_form:
            forms.append(numbered_long_form)
    return tuple(dict.fromkeys(forms))


def _derive_numbered_long_form(short_form: str, long_form: str) -> str | None:
    short_normalized = normalize_answer_surface(short_form)
    long_normalized = normalize_answer_surface(long_form)
    number_match = re.search(r"\b(\d+)\b$", short_normalized)
    if number_match is None:
        return None

    number = number_match.group(1)
    if long_normalized.endswith(number):
        return None
    return clean_text(f"{long_form} {number}")


def _iter_relaxed_normalized_variants(normalized: str) -> Iterable[str]:
    cleaned = clean_text(normalized)
    if not cleaned:
        return ()

    derived: list[str] = []
    for suffix in TRAILING_GENERIC_SUFFIXES:
        if cleaned.endswith(suffix):
            base = clean_text(cleaned[: -len(suffix)])
            if base:
                derived.append(base)

    for prefix in LEADING_DESCRIPTOR_PREFIXES:
        if cleaned.startswith(prefix):
            base = clean_text(cleaned[len(prefix) :])
            if base:
                derived.append(base)

    return tuple(derived)


def _surface_match_variants(
    text: str,
    equivalences: Mapping[str, set[str]],
) -> tuple[str, ...]:
    cleaned = clean_text(text)
    if not cleaned:
        return ()

    raw_surfaces: list[str] = [cleaned]
    without_parentheses = clean_text(re.sub(r"\([^)]*\)", " ", cleaned))
    if without_parentheses and without_parentheses != cleaned:
        raw_surfaces.append(without_parentheses)

    for short_form, long_form in _iter_alias_pairs(cleaned):
        raw_surfaces.append(short_form)
        raw_surfaces.append(long_form)
        numbered_long_form = _derive_numbered_long_form(short_form, long_form)
        if numbered_long_form:
            raw_surfaces.append(numbered_long_form)

    expanded_aliases: set[str] = set()
    pending: list[str] = []
    for surface in raw_surfaces:
        for variant in normalized_surface_variants(surface):
            if variant not in expanded_aliases:
                expanded_aliases.add(variant)
                pending.append(variant)

    while pending:
        current = pending.pop()
        for spelling_variant in _iter_spelling_normalized_variants(current):
            for variant in _normalized_key_variants(spelling_variant):
                if variant not in expanded_aliases:
                    expanded_aliases.add(variant)
                    pending.append(variant)

        for formulation_variant in _iter_formulation_normalized_variants(current):
            for variant in _normalized_key_variants(formulation_variant):
                if variant not in expanded_aliases:
                    expanded_aliases.add(variant)
                    pending.append(variant)

        for inflection_variant in _iter_inflection_normalized_variants(current):
            for variant in _normalized_key_variants(inflection_variant):
                if variant not in expanded_aliases:
                    expanded_aliases.add(variant)
                    pending.append(variant)

        for relaxed_variant in _iter_relaxed_normalized_variants(current):
            for variant in _normalized_key_variants(relaxed_variant):
                if variant not in expanded_aliases:
                    expanded_aliases.add(variant)
                    pending.append(variant)

        for equivalent in equivalences.get(current, set()):
            for variant in _normalized_key_variants(equivalent):
                if variant not in expanded_aliases:
                    expanded_aliases.add(variant)
                    pending.append(variant)

    return tuple(sorted(expanded_aliases))


@lru_cache(maxsize=2048)
def _collect_evidence_alias_equivalences_cached(evidence_key: tuple[str, ...]) -> dict[str, set[str]]:
    equivalences: dict[str, set[str]] = defaultdict(set)
    for text in evidence_key:
        for short_form, long_form in _iter_alias_pairs(text):
            candidate_forms = [short_form, long_form]
            numbered_long_form = _derive_numbered_long_form(short_form, long_form)
            if numbered_long_form:
                candidate_forms.append(numbered_long_form)

            short_variants = normalized_surface_variants(short_form)
            long_variants: set[str] = set()
            for form in candidate_forms[1:]:
                long_variants.update(normalized_surface_variants(form))
            if not short_variants or not long_variants:
                continue

            for left in short_variants:
                for right in long_variants:
                    if left == right:
                        continue
                    equivalences[left].add(right)
                    equivalences[right].add(left)
    return equivalences


def _collect_evidence_alias_equivalences(evidence: Sequence[str]) -> dict[str, set[str]]:
    evidence_key = tuple(clean_text(resource) for resource in evidence if clean_text(resource))
    return _collect_evidence_alias_equivalences_cached(evidence_key)


def surface_match_aliases(surface: str, evidence: Sequence[str]) -> tuple[str, ...]:
    evidence_key = tuple(clean_text(resource) for resource in evidence if clean_text(resource))
    return _surface_match_variants_for_evidence(surface, evidence_key)


@lru_cache(maxsize=65536)
def _surface_match_variants_for_evidence(
    text: str,
    evidence_key: tuple[str, ...],
) -> tuple[str, ...]:
    equivalences = _collect_evidence_alias_equivalences_cached(evidence_key)
    return _surface_match_variants(text, equivalences)


@lru_cache(maxsize=65536)
def _build_surface_match_profile_cached(
    text: str,
    evidence_key: tuple[str, ...],
) -> SurfaceMatchProfile:
    cleaned = clean_text(text)
    if not cleaned:
        empty = frozenset()
        return SurfaceMatchProfile(
            base_variants=empty,
            explicit_alias_variants=empty,
            evidence_alias_variants=empty,
            spelling_variants=empty,
            formulation_variants=empty,
            roman_variants=empty,
            morphology_variants=empty,
            canonical_variants=empty,
        )

    equivalences = _collect_evidence_alias_equivalences_cached(evidence_key)
    base_variants = set(normalized_surface_variants(cleaned))

    explicit_alias_variants: set[str] = set()
    for alias_form in _iter_explicit_alias_forms(cleaned):
        explicit_alias_variants.update(normalized_surface_variants(alias_form))
    explicit_alias_variants -= base_variants

    spelling_variants: set[str] = set()
    for variant in base_variants | explicit_alias_variants:
        spelling_variants.update(_iter_spelling_normalized_variants(variant))
    spelling_variants -= base_variants | explicit_alias_variants

    formulation_variants: set[str] = set()
    for variant in base_variants | explicit_alias_variants | spelling_variants:
        formulation_variants.update(_iter_formulation_normalized_variants(variant))
    formulation_variants -= base_variants | explicit_alias_variants | spelling_variants

    inflection_variants: set[str] = set()
    for variant in base_variants | explicit_alias_variants | spelling_variants | formulation_variants:
        inflection_variants.update(_iter_inflection_normalized_variants(variant))
    inflection_variants -= base_variants | explicit_alias_variants | spelling_variants | formulation_variants

    morphology_variants: set[str] = set(inflection_variants)
    seen_morphology = set(base_variants | explicit_alias_variants | spelling_variants | formulation_variants | inflection_variants)
    pending_morphology = list(seen_morphology)
    while pending_morphology:
        current = pending_morphology.pop()
        for relaxed_variant in _iter_relaxed_normalized_variants(current):
            for variant in _normalized_key_variants(relaxed_variant):
                if variant in seen_morphology:
                    continue
                seen_morphology.add(variant)
                morphology_variants.add(variant)
                pending_morphology.append(variant)

    evidence_alias_variants: set[str] = set()
    seen_evidence = set(base_variants | explicit_alias_variants | spelling_variants | formulation_variants | inflection_variants)
    pending_evidence = list(seen_evidence)
    while pending_evidence:
        current = pending_evidence.pop()
        for equivalent in equivalences.get(current, set()):
            for variant in _normalized_key_variants(equivalent):
                if variant in seen_evidence:
                    continue
                seen_evidence.add(variant)
                evidence_alias_variants.add(variant)
                pending_evidence.append(variant)

    roman_variants: set[str] = set()
    for variant in base_variants | explicit_alias_variants | spelling_variants | formulation_variants | inflection_variants | morphology_variants:
        roman_variants.update(_iter_roman_normalized_variants(variant))
    roman_variants -= base_variants | explicit_alias_variants | spelling_variants | formulation_variants | inflection_variants | morphology_variants

    canonical_variants = frozenset(
        base_variants
        | explicit_alias_variants
        | evidence_alias_variants
        | spelling_variants
        | formulation_variants
        | inflection_variants
        | roman_variants
        | morphology_variants
    )
    return SurfaceMatchProfile(
        base_variants=frozenset(base_variants),
        explicit_alias_variants=frozenset(explicit_alias_variants),
        evidence_alias_variants=frozenset(evidence_alias_variants),
        spelling_variants=frozenset(spelling_variants),
        formulation_variants=frozenset(formulation_variants),
        roman_variants=frozenset(roman_variants),
        morphology_variants=frozenset(morphology_variants),
        canonical_variants=canonical_variants,
    )


def _build_surface_match_profile(
    text: str,
    question: QuestionExample,
) -> SurfaceMatchProfile:
    return _build_surface_match_profile_cached(text, _question_evidence_key(question))


def _direct_answer_equivalent_from_profiles(
    left_profile: SurfaceMatchProfile,
    right_profile: SurfaceMatchProfile,
) -> MatchResult:
    if left_profile.base_variants & right_profile.base_variants:
        return _match_result(MATCH_TYPE_EXACT_NORMALIZED)

    left_explicit_space = left_profile.base_variants | left_profile.explicit_alias_variants
    right_explicit_space = right_profile.base_variants | right_profile.explicit_alias_variants
    if (
        left_profile.explicit_alias_variants & right_explicit_space
        or right_profile.explicit_alias_variants & left_explicit_space
    ):
        return _match_result(MATCH_TYPE_EXPLICIT_ALIAS)

    left_spelling_space = left_explicit_space | left_profile.spelling_variants
    right_spelling_space = right_explicit_space | right_profile.spelling_variants
    if (
        left_spelling_space & right_spelling_space
        and (left_profile.spelling_variants or right_profile.spelling_variants)
    ):
        return _match_result(MATCH_TYPE_CONTROLLED_SPELLING)

    left_formulation_space = left_spelling_space | left_profile.formulation_variants
    right_formulation_space = right_spelling_space | right_profile.formulation_variants
    if (
        left_formulation_space & right_formulation_space
        and (left_profile.formulation_variants or right_profile.formulation_variants)
    ):
        return _match_result(MATCH_TYPE_CONTROLLED_FORMULATION)

    left_evidence_space = left_formulation_space | left_profile.evidence_alias_variants
    right_evidence_space = right_formulation_space | right_profile.evidence_alias_variants
    if (
        left_evidence_space & right_evidence_space
        and (left_profile.evidence_alias_variants or right_profile.evidence_alias_variants)
    ):
        return _match_result(MATCH_TYPE_EVIDENCE_ACRONYM)

    left_roman_space = left_formulation_space | left_profile.roman_variants
    right_roman_space = right_formulation_space | right_profile.roman_variants
    if (
        left_roman_space & right_roman_space
        and (left_profile.roman_variants or right_profile.roman_variants)
    ):
        return _match_result(MATCH_TYPE_CONTROLLED_ROMAN_NUMERAL)

    left_morphology_space = left_formulation_space | left_profile.morphology_variants
    right_morphology_space = right_formulation_space | right_profile.morphology_variants
    if (
        left_morphology_space & right_morphology_space
        and (left_profile.morphology_variants or right_profile.morphology_variants)
    ):
        return _match_result(MATCH_TYPE_CONTROLLED_MORPHOLOGY)

    return NO_MATCH_RESULT


def _direct_answer_equivalent(
    left: str,
    right: str,
    question: QuestionExample,
) -> MatchResult:
    return _direct_answer_equivalent_from_profiles(
        _build_surface_match_profile(left, question),
        _build_surface_match_profile(right, question),
    )


@lru_cache(maxsize=65536)
def _surface_gold_group_support_cached(
    text: str,
    evidence_key: tuple[str, ...],
    gold_signature: tuple[tuple[int, tuple[str, ...]], ...],
) -> tuple[tuple[int, MatchResult], ...]:
    profile = _build_surface_match_profile_cached(text, evidence_key)
    supports: list[tuple[int, MatchResult]] = []
    for group_id, aliases in gold_signature:
        best_result = _best_match_result(
            _direct_answer_equivalent_from_profiles(
                profile,
                _build_surface_match_profile_cached(alias, evidence_key),
            )
            for alias in aliases
        )
        if best_result.matched:
            supports.append((group_id, best_result))
    return tuple(supports)


def _surface_gold_group_support(
    text: str,
    question: QuestionExample,
) -> dict[int, MatchResult]:
    return dict(
        _surface_gold_group_support_cached(
            text,
            _question_evidence_key(question),
            _question_gold_signature(question),
        )
    )


@lru_cache(maxsize=131072)
def _answer_equivalent_cached(
    left: str,
    right: str,
    evidence_key: tuple[str, ...],
    gold_signature: tuple[tuple[int, tuple[str, ...]], ...],
) -> MatchResult:
    direct_result = _direct_answer_equivalent_from_profiles(
        _build_surface_match_profile_cached(left, evidence_key),
        _build_surface_match_profile_cached(right, evidence_key),
    )
    if direct_result.match_type == MATCH_TYPE_EXACT_NORMALIZED:
        return direct_result

    best_result = direct_result
    left_support = dict(_surface_gold_group_support_cached(left, evidence_key, gold_signature))
    right_support = dict(_surface_gold_group_support_cached(right, evidence_key, gold_signature))
    for group_id, left_result in left_support.items():
        right_result = right_support.get(group_id)
        if right_result is None:
            continue
        gold_result = MatchResult(
            matched=True,
            match_type=MATCH_TYPE_GOLD_ALIAS,
            confidence=min(left_result.confidence, right_result.confidence),
        )
        if _better_match_result(gold_result, best_result):
            best_result = gold_result
    return best_result


def answer_equivalent(
    left: str,
    right: str,
    question: QuestionExample,
) -> MatchResult:
    cleaned_left = clean_text(left)
    cleaned_right = clean_text(right)
    if not cleaned_left or not cleaned_right:
        return NO_MATCH_RESULT
    return _answer_equivalent_cached(
        cleaned_left,
        cleaned_right,
        _question_evidence_key(question),
        _question_gold_signature(question),
    )


def _expand_gold_group_aliases(
    aliases: Sequence[str],
    equivalences: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    base_aliases: list[str] = []
    expanded_aliases: set[str] = set()

    for alias in aliases:
        for variant in _surface_match_variants(alias, equivalences):
            if variant not in expanded_aliases:
                expanded_aliases.add(variant)
        normalized = normalize_answer_surface(alias)
        if normalized and normalized not in base_aliases:
            base_aliases.append(normalized)

    return tuple(base_aliases), tuple(sorted(expanded_aliases))


def _normalized_phrase_in_normalized_text(phrase: str, normalized_text: str) -> bool:
    normalized_phrase = normalize_answer_surface(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _gold_alias_supported_by_evidence(aliases: Sequence[str], evidence: Sequence[str]) -> bool:
    normalized_evidence = normalize_answer_surface(" ".join(clean_text(resource) for resource in evidence))
    return any(_normalized_phrase_in_normalized_text(alias, normalized_evidence) for alias in aliases)


def _gold_groups_from_exact_answer(
    exact_answer: Any,
    evidence: Sequence[str],
    gold_support_policy: str = "all",
) -> tuple[GoldGroup, ...]:
    groups: list[GoldGroup] = []
    if not isinstance(exact_answer, list):
        return ()

    equivalences = _collect_evidence_alias_equivalences(evidence)
    for group_id, item in enumerate(exact_answer):
        aliases: list[str] = []
        if isinstance(item, list):
            aliases = [clean_text(alias) for alias in item if clean_text(alias)]
        else:
            value = clean_text(item)
            if value:
                aliases = [value]

        if not aliases:
            continue
        if gold_support_policy == "snippet" and not _gold_alias_supported_by_evidence(aliases, evidence):
            continue

        normalized_aliases, match_normalized_aliases = _expand_gold_group_aliases(aliases, equivalences)
        gold_group = GoldGroup(
            group_id=group_id,
            aliases=tuple(aliases),
            normalized_aliases=normalized_aliases,
            match_normalized_aliases=match_normalized_aliases,
            canonical_alias=aliases[0],
        )
        groups.append(gold_group)
    return tuple(groups)


def _gold_groups_from_prepared_output(
    output: str,
    evidence: Sequence[str],
) -> tuple[GoldGroup, ...]:
    parsed = parse_list_output(output, allow_fallback_split=True)
    equivalences = _collect_evidence_alias_equivalences(evidence)

    groups: list[GoldGroup] = []
    for group_id, item in enumerate(parsed.items):
        normalized_aliases, match_normalized_aliases = _expand_gold_group_aliases((item,), equivalences)
        if not normalized_aliases:
            continue
        groups.append(
            GoldGroup(
                group_id=group_id,
                aliases=(item,),
                normalized_aliases=normalized_aliases,
                match_normalized_aliases=match_normalized_aliases,
                canonical_alias=item,
            )
        )
    return tuple(groups)


def load_question_examples(
    paths: Sequence[str],
    dataset_name: str,
    max_resources: int = 3,
    max_resource_chars: int = 1200,
    gold_support_policy: str = "all",
) -> dict[str, QuestionExample]:
    if gold_support_policy not in {"all", "snippet"}:
        raise ValueError(f"Unsupported gold_support_policy: {gold_support_policy}")

    questions_by_id: dict[str, QuestionExample] = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = read_json(path)
        if isinstance(payload, list):
            rows = load_prepared_records(path)
            for row in rows:
                if clean_text(row.get("type", "")).lower() != "list":
                    continue
                question_id = clean_text(row.get("id", ""))
                if not question_id or question_id in questions_by_id:
                    continue
                question = QuestionExample(
                    dataset=dataset_name,
                    question_id=question_id,
                    question_type="list",
                    question_text=clean_text(row.get("input_1", "")),
                    instruction=clean_text(row.get("instruction", "")) or QUESTION_INSTRUCTIONS["list"],
                    evidence=tuple(
                        clean_text(row.get(key, ""))
                        for key in ("input_2", "input_3", "input_4")
                        if clean_text(row.get(key, ""))
                    ),
                    gold_groups=(),
                    source_path=str(path),
                )
                question = QuestionExample(
                    dataset=question.dataset,
                    question_id=question.question_id,
                    question_type=question.question_type,
                    question_text=question.question_text,
                    instruction=question.instruction,
                    evidence=question.evidence,
                    gold_groups=_gold_groups_from_prepared_output(clean_text(row.get("output", "")), question.evidence),
                    source_path=question.source_path,
                )
                if question.gold_groups:
                    questions_by_id[question.question_id] = question
            continue

        raw_questions = payload.get("questions") if isinstance(payload, Mapping) else None
        if not isinstance(raw_questions, list):
            raise ValueError(f"Unsupported question input format: {path}")

        for question_row in raw_questions:
            if not isinstance(question_row, Mapping):
                continue
            if clean_text(question_row.get("type", "")).lower() != "list":
                continue
            question_id = clean_text(question_row.get("id", ""))
            if not question_id or question_id in questions_by_id:
                continue

            resources = build_resources(
                question_row,
                max_resources=max_resources,
                max_resource_chars=max_resource_chars,
            )
            question = QuestionExample(
                dataset=dataset_name,
                question_id=question_id,
                question_type="list",
                question_text=clean_text(question_row.get("body", "")),
                instruction=QUESTION_INSTRUCTIONS["list"],
                evidence=tuple(resource for resource in resources if clean_text(resource)),
                gold_groups=_gold_groups_from_exact_answer(
                    question_row.get("exact_answer"),
                    tuple(resource for resource in resources if clean_text(resource)),
                    gold_support_policy=gold_support_policy,
                ),
                source_path=str(path),
            )
            if question.gold_groups:
                questions_by_id[question.question_id] = question
    return questions_by_id


def load_candidate_bank_records(
    paths: Sequence[str],
    questions_by_id: Mapping[str, QuestionExample],
    dataset_name: str,
    default_generator_checkpoint: str | None = None,
) -> list[CandidateBankRecord]:
    records: list[CandidateBankRecord] = []
    per_question_counters: defaultdict[str, int] = defaultdict(int)

    for raw_path in paths:
        path = Path(raw_path)
        for row in load_json_records(path):
            question_type = clean_text(row.get("question_type") or row.get("type") or "list").lower()
            if question_type != "list":
                continue

            question_id = clean_text(row.get("question_id") or row.get("id"))
            question = questions_by_id.get(question_id)
            if question is None:
                continue

            if "sample_id" in row and row.get("sample_id") not in {None, ""}:
                sample_id = int(row["sample_id"])
            else:
                sample_id = per_question_counters[question_id]
                per_question_counters[question_id] += 1

            prompt = clean_text(row.get("prompt"))
            if not prompt:
                prompt = build_prompt_text(question)

            evidence_value = row.get("evidence")
            evidence = tuple(question.evidence)
            if isinstance(evidence_value, Sequence) and not isinstance(evidence_value, (str, bytes)):
                evidence = tuple(clean_text(item) for item in evidence_value if clean_text(item))

            raw_output = clean_text(row.get("raw_output") or row.get("prediction"))
            if not raw_output:
                continue

            response_id = clean_text(row.get("response_id"))
            if not response_id:
                legacy_bank_id = row.get("bank_id")
                if legacy_bank_id not in {None, ""}:
                    response_id = f"{question_id}-legacybank{legacy_bank_id}-sample{sample_id}"
                else:
                    response_id = f"{question_id}-sample{sample_id}"

            generated_token_count = row.get("generated_token_count")
            token_count = int(generated_token_count) if isinstance(generated_token_count, int) else None
            generator_checkpoint = clean_text(
                row.get("generator_checkpoint")
                or row.get("model_ref")
                or row.get("model")
                or default_generator_checkpoint
                or path.stem
            ) or None

            records.append(
                CandidateBankRecord(
                    dataset=clean_text(row.get("dataset") or dataset_name) or dataset_name,
                    question_id=question_id,
                    sample_id=sample_id,
                    prompt=prompt,
                    question_text=question.question_text,
                    evidence=evidence,
                    raw_output=raw_output,
                    generated_token_count=token_count,
                    generator_checkpoint=generator_checkpoint,
                    response_id=response_id,
                    source_path=str(path),
                    prompt_instruction=clean_text(row.get("prompt_instruction")) or question.instruction,
                )
            )
    return records


@lru_cache(maxsize=65536)
def _official_gold_group_aliases_cached(
    question: QuestionExample,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    groups: list[tuple[int, tuple[str, ...]]] = []
    for group in question.gold_groups:
        normalized_aliases = tuple(
            dict.fromkeys(
                normalized
                for normalized in (_official_exact_normalize(alias) for alias in group.aliases)
                if normalized
            )
        )
        groups.append((group.group_id, normalized_aliases))
    return tuple(groups)


@lru_cache(maxsize=65536)
def _match_official_items_cached(
    question: QuestionExample,
    items_key: tuple[str, ...],
) -> tuple[tuple[MatchedCandidate, ...], tuple[int, ...]]:
    if not items_key:
        return (), ()

    gold_groups = _official_gold_group_aliases_cached(question)
    duplicate_counts = Counter(
        normalized
        for normalized in (_official_exact_normalize(raw_item) for raw_item in items_key)
        if normalized
    )

    matched_candidates: list[MatchedCandidate] = []
    matched_gold_group_ids: list[int] = []
    used_gold_group_ids: set[int] = set()
    for item_index, raw_item in enumerate(items_key):
        surface = clean_text(raw_item)
        normalized = _official_exact_normalize(surface)
        if not surface or not normalized:
            continue

        matched_gold_group_id: int | None = None
        for group_id, group_aliases in gold_groups:
            if group_id in used_gold_group_ids:
                continue
            if normalized not in set(group_aliases):
                continue
            matched_gold_group_id = group_id
            used_gold_group_ids.add(group_id)
            matched_gold_group_ids.append(group_id)
            break

        matched_candidates.append(
            MatchedCandidate(
                surface=surface,
                normalized=normalized,
                matched_gold_group_id=matched_gold_group_id,
                label=LABEL_GOLD_MATCHED if matched_gold_group_id is not None else LABEL_METRIC_NEGATIVE,
                match_type=MATCH_TYPE_EXACT_NORMALIZED if matched_gold_group_id is not None else None,
                match_confidence=1.0 if matched_gold_group_id is not None else 0.0,
                first_index=item_index,
                duplicate_count=duplicate_counts[normalized],
            )
        )

    return tuple(matched_candidates), tuple(matched_gold_group_ids)


@lru_cache(maxsize=65536)
def _dedupe_semantic_items_for_question_cached(
    question: QuestionExample,
    items_key: tuple[str, ...],
) -> tuple[SemanticListItem, ...]:
    indexed_items: list[tuple[int, str, str]] = []
    for index, raw_item in enumerate(items_key):
        surface = clean_text(raw_item)
        normalized = normalize_answer_surface(surface)
        if not surface or not normalized:
            continue
        indexed_items.append((index, surface, normalized))

    if not indexed_items:
        return ()

    clusters: list[list[tuple[int, str, str]]] = []
    for item in indexed_items:
        for cluster in clusters:
            representative = min(cluster, key=lambda entry: entry[0])
            if not answer_equivalent(item[1], representative[1], question).matched:
                continue
            if all(answer_equivalent(item[1], member[1], question).matched for member in cluster):
                cluster.append(item)
                break
        else:
            clusters.append([item])

    semantic_items: list[SemanticListItem] = []
    for members in sorted(clusters, key=lambda entries: min(entry[0] for entry in entries)):
        representative = min(members, key=lambda entry: entry[0])
        semantic_items.append(
            SemanticListItem(
                surface=representative[1],
                normalized=representative[2],
                first_index=representative[0],
                duplicate_count=len(members),
            )
        )
    return tuple(semantic_items)


def dedupe_semantic_items_for_question(
    items: Sequence[str],
    question: QuestionExample,
) -> list[SemanticListItem]:
    return list(_dedupe_semantic_items_for_question_cached(question, tuple(items)))


@lru_cache(maxsize=65536)
def _semantic_set_key_for_question_cached(
    question: QuestionExample,
    items_key: tuple[str, ...],
) -> tuple[str, ...]:
    semantic_items = _dedupe_semantic_items_for_question_cached(question, items_key)
    matched_candidates, _matched_gold_group_ids = _match_semantic_items_cached(question, semantic_items)
    keys: list[str] = []
    for candidate in matched_candidates:
        if candidate.matched_gold_group_id is not None:
            keys.append(f"gold:{candidate.matched_gold_group_id}")
        else:
            keys.append(f"surface:{candidate.normalized}")
    return tuple(sorted(keys))


def semantic_set_key_for_question(
    question: QuestionExample,
    items: Sequence[str],
) -> tuple[str, ...]:
    return _semantic_set_key_for_question_cached(question, tuple(items))


@lru_cache(maxsize=65536)
def _best_surface_semantic_match_cached(
    question: QuestionExample,
    surface: str,
    items_key: tuple[str, ...],
) -> MatchResult | None:
    best_result = NO_MATCH_RESULT
    for item in items_key:
        result = answer_equivalent(surface, item, question)
        if _better_match_result(result, best_result):
            best_result = result
    return best_result if best_result.matched else None


def best_surface_semantic_match(
    question: QuestionExample,
    surface: str,
    items: Sequence[str],
) -> MatchResult | None:
    return _best_surface_semantic_match_cached(question, clean_text(surface), tuple(items))


def _match_weight(result: MatchResult) -> int:
    if not result.matched:
        return 0
    return (
        MATCH_CARDINALITY_BONUS
        + MATCH_TYPE_PRIORITY[result.match_type] * 1_000
        + int(round(result.confidence * 100))
    )


def _hungarian_maximize(weights: Sequence[Sequence[int]]) -> list[int]:
    size = len(weights)
    if size == 0:
        return []

    max_weight = max((max(row, default=0) for row in weights), default=0)
    costs = [[max_weight - weight for weight in row] for row in weights]
    potentials_row = [0] * (size + 1)
    potentials_col = [0] * (size + 1)
    matching = [0] * (size + 1)
    predecessor = [0] * (size + 1)

    for row in range(1, size + 1):
        matching[0] = row
        column = 0
        min_reduced_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)

        while True:
            used[column] = True
            current_row = matching[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][candidate_column - 1]
                    - potentials_row[current_row]
                    - potentials_col[candidate_column]
                )
                if reduced_cost < min_reduced_cost[candidate_column]:
                    min_reduced_cost[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if min_reduced_cost[candidate_column] < delta:
                    delta = min_reduced_cost[candidate_column]
                    next_column = candidate_column

            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_row[matching[candidate_column]] += delta
                    potentials_col[candidate_column] -= delta
                else:
                    min_reduced_cost[candidate_column] -= delta

            column = next_column
            if matching[column] == 0:
                break

        while True:
            previous_column = predecessor[column]
            matching[column] = matching[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        if matching[column] != 0:
            assignment[matching[column] - 1] = column - 1
    return assignment


@lru_cache(maxsize=65536)
def _match_semantic_items_cached(
    question: QuestionExample,
    semantic_items_key: tuple[SemanticListItem, ...],
) -> tuple[tuple[MatchedCandidate, ...], tuple[int, ...]]:
    if not semantic_items_key:
        return (), ()

    gold_groups = list(question.gold_groups)
    size = max(len(semantic_items_key), len(gold_groups))
    weights = [[0 for _column in range(size)] for _row in range(size)]
    edge_results: list[list[MatchResult]] = [
        [NO_MATCH_RESULT for _gold_group in gold_groups]
        for _item in semantic_items_key
    ]

    for item_index, item in enumerate(semantic_items_key):
        support_by_group = _surface_gold_group_support(item.surface, question)
        for group_index, gold_group in enumerate(gold_groups):
            result = support_by_group.get(gold_group.group_id, NO_MATCH_RESULT)
            edge_results[item_index][group_index] = result
            weights[item_index][group_index] = _match_weight(result)

    assignment = _hungarian_maximize(weights)
    matched_candidates: list[MatchedCandidate] = []
    matched_gold_group_ids: list[int] = []
    for item_index, item in enumerate(semantic_items_key):
        matched_gold_group_id: int | None = None
        match_result = NO_MATCH_RESULT
        assigned_group_index = assignment[item_index] if item_index < len(assignment) else -1
        if 0 <= assigned_group_index < len(gold_groups):
            candidate_result = edge_results[item_index][assigned_group_index]
            if candidate_result.matched:
                matched_gold_group_id = gold_groups[assigned_group_index].group_id
                match_result = candidate_result
                matched_gold_group_ids.append(matched_gold_group_id)

        matched_candidates.append(
            MatchedCandidate(
                surface=item.surface,
                normalized=item.normalized,
                matched_gold_group_id=matched_gold_group_id,
                label=LABEL_GOLD_MATCHED if matched_gold_group_id is not None else LABEL_METRIC_NEGATIVE,
                match_type=match_result.match_type if match_result.matched else None,
                match_confidence=match_result.confidence if match_result.matched else 0.0,
                first_index=item.first_index,
                duplicate_count=item.duplicate_count,
            )
        )
    return tuple(matched_candidates), tuple(matched_gold_group_ids)


def match_semantic_items(
    semantic_items: Sequence[SemanticListItem],
    question: QuestionExample,
) -> tuple[list[MatchedCandidate], tuple[int, ...]]:
    matched_candidates, matched_gold_group_ids = _match_semantic_items_cached(
        question,
        tuple(semantic_items),
    )
    return list(matched_candidates), matched_gold_group_ids


@lru_cache(maxsize=65536)
def _score_item_surfaces_cached(
    question: QuestionExample,
    surfaces_key: tuple[str, ...],
) -> ResponseMetrics:
    matched_candidates, matched_gold_group_ids = _match_official_items_cached(question, surfaces_key)
    return _build_response_metrics(
        question=question,
        matched_candidates=matched_candidates,
        matched_gold_group_ids=matched_gold_group_ids,
    )


def score_item_surfaces(question: QuestionExample, surfaces: Sequence[str]) -> ResponseMetrics:
    return _score_item_surfaces_cached(question, tuple(surfaces))


def _build_response_metrics(
    question: QuestionExample,
    matched_candidates: Sequence[MatchedCandidate],
    matched_gold_group_ids: Sequence[int],
) -> ResponseMetrics:
    true_positives = len(matched_gold_group_ids)
    prediction_count = len(matched_candidates)
    gold_count = len(question.gold_groups)
    precision = safe_ratio(true_positives, prediction_count)
    recall = safe_ratio(true_positives, gold_count)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return ResponseMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        prediction_count=prediction_count,
        gold_count=gold_count,
        invalid_addition_rate=1.0 - precision if prediction_count else 0.0,
        valid_omission_rate=1.0 - recall if gold_count else 0.0,
    )


def build_matched_response(
    record: CandidateBankRecord,
    question: QuestionExample,
    allow_fallback_split: bool = True,
    allow_fallback_pair_construction: bool = False,
) -> MatchedResponse:
    parsed = parse_list_output(record.raw_output, allow_fallback_split=allow_fallback_split)
    matched_candidates, matched_gold_group_ids = _match_official_items_cached(question, parsed.items)
    metrics = _build_response_metrics(
        question=question,
        matched_candidates=matched_candidates,
        matched_gold_group_ids=matched_gold_group_ids,
    )
    missing_gold_group_ids = tuple(
        gold_group.group_id
        for gold_group in question.gold_groups
        if gold_group.group_id not in set(matched_gold_group_ids)
    )

    pair_eligible = parsed.status == "ok" or (allow_fallback_pair_construction and parsed.status == "fallback_split")
    pair_exclusion_reason: str | None = None
    if not pair_eligible:
        pair_exclusion_reason = parsed.status or "unknown_parser_status"

    return MatchedResponse(
        record=record,
        parsed=parsed,
        candidates=tuple(matched_candidates),
        metrics=metrics,
        matched_gold_group_ids=tuple(matched_gold_group_ids),
        missing_gold_group_ids=missing_gold_group_ids,
        has_duplicate_semantic_candidates=any(candidate.duplicate_count > 1 for candidate in matched_candidates),
        pair_eligible=pair_eligible,
        pair_exclusion_reason=pair_exclusion_reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match candidate-bank list outputs to BioASQ gold alias groups using one-to-one matching."
        )
    )
    parser.add_argument(
        "--question-input",
        nargs="+",
        required=True,
        help="Raw BioASQ JSON or prepared JSON files containing list questions and gold answers.",
    )
    parser.add_argument(
        "--candidate-input",
        nargs="+",
        required=True,
        help="Candidate-bank JSON/JSONL or evaluation predictions JSON files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL file where matched response rows will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="bioasq",
        help="Dataset label stored in emitted rows.",
    )
    parser.add_argument(
        "--allow-fallback-split",
        action="store_true",
        help="Allow newline/semicolon fallback parsing for untagged outputs.",
    )
    parser.add_argument(
        "--allow-fallback-pair-construction",
        action="store_true",
        help="Allow fallback-split responses to seed preference pairs. Malformed or empty responses remain ineligible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_by_id = load_question_examples(
        paths=args.question_input,
        dataset_name=args.dataset_name,
    )
    records = load_candidate_bank_records(
        paths=args.candidate_input,
        questions_by_id=questions_by_id,
        dataset_name=args.dataset_name,
    )

    rows = []
    for record in records:
        question = questions_by_id[record.question_id]
        matched = build_matched_response(
            record=record,
            question=question,
            allow_fallback_split=args.allow_fallback_split,
            allow_fallback_pair_construction=args.allow_fallback_pair_construction,
        )
        rows.append(to_jsonable(matched))

    write_jsonl(Path(args.output), rows)


if __name__ == "__main__":
    main()
