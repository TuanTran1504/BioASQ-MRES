from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

from .normalize_set_answers import normalize_answer_surface

_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class SemanticEquivalenceDecision:
    matched: bool
    match_type: str
    confidence: float
    uncertain: bool = False


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in normalize_answer_surface(text).split() if token)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _contains_digit(text: str) -> bool:
    return any(char.isdigit() for char in text)


def _is_informative_single_token(token: str) -> bool:
    return len(token) >= 6 or _contains_digit(token)


def _token_sequence_contained(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    if not shorter or len(shorter) > len(longer):
        return False
    width = len(shorter)
    for start in range(len(longer) - width + 1):
        if tuple(longer[start : start + width]) == shorter:
            return True
    return False


def _acronym_forms(text: str) -> set[str]:
    pieces = re.findall(r"[A-Za-z]+\d*|\d+", text)
    if not pieces:
        return set()

    candidate_sequences = [pieces]
    filtered = [piece for piece in pieces if piece.lower() not in _STOPWORDS]
    if filtered != pieces:
        candidate_sequences.append(filtered)

    forms: set[str] = set()
    for sequence in candidate_sequences:
        if not sequence:
            continue

        initials = []
        for piece in sequence:
            first_alpha = next((char.lower() for char in piece if char.isalpha()), "")
            digits = "".join(char for char in piece if char.isdigit())
            initials.append(first_alpha + digits)
        compact_initials = _compact("".join(initials))
        if 2 <= len(compact_initials) <= 12:
            forms.add(compact_initials)

        uppercase_hint = _compact("".join(char for piece in sequence for char in piece if char.isupper() or char.isdigit()))
        if 2 <= len(uppercase_hint) <= 12:
            forms.add(uppercase_hint)
    return forms


def _question_context_key(question: Any | None) -> str:
    if question is None:
        return "global"

    question_id = str(getattr(question, "question_id", "") or "")
    evidence = tuple(str(item) for item in (getattr(question, "evidence", ()) or ()))
    gold_signature = []
    for group in getattr(question, "gold_groups", ()) or ():
        gold_signature.append(
            (
                getattr(group, "group_id", None),
                tuple(str(alias) for alias in (getattr(group, "aliases", ()) or ())),
            )
        )
    digest = hashlib.sha256(repr((question_id, evidence, tuple(gold_signature))).encode("utf-8")).hexdigest()[:16]
    return f"{question_id}:{digest}" if question_id else digest


def _question_context_text(question: Any | None) -> str:
    if question is None:
        return ""

    parts = []
    question_text = str(getattr(question, "question_text", "") or "").strip()
    if question_text:
        parts.append(f"Question: {question_text}")

    evidence = [str(item).strip() for item in (getattr(question, "evidence", ()) or ()) if str(item).strip()]
    if evidence:
        parts.append("Evidence: " + " ".join(evidence[:4]))

    aliases: list[str] = []
    for group in getattr(question, "gold_groups", ()) or ():
        aliases.extend(str(alias).strip() for alias in (getattr(group, "aliases", ()) or ()) if str(alias).strip())
    if aliases:
        parts.append("Gold aliases: " + "; ".join(dict.fromkeys(aliases[:50])))

    return "\n".join(parts)


def _context_contains_parenthetical_pair(context: str, short: str, long: str) -> bool:
    if not context:
        return False
    compact_context = _compact(context)
    compact_short = _compact(short)
    compact_long = _compact(long)
    if not compact_short or not compact_long:
        return False
    return (
        _compact(f"{long} ({short})") in compact_context
        or _compact(f"{short} ({long})") in compact_context
    )


def _gold_alias_group_contains_pair(question: Any | None, short: str, long: str) -> bool:
    if question is None:
        return False
    compact_short = _compact(short)
    compact_long = _compact(long)
    if not compact_short or not compact_long:
        return False
    for group in getattr(question, "gold_groups", ()) or ():
        alias_compacts = {_compact(str(alias)) for alias in (getattr(group, "aliases", ()) or ())}
        if compact_short in alias_compacts and compact_long in alias_compacts:
            return True
    return False


def _acronym_candidates(left: str, right: str) -> tuple[tuple[str, str], ...]:
    left_compact = _compact(left)
    right_compact = _compact(right)
    left_acronyms = _acronym_forms(left)
    right_acronyms = _acronym_forms(right)

    candidates: list[tuple[str, str]] = []
    if left_compact and left_compact in right_acronyms:
        candidates.append((left, right))
    if right_compact and right_compact in left_acronyms:
        candidates.append((right, left))
    return tuple(candidates)


class HybridSemanticEquivalenceLabeler:
    def __init__(
        self,
        *,
        embedding_model_name: str | None = None,
        embedding_match_threshold: float = 0.90,
        embedding_uncertain_threshold: float = 0.82,
        embedding_no_match_threshold: float = 0.50,
        nli_model_name: str | None = None,
        nli_match_threshold: float = 0.80,
        nli_uncertain_threshold: float = 0.60,
    ) -> None:
        if embedding_no_match_threshold > embedding_uncertain_threshold:
            raise ValueError("embedding_no_match_threshold must be <= embedding_uncertain_threshold")
        if embedding_uncertain_threshold > embedding_match_threshold:
            raise ValueError("embedding_uncertain_threshold must be <= embedding_match_threshold")
        if nli_uncertain_threshold > nli_match_threshold:
            raise ValueError("nli_uncertain_threshold must be <= nli_match_threshold")

        self.embedding_model_name = embedding_model_name or None
        self.embedding_match_threshold = float(embedding_match_threshold)
        self.embedding_uncertain_threshold = float(embedding_uncertain_threshold)
        self.embedding_no_match_threshold = float(embedding_no_match_threshold)
        self.nli_model_name = nli_model_name or None
        self.nli_match_threshold = float(nli_match_threshold)
        self.nli_uncertain_threshold = float(nli_uncertain_threshold)

        self._embedding_model: Any | None = None
        self._nli_tokenizer: Any | None = None
        self._nli_model: Any | None = None
        self._decision_cache: dict[tuple[str, str, str], SemanticEquivalenceDecision] = {}
        self._embedding_cache: dict[str, Any] = {}
        self._nli_cache: dict[tuple[str, str], tuple[float, float]] = {}

    @property
    def enabled(self) -> bool:
        return True

    def answer_equivalent(
        self,
        left: str,
        right: str,
        *,
        question: Any | None = None,
    ) -> SemanticEquivalenceDecision:
        context_key = _question_context_key(question)
        key = (*tuple(sorted((_compact(left), _compact(right)))), context_key)
        cached = self._decision_cache.get(key)
        if cached is not None:
            return cached

        fallback_decision = self._rule_based_decision(left, right, question=question)
        if fallback_decision is not None and fallback_decision.matched:
            self._decision_cache[key] = fallback_decision
            return fallback_decision

        if self.embedding_model_name:
            decision = self._embedding_decision(left, right)
            if decision is not None and decision.matched:
                self._decision_cache[key] = decision
                return decision
            if (
                decision is not None
                and not decision.uncertain
                and decision.match_type == "biomedical_embedding_low_similarity_no_match"
            ):
                self._decision_cache[key] = decision
                return decision
            if decision is not None:
                fallback_decision = decision
        else:
            decision = None

        if self.nli_model_name:
            nli_decision = self._nli_decision(left, right, question=question)
            if nli_decision is not None:
                self._decision_cache[key] = nli_decision
                return nli_decision

        if fallback_decision is not None:
            self._decision_cache[key] = fallback_decision
            return fallback_decision

        final = SemanticEquivalenceDecision(
            matched=False,
            match_type="no_match",
            confidence=0.0,
            uncertain=False,
        )
        self._decision_cache[key] = final
        return final

    def _rule_based_decision(
        self,
        left: str,
        right: str,
        *,
        question: Any | None = None,
    ) -> SemanticEquivalenceDecision | None:
        left_normalized = normalize_answer_surface(left)
        right_normalized = normalize_answer_surface(right)
        if not left_normalized or not right_normalized:
            return None
        if left_normalized == right_normalized:
            return SemanticEquivalenceDecision(
                matched=True,
                match_type="exact_normalized",
                confidence=1.0,
            )

        left_compact = _compact(left)
        right_compact = _compact(right)
        if left_compact and left_compact == right_compact:
            return SemanticEquivalenceDecision(
                matched=True,
                match_type="exact_compact",
                confidence=0.99,
            )

        acronym_candidates = _acronym_candidates(left, right)
        left_acronyms = _acronym_forms(left)
        right_acronyms = _acronym_forms(right)
        if acronym_candidates or bool(left_acronyms & right_acronyms):
            context_text = _question_context_text(question)
            for short, long in acronym_candidates:
                if (
                    _context_contains_parenthetical_pair(context_text, short, long)
                    or _gold_alias_group_contains_pair(question, short, long)
                ):
                    return SemanticEquivalenceDecision(
                        matched=True,
                        match_type="rule_explicit_acronym_equivalent",
                        confidence=0.96,
                    )
            return SemanticEquivalenceDecision(
                matched=False,
                match_type="rule_acronym_ambiguous",
                confidence=0.55,
                uncertain=True,
            )

        left_tokens = _normalized_tokens(left)
        right_tokens = _normalized_tokens(right)
        shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
        if _token_sequence_contained(shorter, longer):
            if len(shorter) >= 2:
                return SemanticEquivalenceDecision(
                    matched=False,
                    match_type="rule_phrase_containment_ambiguous",
                    confidence=0.55,
                    uncertain=True,
                )
            if shorter and _is_informative_single_token(shorter[0]):
                return SemanticEquivalenceDecision(
                    matched=False,
                    match_type="rule_single_token_containment_ambiguous",
                    confidence=0.55,
                    uncertain=True,
                )
        return None

    def _load_embedding_model(self) -> Any:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _embedding_for(self, text: str) -> Any:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        model = self._load_embedding_model()
        vector = model.encode(text, normalize_embeddings=True)
        self._embedding_cache[text] = vector
        return vector

    def _embedding_decision(self, left: str, right: str) -> SemanticEquivalenceDecision | None:
        left_vector = self._embedding_for(left)
        right_vector = self._embedding_for(right)
        similarity = float(sum(float(a) * float(b) for a, b in zip(left_vector, right_vector)))
        if similarity >= self.embedding_match_threshold:
            return SemanticEquivalenceDecision(
                matched=True,
                match_type="biomedical_embedding_match",
                confidence=similarity,
            )
        if similarity <= self.embedding_no_match_threshold:
            return SemanticEquivalenceDecision(
                matched=False,
                match_type="biomedical_embedding_low_similarity_no_match",
                confidence=similarity,
                uncertain=False,
            )
        if similarity >= self.embedding_uncertain_threshold:
            return SemanticEquivalenceDecision(
                matched=False,
                match_type="biomedical_embedding_ambiguous",
                confidence=similarity,
                uncertain=True,
            )
        return SemanticEquivalenceDecision(
            matched=False,
            match_type="biomedical_embedding_review_band",
            confidence=similarity,
            uncertain=True,
        )

    def _load_nli_model(self) -> tuple[Any, Any]:
        if self._nli_model is None or self._nli_tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            self._nli_model.eval()
        return self._nli_tokenizer, self._nli_model

    def _label_id_map(self, model: Any) -> dict[str, int]:
        id2label = getattr(model.config, "id2label", {}) or {}
        mapping: dict[str, int] = {}
        for index, label in id2label.items():
            normalized = str(label).lower()
            if "entail" in normalized:
                mapping["entailment"] = int(index)
            elif "contrad" in normalized:
                mapping["contradiction"] = int(index)
            elif "neutral" in normalized:
                mapping["neutral"] = int(index)
        if "entailment" not in mapping:
            raise ValueError(f"NLI model {self.nli_model_name!r} does not expose entailment labels")
        return mapping

    def _nli_scores(self, premise: str, hypothesis: str) -> tuple[float, float]:
        key = (premise, hypothesis)
        cached = self._nli_cache.get(key)
        if cached is not None:
            return cached

        tokenizer, model = self._load_nli_model()
        label_map = self._label_id_map(model)

        import torch

        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        entailment = float(probabilities[label_map["entailment"]])
        contradiction = float(probabilities[label_map.get("contradiction", -1)]) if "contradiction" in label_map else 0.0
        result = (entailment, contradiction)
        self._nli_cache[key] = result
        return result

    def _nli_decision(
        self,
        left: str,
        right: str,
        *,
        question: Any | None = None,
    ) -> SemanticEquivalenceDecision | None:
        context = _question_context_text(question)
        left_statement = f"Candidate answer: {left}\n{context}" if context else left
        right_statement = f"Candidate answer: {right}\n{context}" if context else right
        left_to_right = self._nli_scores(left_statement, right_statement)
        right_to_left = self._nli_scores(right_statement, left_statement)
        entailment = min(left_to_right[0], right_to_left[0])
        contradiction = max(left_to_right[1], right_to_left[1])

        if entailment >= self.nli_match_threshold and contradiction < 0.5:
            return SemanticEquivalenceDecision(
                matched=True,
                match_type="nli_equivalence_match",
                confidence=entailment,
            )
        if entailment >= self.nli_uncertain_threshold:
            return SemanticEquivalenceDecision(
                matched=False,
                match_type="nli_equivalence_ambiguous",
                confidence=entailment,
                uncertain=True,
            )
        if contradiction >= self.nli_match_threshold:
            return SemanticEquivalenceDecision(
                matched=False,
                match_type="nli_contradiction_no_match",
                confidence=contradiction,
                uncertain=False,
            )
        return SemanticEquivalenceDecision(
            matched=False,
            match_type="nli_no_match",
            confidence=max(entailment, contradiction),
            uncertain=False,
        )
