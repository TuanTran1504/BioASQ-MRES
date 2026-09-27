from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.model_registry import slugify
from src.utility.data import clean_text
from src.utility.eval_models import prime_unsloth_runtime

from .audit_preference_pairs import build_summary, render_pair_block
from .common import summarize_numeric, truncate_multiline, write_json, write_jsonl
from .construct_set_edit_pairs import build_response_audit_summary, group_responses_by_question
from .match_gold_groups import score_item_surfaces
from .normalize_set_answers import parse_list_output, serialize_list_items
from .schemas import MatchedResponse, PreferencePair, QuestionExample, to_jsonable


prime_unsloth_runtime()

PAIR_TYPE_WHOLE_RESPONSE_TASK_DCRM = "whole_response_task_dcrm"
REWARD_FORMULA_F1 = "f1"
REWARD_FORMULA_F1_RECALL = "f1_recall"
REWARD_FORMULA_F1_RECALL_ENTITY_GAP = "f1_recall_entity_gap"
REWARD_FORMULA_WEIGHTED = "weighted"
EPSILON_DEFAULT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct BoN^2-style DCRM pairs from a scored candidate bank using a task-aware "
            "reward built from BioASQ metrics."
        )
    )
    parser.add_argument("--question-input", nargs="+", required=True)
    parser.add_argument("--candidate-input", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--manual-audit-md", required=True)
    parser.add_argument("--dataset-name", default="bioasq")
    parser.add_argument(
        "--reference-logp-field",
        default="policy_logp_sum",
        help=(
            "Candidate-bank field used as the reference log-probability term in the DCRM "
            "denominator. For same-source setups this is typically policy_logp_sum from the "
            "generator model itself."
        ),
    )
    parser.add_argument(
        "--completion-source",
        choices=["raw", "parsed", "parsed_or_raw"],
        default="parsed",
        help=(
            "What completion text to use when tokenizing responses for token-level edit distance. "
            "'parsed' serializes parsed list items, while 'raw' uses the literal model output."
        ),
    )
    parser.add_argument(
        "--reward-formula",
        choices=[
            REWARD_FORMULA_F1,
            REWARD_FORMULA_F1_RECALL,
            REWARD_FORMULA_F1_RECALL_ENTITY_GAP,
            REWARD_FORMULA_WEIGHTED,
        ],
        default=REWARD_FORMULA_F1,
    )
    parser.add_argument("--f1-weight", type=float, default=1.0)
    parser.add_argument("--recall-weight", type=float, default=0.0)
    parser.add_argument("--precision-weight", type=float, default=0.0)
    parser.add_argument(
        "--entity-gap-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight for the normalized absolute difference between predicted entity count "
            "and gold entity count."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help="Additive constant in the DCRM denominator. The paper uses epsilon=1.",
    )
    parser.add_argument(
        "--tokenizer-model-ref",
        required=True,
        help="Model or adapter path used to tokenize completions for token-level edit distance.",
    )
    parser.add_argument("--registry-path", default="models/registry.json")
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-format", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--allow-fallback-pair-construction", action="store_true")
    parser.add_argument("--allow-multiple-generator-checkpoints", action="store_true")
    parser.add_argument("--manual-audit-sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_ref,
        local_files_only=bool(args.local_files_only),
        use_fast=True,
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_score_lookup(paths: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() != ".jsonl":
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = dict(json.loads(line))
                question_id = clean_text(row.get("question_id") or row.get("id"))
                response_id = clean_text(row.get("response_id"))
                if not question_id or not response_id:
                    continue
                lookup[(question_id, response_id)] = row
    return lookup


def canonical_completion_from_row(row: Mapping[str, Any], allow_fallback_split: bool) -> str:
    parsed_items_value = row.get("parsed_items")
    parsed_items: list[str] = []
    if isinstance(parsed_items_value, Sequence) and not isinstance(parsed_items_value, (str, bytes)):
        parsed_items = [clean_text(item) for item in parsed_items_value if clean_text(item)]
    if not parsed_items:
        raw_output = clean_text(row.get("raw_output") or row.get("prediction"))
        if raw_output:
            parsed = parse_list_output(raw_output, allow_fallback_split=allow_fallback_split)
            parsed_items = [clean_text(item) for item in parsed.items if clean_text(item)]
    return serialize_list_items(parsed_items)


def completion_text_from_row(
    row: Mapping[str, Any],
    *,
    completion_source: str,
    allow_fallback_split: bool,
) -> str:
    raw_output = clean_text(row.get("raw_output") or row.get("prediction"))
    parsed_text = canonical_completion_from_row(row, allow_fallback_split=allow_fallback_split)
    if completion_source == "raw":
        return raw_output
    if completion_source == "parsed":
        return parsed_text
    return parsed_text or raw_output


def token_ids_for_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = tokenized.get("input_ids")
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return tuple(int(token_id) for token_id in input_ids)
    if hasattr(input_ids, "tolist"):
        payload = input_ids.tolist()
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            payload = payload[0]
        if isinstance(payload, list):
            return tuple(int(token_id) for token_id in payload)
    return tuple()


def levenshtein_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def normalized_entity_gap(metrics: Any) -> float:
    gold_count = max(1, int(metrics.gold_count))
    return abs(int(metrics.prediction_count) - int(metrics.gold_count)) / gold_count


def reward_weights_for_formula(args: argparse.Namespace) -> dict[str, float]:
    if args.reward_formula == REWARD_FORMULA_F1:
        return {
            "f1": 1.0,
            "recall": 0.0,
            "precision": 0.0,
            "entity_gap": 0.0,
        }
    if args.reward_formula == REWARD_FORMULA_F1_RECALL:
        return {
            "f1": 1.0,
            "recall": 0.2,
            "precision": 0.0,
            "entity_gap": 0.0,
        }
    if args.reward_formula == REWARD_FORMULA_F1_RECALL_ENTITY_GAP:
        return {
            "f1": 1.0,
            "recall": 0.2,
            "precision": 0.0,
            "entity_gap": 0.1,
        }
    return {
        "f1": float(args.f1_weight),
        "recall": float(args.recall_weight),
        "precision": float(args.precision_weight),
        "entity_gap": float(args.entity_gap_weight),
    }


def response_reward(metrics: Any, weights: Mapping[str, float]) -> float:
    return (
        float(weights.get("f1", 0.0)) * float(metrics.f1)
        + float(weights.get("recall", 0.0)) * float(metrics.recall)
        + float(weights.get("precision", 0.0)) * float(metrics.precision)
        - float(weights.get("entity_gap", 0.0)) * normalized_entity_gap(metrics)
    )


def pair_row_key(pair: PreferencePair) -> tuple[str, str, str]:
    return pair.question_id, pair.chosen, pair.rejected


def build_task_dcrm_pair(
    *,
    pair_id: str,
    question: QuestionExample,
    chosen_response: MatchedResponse,
    rejected_response: MatchedResponse,
    pair_label: str,
) -> PreferencePair:
    chosen_items = tuple(candidate.surface for candidate in chosen_response.candidates)
    rejected_items = tuple(candidate.surface for candidate in rejected_response.candidates)
    semantic_set_edit_distance = len(set(chosen_items).symmetric_difference(set(rejected_items)))

    return PreferencePair(
        pair_id=pair_id,
        dataset=question.dataset,
        question_id=question.question_id,
        question_text=question.question_text,
        question_source_path=question.source_path,
        prompt=chosen_response.record.prompt or rejected_response.record.prompt,
        chosen=serialize_list_items(chosen_items),
        rejected=serialize_list_items(rejected_items),
        pair_type=PAIR_TYPE_WHOLE_RESPONSE_TASK_DCRM,
        base_response_id=rejected_response.record.response_id,
        edited_candidate="",
        edited_candidate_normalized="",
        edited_gold_group_id=None,
        candidate_label=pair_label,
        candidate_label_source="task_dcrm_best_of_n2",
        positive_source=None,
        semantic_set_edit_distance=semantic_set_edit_distance,
        chosen_items=chosen_items,
        rejected_items=rejected_items,
        chosen_precision=float(chosen_response.metrics.precision),
        chosen_recall=float(chosen_response.metrics.recall),
        chosen_f1=float(chosen_response.metrics.f1),
        rejected_precision=float(rejected_response.metrics.precision),
        rejected_recall=float(rejected_response.metrics.recall),
        rejected_f1=float(rejected_response.metrics.f1),
        delta_precision=float(chosen_response.metrics.precision - rejected_response.metrics.precision),
        delta_recall=float(chosen_response.metrics.recall - rejected_response.metrics.recall),
        delta_f1=float(chosen_response.metrics.f1 - rejected_response.metrics.f1),
        generator_checkpoint=chosen_response.record.generator_checkpoint,
        sample_id=chosen_response.record.sample_id,
    )


def build_task_dcrm_audit_markdown(
    *,
    rows: Sequence[Mapping[str, Any]],
    questions_by_id: Mapping[str, QuestionExample],
    sample_size: int,
    seed: int,
    reward_formula: str,
) -> str:
    rng = random.Random(seed)
    selected_rows = [dict(row) for row in rows]
    rng.shuffle(selected_rows)
    selected_rows = selected_rows[:sample_size]

    parts = [
        "# Study6 Task-DCRM Pair Audit",
        "",
        f"Selected pairs: {len(selected_rows)} / {len(rows)}",
        "",
        (
            "Pairs are selected with a BoN^2-style task-aware DCRM score: "
            "(sigmoid(reward_margin) - 0.5) / (token_edit_distance + reference_logp_delta + epsilon). "
            f"Reward formula: {reward_formula}."
        ),
        "",
    ]
    for row in selected_rows:
        question = questions_by_id.get(clean_text(row.get("question_id")))
        evidence_preview = truncate_multiline(question.evidence if question else [], max_chars_per_item=350)
        parts.append(render_pair_block(row, evidence_preview=evidence_preview))
        parts.append(
            f"\nChosen reward: {float(row.get('chosen_reward', 0.0)):.6f}\n\n"
            f"Rejected reward: {float(row.get('rejected_reward', 0.0)):.6f}\n\n"
            f"Reward margin: {float(row.get('reward_margin', 0.0)):.6f}\n\n"
            f"Token edit distance: {int(row.get('token_edit_distance', 0))}\n\n"
            f"Reference logp delta: {float(row.get('reference_logp_delta', 0.0)):.6f}\n\n"
            f"DCRM score: {float(row.get('task_dcrm_score', 0.0)):.6f}\n"
        )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    questions_by_id, responses_by_question, responses_by_question_and_checkpoint = group_responses_by_question(
        question_input=args.question_input,
        candidate_input=args.candidate_input,
        dataset_name=args.dataset_name,
        allow_fallback_split=bool(args.allow_fallback_split),
        allow_fallback_pair_construction=bool(args.allow_fallback_pair_construction),
        max_resources=5,
        max_resource_chars=1200,
        gold_support_policy="all",
        question_limit=args.limit,
    )

    generator_checkpoints = {
        clean_text(response.record.generator_checkpoint) or "unknown"
        for responses in responses_by_question.values()
        for response in responses
    }
    if len(generator_checkpoints) > 1 and not args.allow_multiple_generator_checkpoints:
        raise ValueError(
            "Multiple generator checkpoints were found in the candidate inputs. "
            "Construct pairs from one frozen generator checkpoint, or pass "
            "--allow-multiple-generator-checkpoints to partition by checkpoint. "
            f"Found: {sorted(generator_checkpoints)}"
        )

    tokenizer = load_tokenizer(args)
    score_lookup = load_score_lookup(args.candidate_input)
    reward_weights = reward_weights_for_formula(args)

    pair_rows: list[dict[str, Any]] = []
    pair_audit = Counter()

    for (question_id, checkpoint_key), responses in responses_by_question_and_checkpoint.items():
        question = questions_by_id[question_id]
        checkpoint_slug = slugify(checkpoint_key, fallback="checkpoint")
        pair_id_prefix = f"{slugify(question.dataset)}-{slugify(question.question_id)}-{checkpoint_slug}"

        eligible: list[tuple[MatchedResponse, dict[str, Any], float, tuple[int, ...]]] = []
        for response in responses:
            pair_audit["responses_total"] += 1
            if not response.pair_eligible:
                pair_audit["responses_ineligible"] += 1
                continue
            score_row = score_lookup.get((response.record.question_id, response.record.response_id))
            if score_row is None:
                pair_audit["responses_missing_score"] += 1
                continue
            logp_value = score_row.get(args.reference_logp_field)
            if not isinstance(logp_value, (int, float)) or not math.isfinite(float(logp_value)):
                pair_audit["responses_missing_score"] += 1
                continue
            completion_text = completion_text_from_row(
                score_row,
                completion_source=str(args.completion_source),
                allow_fallback_split=bool(args.allow_fallback_split),
            )
            token_ids = token_ids_for_text(tokenizer, completion_text)
            if not token_ids:
                pair_audit["responses_empty_completion"] += 1
                continue
            reward_value = response_reward(response.metrics, reward_weights)
            eligible.append((response, score_row, reward_value, token_ids))

        pair_audit["responses_eligible"] += len(eligible)
        if len(eligible) < 2:
            pair_audit["questions_with_too_few_responses"] += 1
            continue

        best_candidate: dict[str, Any] | None = None
        for left_index, (left_response, _left_row, left_reward, left_tokens) in enumerate(eligible):
            for right_response, _right_row, right_reward, right_tokens in eligible[left_index + 1 :]:
                pair_audit["candidate_response_pairs_total"] += 1

                if left_reward == right_reward:
                    pair_audit["filtered_tied_reward_pairs"] += 1
                    continue

                if left_reward > right_reward:
                    chosen_response = left_response
                    rejected_response = right_response
                    chosen_reward = left_reward
                    rejected_reward = right_reward
                    chosen_tokens = left_tokens
                    rejected_tokens = right_tokens
                    chosen_logp = float(score_lookup[(left_response.record.question_id, left_response.record.response_id)][args.reference_logp_field])
                    rejected_logp = float(score_lookup[(right_response.record.question_id, right_response.record.response_id)][args.reference_logp_field])
                else:
                    chosen_response = right_response
                    rejected_response = left_response
                    chosen_reward = right_reward
                    rejected_reward = left_reward
                    chosen_tokens = right_tokens
                    rejected_tokens = left_tokens
                    chosen_logp = float(score_lookup[(right_response.record.question_id, right_response.record.response_id)][args.reference_logp_field])
                    rejected_logp = float(score_lookup[(left_response.record.question_id, left_response.record.response_id)][args.reference_logp_field])

                if chosen_response.record.response_id == rejected_response.record.response_id:
                    pair_audit["filtered_duplicate_response_ids"] += 1
                    continue

                reward_margin = chosen_reward - rejected_reward
                if reward_margin <= 0.0:
                    pair_audit["filtered_non_positive_reward_margin_pairs"] += 1
                    continue

                token_edit_distance = levenshtein_distance(chosen_tokens, rejected_tokens)
                if token_edit_distance == 0:
                    pair_audit["filtered_zero_edit_distance_pairs"] += 1
                    continue

                reference_logp_delta = abs(chosen_logp - rejected_logp)
                denominator = float(token_edit_distance) + float(reference_logp_delta) + float(args.epsilon)
                dcrm_score = (sigmoid(reward_margin) - 0.5) / denominator

                candidate = {
                    "chosen_response": chosen_response,
                    "rejected_response": rejected_response,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "reward_margin": reward_margin,
                    "token_edit_distance": token_edit_distance,
                    "reference_logp_delta": reference_logp_delta,
                    "chosen_logp": chosen_logp,
                    "rejected_logp": rejected_logp,
                    "task_dcrm_score": dcrm_score,
                }

                if best_candidate is None:
                    best_candidate = candidate
                    continue

                if (
                    candidate["task_dcrm_score"],
                    candidate["reward_margin"],
                    -candidate["reference_logp_delta"],
                    -candidate["token_edit_distance"],
                    candidate["chosen_response"].metrics.f1,
                    clean_text(candidate["chosen_response"].record.response_id),
                ) > (
                    best_candidate["task_dcrm_score"],
                    best_candidate["reward_margin"],
                    -best_candidate["reference_logp_delta"],
                    -best_candidate["token_edit_distance"],
                    best_candidate["chosen_response"].metrics.f1,
                    clean_text(best_candidate["chosen_response"].record.response_id),
                ):
                    best_candidate = candidate

        if best_candidate is None:
            pair_audit["questions_without_valid_pairs"] += 1
            continue

        pair_audit["questions_with_selected_pair"] += 1
        pair = build_task_dcrm_pair(
            pair_id=f"{pair_id_prefix}-task-dcrm-0000",
            question=question,
            chosen_response=best_candidate["chosen_response"],
            rejected_response=best_candidate["rejected_response"],
            pair_label=f"whole_response_ranked_by_task_dcrm_{slugify(args.reward_formula, fallback='reward')}",
        )
        row = to_jsonable(pair)
        row.update(
            {
                "reward_formula": str(args.reward_formula),
                "reward_weights": dict(reward_weights),
                "chosen_reward": float(best_candidate["chosen_reward"]),
                "rejected_reward": float(best_candidate["rejected_reward"]),
                "reward_margin": float(best_candidate["reward_margin"]),
                "token_edit_distance": int(best_candidate["token_edit_distance"]),
                "reference_logp_field": str(args.reference_logp_field),
                "reference_logp_delta": float(best_candidate["reference_logp_delta"]),
                "chosen_reference_logp": float(best_candidate["chosen_logp"]),
                "rejected_reference_logp": float(best_candidate["rejected_logp"]),
                "epsilon": float(args.epsilon),
                "task_dcrm_score": float(best_candidate["task_dcrm_score"]),
            }
        )
        pair_rows.append(row)

    pair_rows = sorted(pair_rows, key=lambda row: (clean_text(row.get("question_id")), clean_text(row.get("pair_id"))))
    write_jsonl(Path(args.output_jsonl), pair_rows)

    question_gold_counts = {
        question_id: len(question.gold_groups)
        for question_id, question in questions_by_id.items()
    }
    summary = build_summary(pair_rows, question_gold_counts=question_gold_counts)
    summary["response_audit"] = build_response_audit_summary(responses_by_question)
    summary["task_dcrm_pair_audit"] = dict(sorted(pair_audit.items()))
    summary["pair_construction_policy"] = {
        "pair_type": PAIR_TYPE_WHOLE_RESPONSE_TASK_DCRM,
        "description": (
            "Study6 task-aware BoN^2 pair selection using a DCRM-style pair score with "
            "task reward margins, token-level edit distance, and reference log-prob deltas."
        ),
        "reward_formula": str(args.reward_formula),
        "reward_weights": dict(reward_weights),
        "reference_logp_field": str(args.reference_logp_field),
        "completion_source": str(args.completion_source),
        "epsilon": float(args.epsilon),
        "gold_support_policy": "all",
        "selected_pairs_per_question": 1,
    }
    summary["task_dcrm_score_distribution"] = summarize_numeric(
        float(row["task_dcrm_score"]) for row in pair_rows if isinstance(row.get("task_dcrm_score"), (int, float))
    )
    summary["reward_margin_distribution"] = summarize_numeric(
        float(row["reward_margin"]) for row in pair_rows if isinstance(row.get("reward_margin"), (int, float))
    )
    summary["reference_logp_delta_distribution"] = summarize_numeric(
        float(row["reference_logp_delta"])
        for row in pair_rows
        if isinstance(row.get("reference_logp_delta"), (int, float))
    )
    summary["token_edit_distance_distribution"] = summarize_numeric(
        float(row["token_edit_distance"]) for row in pair_rows if isinstance(row.get("token_edit_distance"), (int, float))
    )
    summary["chosen_reward_distribution"] = summarize_numeric(
        float(row["chosen_reward"]) for row in pair_rows if isinstance(row.get("chosen_reward"), (int, float))
    )
    summary["rejected_reward_distribution"] = summarize_numeric(
        float(row["rejected_reward"]) for row in pair_rows if isinstance(row.get("rejected_reward"), (int, float))
    )
    write_json(Path(args.summary_json), summary)

    markdown = build_task_dcrm_audit_markdown(
        rows=pair_rows,
        questions_by_id=questions_by_id,
        sample_size=int(args.manual_audit_sample_size),
        seed=int(args.seed),
        reward_formula=str(args.reward_formula),
    )
    Path(args.manual_audit_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_audit_md).write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(pair_rows):,} Study6 task-DCRM pairs to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
