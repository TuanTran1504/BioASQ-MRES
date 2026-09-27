from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cse_dpo.common import load_json_records, read_json, summarize_numeric, write_json, write_jsonl
from cse_dpo.construct_factoid_baseline_data_views import (
    EXPECTED_CLASS_SIZE,
    representative_permutation_index,
)


@dataclass
class TrainExample:
    question_id: str
    split: str
    prompt: str
    gold: str
    accepted_gold: list[str]
    wrongs: list[str]
    rejected_rank: int
    rank_assignment_index: int
    rank_assignment_seed: int
    preferred_sequences: list[dict[str, Any]]
    rejected_sequences: list[dict[str, Any]]
    preferred_reference_logps: list[float]
    rejected_reference_logps: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an exact-orbit factoid DPO policy from question-class rows with "
            "precomputed frozen-reference sequence log-probabilities."
        )
    )
    parser.add_argument(
        "--question-classes-with-reference-jsonl",
        required=True,
        help="JSONL emitted by precompute_factoid_reference_logprobs.py.",
    )
    parser.add_argument(
        "--validation-question-classes-with-reference-jsonl",
        default=None,
        help=(
            "Optional separate validation JSONL emitted by "
            "precompute_factoid_reference_logprobs.py. When omitted, "
            "--validation-ratio is applied to the training input."
        ),
    )
    parser.add_argument("--model-name", required=True, help="Base model or SFT adapter path.")
    parser.add_argument("--output-dir", required=True, help="Directory for logs, summaries, and best checkpoints.")
    parser.add_argument("--save-model-dir", required=True, help="Directory where the final adapter/tokenizer is saved.")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "representative_dpo",
            "softmax_dpo",
            "random_permutation_dpo",
            "permutation_augmented_dpo",
            "class_dpo",
            "rp_class_dpo",
        ],
        help="Exact-orbit training objective to optimize.",
    )
    parser.add_argument(
        "--rp-lambda",
        type=float,
        default=0.1,
        help="Within-class KL coefficient used only for rp_class_dpo.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--scoring-batch-size", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "class_margin", "fixed_slate_mrr"],
        default="val_loss",
        help="Metric used to select and early-stop checkpoints.",
    )
    parser.add_argument("--skip-fixed-slate-mrr", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules used when attaching a new LoRA adapter to a base model.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit-train-questions", type=int, default=None)
    parser.add_argument("--limit-validation-questions", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint directory created by this trainer to resume from.",
    )
    return parser.parse_args()


def resolve_dtype(name: str | None) -> torch.dtype | None:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def stable_logsumexp(values: Sequence[float]) -> float | None:
    if not values:
        return None
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    max_value = max(finite_values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in finite_values))


def format_factoid(entity: str) -> str:
    return f"[BE]{str(entity).strip()}[EE]"


def serialize_sequence(entities: Sequence[str]) -> str:
    return " ".join(format_factoid(entity) for entity in entities)


def split_train_validation(
    rows: list[TrainExample],
    *,
    ratio: float,
    seed: int,
) -> tuple[list[TrainExample], list[TrainExample]]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1 when no validation file is supplied.")
    if len(rows) < 2:
        raise ValueError("Need at least 2 questions to create a train/validation split.")
    ordered = list(rows)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_validation = max(1, round(len(ordered) * ratio))
    n_validation = min(n_validation, len(ordered) - 1)
    return ordered[n_validation:], ordered[:n_validation]


def validate_row_shape(raw_row: dict[str, Any]) -> None:
    preferred = list(raw_row.get("preferred_sequences", []))
    rejected = list(raw_row.get("rejected_sequences", []))
    if len(preferred) != EXPECTED_CLASS_SIZE:
        raise ValueError(
            f"Question {raw_row.get('question_id')} has {len(preferred)} preferred sequences; "
            f"expected {EXPECTED_CLASS_SIZE}."
        )
    if len(rejected) != EXPECTED_CLASS_SIZE:
        raise ValueError(
            f"Question {raw_row.get('question_id')} has {len(rejected)} rejected sequences; "
            f"expected {EXPECTED_CLASS_SIZE}."
        )


def parse_train_example(raw_row: dict[str, Any]) -> TrainExample:
    validate_row_shape(raw_row)
    preferred_sequences = [dict(item) for item in raw_row["preferred_sequences"]]
    rejected_sequences = [dict(item) for item in raw_row["rejected_sequences"]]
    preferred_reference_logps = []
    rejected_reference_logps = []
    for entry in preferred_sequences:
        value = entry.get("reference_logp_sum")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing preferred reference log-prob for question {raw_row.get('question_id')}.")
        preferred_reference_logps.append(float(value))
    for entry in rejected_sequences:
        value = entry.get("reference_logp_sum")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing rejected reference log-prob for question {raw_row.get('question_id')}.")
        rejected_reference_logps.append(float(value))
    return TrainExample(
        question_id=str(raw_row.get("question_id", "")),
        split=str(raw_row.get("split", "")),
        prompt=str(raw_row.get("prompt", "")),
        gold=str(raw_row.get("gold", "")),
        accepted_gold=[str(value) for value in raw_row.get("accepted_gold", [])],
        wrongs=[str(value) for value in raw_row.get("wrongs", [])],
        rejected_rank=int(raw_row.get("rejected_rank", 0) or 0),
        rank_assignment_index=int(raw_row.get("rank_assignment_index", 0) or 0),
        rank_assignment_seed=int(raw_row.get("rank_assignment_seed", 0) or 0),
        preferred_sequences=preferred_sequences,
        rejected_sequences=rejected_sequences,
        preferred_reference_logps=preferred_reference_logps,
        rejected_reference_logps=rejected_reference_logps,
    )


def load_examples(path: Path) -> list[TrainExample]:
    return [parse_train_example(dict(row)) for row in load_json_records(path)]


def load_tokenizer(model_name: str, *, local_files_only: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def maybe_attach_lora(model: Any, args: argparse.Namespace) -> Any:
    if isinstance(model, PeftModel):
        model.train()
        return model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=list(args.target_modules),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.train()
    return model


def load_policy_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    model_path = Path(args.model_name)
    dtype = resolve_dtype(args.dtype)
    if dtype is None and device.type == "cuda":
        dtype = torch.float16
    if dtype is None:
        dtype = torch.float32

    tokenizer = load_tokenizer(args.model_name, local_files_only=bool(args.local_files_only))
    if model_path.is_dir() and (model_path / "adapter_config.json").exists():
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_name,
                is_trainable=True,
                torch_dtype=dtype,
                local_files_only=bool(args.local_files_only),
            )
        except TypeError:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=dtype,
                local_files_only=bool(args.local_files_only),
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=bool(args.local_files_only),
        )

    model = maybe_attach_lora(model, args)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.to(device)
    return model, tokenizer


def build_eval_rankings(example: TrainExample) -> tuple[list[str], list[int]]:
    slate = [example.gold, *example.wrongs]
    rankings = []
    gold_ranks = []
    for ranking in itertools.permutations(slate):
        entities = list(ranking)
        rankings.append(serialize_sequence(entities))
        gold_ranks.append(entities.index(example.gold) + 1)
    return rankings, gold_ranks


def encode_pair(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_ids = list(completion_ids) + [int(eos_token_id)]
    if len(completion_ids) >= max_seq_length:
        raise ValueError("Completion alone exceeds --max-seq-length.")
    prompt_ids = prompt_ids[-(max_seq_length - len(completion_ids)) :]
    input_ids = list(prompt_ids) + list(completion_ids)
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return input_ids, labels


def collate_pairs(
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    max_seq_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = [
        encode_pair(tokenizer, prompt=prompt, completion=completion, max_seq_length=max_seq_length)
        for prompt, completion in pairs
    ]
    width = max(len(input_ids) for input_ids, _ in encoded)
    input_ids_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    for input_ids, labels in encoded:
        pad = width - len(input_ids)
        input_ids_rows.append(input_ids + [pad_token_id] * pad)
        attention_rows.append([1] * len(input_ids) + [0] * pad)
        label_rows.append(labels + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
    }


def sequence_logps(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    max_seq_length: int,
    device: torch.device,
    scoring_batch_size: int,
    require_grad: bool,
) -> torch.Tensor:
    results: list[torch.Tensor] = []
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        for start in range(0, len(pairs), scoring_batch_size):
            batch = collate_pairs(
                tokenizer,
                pairs[start : start + scoring_batch_size],
                max_seq_length=max_seq_length,
                device=device,
            )
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            targets = batch["labels"][:, 1:]
            mask = targets.ne(-100)
            safe_targets = targets.masked_fill(~mask, 0)
            selected = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            normalizer = torch.logsumexp(logits, dim=-1)
            token_logps = (selected - normalizer).float()
            results.append((token_logps * mask).sum(dim=-1))
            del batch, outputs, logits, targets, safe_targets, selected, normalizer, token_logps
    return torch.cat(results, dim=0)


def class_objective(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    *,
    method: str,
    beta: float,
    rp_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if method in {"representative_dpo", "random_permutation_dpo"}:
        gap = policy_scores[0, 0] - reference_scores[0, 0] - policy_scores[1, 0] + reference_scores[1, 0]
        return -F.logsigmoid(beta * gap), gap, None

    if method == "softmax_dpo":
        chosen_reward = policy_scores[0, 0] - reference_scores[0, 0]
        rejected_rewards = policy_scores[1] - reference_scores[1]
        all_rewards = torch.cat((chosen_reward.view(1), rejected_rewards), dim=0)
        log_probs = F.log_softmax(beta * all_rewards, dim=0)
        loss = -log_probs[0]
        gap = chosen_reward - torch.logsumexp(rejected_rewards, dim=0)
        return loss, gap, None

    if method == "permutation_augmented_dpo":
        per_pair_gap = policy_scores[0] - reference_scores[0] - policy_scores[1] + reference_scores[1]
        loss = -F.logsigmoid(beta * per_pair_gap).mean()
        return loss, per_pair_gap.mean(), None

    policy_class = torch.logsumexp(policy_scores, dim=-1)
    reference_class = torch.logsumexp(reference_scores, dim=-1)
    gap = policy_class[0] - reference_class[0] - policy_class[1] + reference_class[1]
    preference_loss = -F.logsigmoid(beta * gap)
    if method == "class_dpo":
        return preference_loss, gap, None

    logq_policy = policy_scores - policy_class.unsqueeze(-1)
    logq_reference = reference_scores - reference_class.unsqueeze(-1)
    q_policy = logq_policy.exp()
    within_kl = (q_policy * (logq_policy - logq_reference)).sum(dim=-1).mean()
    return preference_loss + rp_lambda * within_kl, gap, within_kl


def select_training_scores(
    example: TrainExample,
    *,
    method: str,
    tokenizer: Any,
    model: Any,
    max_seq_length: int,
    device: torch.device,
    scoring_batch_size: int,
    epoch_rng: random.Random,
    require_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = example.prompt
    if method in {"representative_dpo", "softmax_dpo"}:
        permutation_index = representative_permutation_index(
            {"rank_assignment_index": example.rank_assignment_index}
        ) - 1
        if method == "representative_dpo":
            pairs = [
                (prompt, example.preferred_sequences[permutation_index]["serialized"]),
                (prompt, example.rejected_sequences[permutation_index]["serialized"]),
            ]
            policy_scores = sequence_logps(
                model,
                tokenizer,
                pairs,
                max_seq_length=max_seq_length,
                device=device,
                scoring_batch_size=scoring_batch_size,
                require_grad=require_grad,
            ).view(2, 1)
            reference_scores = torch.tensor(
                [
                    [example.preferred_reference_logps[permutation_index]],
                    [example.rejected_reference_logps[permutation_index]],
                ],
                dtype=torch.float32,
                device=device,
            )
            return policy_scores, reference_scores

        chosen_pair = [
            (prompt, example.preferred_sequences[permutation_index]["serialized"]),
        ]
        rejected_pairs = [
            (prompt, sequence["serialized"])
            for sequence in example.rejected_sequences
        ]
        chosen_policy = sequence_logps(
            model,
            tokenizer,
            chosen_pair,
            max_seq_length=max_seq_length,
            device=device,
            scoring_batch_size=1,
            require_grad=require_grad,
        )
        rejected_policy = sequence_logps(
            model,
            tokenizer,
            rejected_pairs,
            max_seq_length=max_seq_length,
            device=device,
            scoring_batch_size=scoring_batch_size,
            require_grad=require_grad,
        )
        policy_scores = torch.stack(
            (
                chosen_policy.repeat(EXPECTED_CLASS_SIZE),
                rejected_policy,
            ),
            dim=0,
        )
        reference_scores = torch.tensor(
            [
                [example.preferred_reference_logps[permutation_index]] * EXPECTED_CLASS_SIZE,
                example.rejected_reference_logps,
            ],
            dtype=torch.float32,
            device=device,
        )
        return policy_scores, reference_scores

    if method == "random_permutation_dpo":
        permutation_index = epoch_rng.randrange(EXPECTED_CLASS_SIZE)
        pairs = [
            (prompt, example.preferred_sequences[permutation_index]["serialized"]),
            (prompt, example.rejected_sequences[permutation_index]["serialized"]),
        ]
        policy_scores = sequence_logps(
            model,
            tokenizer,
            pairs,
            max_seq_length=max_seq_length,
            device=device,
            scoring_batch_size=scoring_batch_size,
            require_grad=require_grad,
        ).view(2, 1)
        reference_scores = torch.tensor(
            [
                [example.preferred_reference_logps[permutation_index]],
                [example.rejected_reference_logps[permutation_index]],
            ],
            dtype=torch.float32,
            device=device,
        )
        return policy_scores, reference_scores

    pairs = [
        (prompt, sequence["serialized"])
        for sequence in example.preferred_sequences + example.rejected_sequences
    ]
    policy_scores = sequence_logps(
        model,
        tokenizer,
        pairs,
        max_seq_length=max_seq_length,
        device=device,
        scoring_batch_size=scoring_batch_size,
        require_grad=require_grad,
    ).view(2, EXPECTED_CLASS_SIZE)
    reference_scores = torch.tensor(
        [example.preferred_reference_logps, example.rejected_reference_logps],
        dtype=torch.float32,
        device=device,
    )
    return policy_scores, reference_scores


def memory_efficient_class_dpo_backward(
    example: TrainExample,
    *,
    tokenizer: Any,
    model: Any,
    beta: float,
    max_seq_length: int,
    device: torch.device,
    scoring_batch_size: int,
    gradient_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    policy_scores, reference_scores = select_training_scores(
        example,
        method="class_dpo",
        tokenizer=tokenizer,
        model=model,
        max_seq_length=max_seq_length,
        device=device,
        scoring_batch_size=scoring_batch_size,
        epoch_rng=random.Random(0),
        require_grad=False,
    )
    loss, gap, within_kl = class_objective(
        policy_scores,
        reference_scores,
        method="class_dpo",
        beta=beta,
        rp_lambda=0.0,
    )

    class_probabilities = torch.softmax(policy_scores, dim=-1)
    loss_gradient = -float(beta) * torch.sigmoid(-float(beta) * gap)
    member_weights = torch.cat(
        (
            loss_gradient * class_probabilities[0],
            -loss_gradient * class_probabilities[1],
        )
    )
    pairs = [
        (example.prompt, sequence["serialized"])
        for sequence in example.preferred_sequences + example.rejected_sequences
    ]
    for pair, weight in zip(pairs, member_weights):
        member_score = sequence_logps(
            model,
            tokenizer,
            [pair],
            max_seq_length=max_seq_length,
            device=device,
            scoring_batch_size=1,
            require_grad=True,
        )[0]
        (member_score * weight * float(gradient_scale)).backward()
        del member_score

    del class_probabilities, loss_gradient, member_weights
    return policy_scores, reference_scores, loss, gap, within_kl


def full_policy_scores(
    example: TrainExample,
    *,
    tokenizer: Any,
    model: Any,
    max_seq_length: int,
    device: torch.device,
    scoring_batch_size: int,
) -> torch.Tensor:
    pairs = [
        (example.prompt, sequence["serialized"])
        for sequence in example.preferred_sequences + example.rejected_sequences
    ]
    return sequence_logps(
        model,
        tokenizer,
        pairs,
        max_seq_length=max_seq_length,
        device=device,
        scoring_batch_size=scoring_batch_size,
        require_grad=False,
    ).view(2, EXPECTED_CLASS_SIZE)


def fixed_slate_ranking_metrics(
    example: TrainExample,
    *,
    tokenizer: Any,
    model: Any,
    max_seq_length: int,
    device: torch.device,
    scoring_batch_size: int,
) -> dict[str, Any]:
    rankings, gold_ranks = build_eval_rankings(example)
    pairs = [(example.prompt, completion) for completion in rankings]
    scores = sequence_logps(
        model,
        tokenizer,
        pairs,
        max_seq_length=max_seq_length,
        device=device,
        scoring_batch_size=scoring_batch_size,
        require_grad=False,
    )
    best_index = int(torch.argmax(scores).item())
    gold_rank = int(gold_ranks[best_index])
    return {
        "gold_rank": gold_rank,
        "reciprocal_rank": 1.0 / gold_rank,
        "top1": float(gold_rank == 1),
        "top3": float(gold_rank <= 3),
        "best_ranking": rankings[best_index],
    }


def evaluate_model(
    examples: Sequence[TrainExample],
    *,
    model: Any,
    tokenizer: Any,
    method: str,
    beta: float,
    rp_lambda: float,
    max_seq_length: int,
    device: torch.device,
    scoring_batch_size: int,
    compute_fixed_slate_mrr: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            policy_scores = full_policy_scores(
                example,
                tokenizer=tokenizer,
                model=model,
                max_seq_length=max_seq_length,
                device=device,
                scoring_batch_size=scoring_batch_size,
            )
            reference_scores = torch.tensor(
                [example.preferred_reference_logps, example.rejected_reference_logps],
                dtype=torch.float32,
                device=device,
            )
            eval_method = method if method != "random_permutation_dpo" else "permutation_augmented_dpo"
            if method == "softmax_dpo":
                policy_scores, reference_scores = select_training_scores(
                    example,
                    method="softmax_dpo",
                    tokenizer=tokenizer,
                    model=model,
                    max_seq_length=max_seq_length,
                    device=device,
                    scoring_batch_size=scoring_batch_size,
                    epoch_rng=random.Random(0),
                    require_grad=False,
                )
                loss, gap, within_kl = class_objective(
                    policy_scores,
                    reference_scores,
                    method="softmax_dpo",
                    beta=beta,
                    rp_lambda=rp_lambda,
                )
                chosen_reward = policy_scores[0, 0] - reference_scores[0, 0]
                rejected_rewards = policy_scores[1] - reference_scores[1]
                reward_vector = torch.cat((chosen_reward.view(1), rejected_rewards), dim=0)
                reference_reward_vector = torch.cat(
                    (
                        (reference_scores[0, 0] - reference_scores[0, 0]).view(1),
                        torch.zeros_like(rejected_rewards),
                    ),
                    dim=0,
                )
                logq_policy = F.log_softmax(beta * reward_vector, dim=0)
                logq_reference = F.log_softmax(beta * reference_reward_vector, dim=0)
                q_policy = logq_policy.exp()
                within_kls = (q_policy * (logq_policy - logq_reference)).sum()
                ratio_variance = (reward_vector - reference_reward_vector).var(unbiased=False)
                row = {
                    "question_id": example.question_id,
                    "rejected_rank": example.rejected_rank,
                    "loss": float(loss.cpu()),
                    "class_margin": float(gap.cpu()),
                    "preferred_union_share": float(q_policy[0].cpu()),
                    "within_kl": float(within_kls.cpu()),
                    "ratio_variance": float(ratio_variance.cpu()),
                    "max_member_share": float(q_policy.max().cpu()),
                    "reference_preferred_logsumexp": None,
                    "reference_rejected_logsumexp": None,
                    "policy_preferred_logsumexp": None,
                    "policy_rejected_logsumexp": None,
                    "objective_within_kl": None,
                }
            else:
                loss, gap, within_kl = class_objective(
                    policy_scores,
                    reference_scores,
                    method=eval_method,
                    beta=beta,
                    rp_lambda=rp_lambda,
                )
                policy_class = torch.logsumexp(policy_scores, dim=-1)
                reference_class = torch.logsumexp(reference_scores, dim=-1)
                logq_policy = policy_scores - policy_class.unsqueeze(-1)
                logq_reference = reference_scores - reference_class.unsqueeze(-1)
                q_policy = logq_policy.exp()
                within_kls = (q_policy * (logq_policy - logq_reference)).sum(dim=-1)
                ratio_variance = (policy_scores - reference_scores).var(dim=-1, unbiased=False).mean()
                row = {
                    "question_id": example.question_id,
                    "rejected_rank": example.rejected_rank,
                    "loss": float(loss.cpu()),
                    "class_margin": float(gap.cpu()),
                    "preferred_union_share": float(torch.sigmoid(policy_class[0] - policy_class[1]).cpu()),
                    "within_kl": float(within_kls.mean().cpu()),
                    "ratio_variance": float(ratio_variance.cpu()),
                    "max_member_share": float(q_policy.max(dim=-1).values.mean().cpu()),
                    "reference_preferred_logsumexp": float(reference_class[0].cpu()),
                    "reference_rejected_logsumexp": float(reference_class[1].cpu()),
                    "policy_preferred_logsumexp": float(policy_class[0].cpu()),
                    "policy_rejected_logsumexp": float(policy_class[1].cpu()),
                    "objective_within_kl": None if within_kl is None else float(within_kl.cpu()),
                }
            if compute_fixed_slate_mrr:
                row.update(
                    fixed_slate_ranking_metrics(
                        example,
                        tokenizer=tokenizer,
                        model=model,
                        max_seq_length=max_seq_length,
                        device=device,
                        scoring_batch_size=scoring_batch_size,
                    )
                )
            else:
                row.update(
                    {
                        "gold_rank": None,
                        "reciprocal_rank": None,
                        "top1": None,
                        "top3": None,
                        "best_ranking": None,
                    }
                )
            rows.append(row)
    model.train()

    loss_values = [row["loss"] for row in rows]
    class_margin_values = [row["class_margin"] for row in rows]
    within_kl_values = [row["within_kl"] for row in rows]
    ratio_variance_values = [row["ratio_variance"] for row in rows]
    preferred_union_values = [row["preferred_union_share"] for row in rows]
    max_member_values = [row["max_member_share"] for row in rows]
    summary = {
        "question_count": len(rows),
        "val_loss": float(np.mean(loss_values)) if loss_values else None,
        "class_margin": float(np.mean(class_margin_values)) if class_margin_values else None,
        "within_kl": float(np.mean(within_kl_values)) if within_kl_values else None,
        "ratio_variance": float(np.mean(ratio_variance_values)) if ratio_variance_values else None,
        "preferred_union_share": float(np.mean(preferred_union_values)) if preferred_union_values else None,
        "max_member_share": float(np.mean(max_member_values)) if max_member_values else None,
    }
    if compute_fixed_slate_mrr:
        reciprocal_ranks = [row["reciprocal_rank"] for row in rows]
        top1_values = [row["top1"] for row in rows]
        top3_values = [row["top3"] for row in rows]
        gold_ranks = [row["gold_rank"] for row in rows]
        summary.update(
            {
                "fixed_slate_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
                "fixed_slate_top1": float(np.mean(top1_values)) if top1_values else None,
                "fixed_slate_top3": float(np.mean(top3_values)) if top3_values else None,
                "fixed_slate_mean_gold_rank": float(np.mean(gold_ranks)) if gold_ranks else None,
            }
        )
    else:
        summary.update(
            {
                "fixed_slate_mrr": None,
                "fixed_slate_top1": None,
                "fixed_slate_top3": None,
                "fixed_slate_mean_gold_rank": None,
            }
        )
    return summary, rows


def metric_value(summary: dict[str, Any], selection_metric: str) -> float | None:
    value = summary.get(selection_metric)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def is_better(metric_name: str, candidate: float | None, best: float | None) -> bool:
    if candidate is None:
        return False
    if best is None:
        return True
    if metric_name == "val_loss":
        return candidate < best
    return candidate > best


def save_model_and_tokenizer(model: Any, tokenizer: Any, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def save_training_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    epoch_index: int,
    next_question_index: int,
    global_step: int,
    optimizer_step: int,
    history_rows: list[dict[str, Any]],
    eval_history_rows: list[dict[str, Any]],
    best_metric: float | None,
    best_summary: dict[str, Any] | None,
    best_eval_rows: list[dict[str, Any]] | None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_model_and_tokenizer(model, tokenizer, checkpoint_dir / "model")
    torch.save(
        optimizer.state_dict(),
        checkpoint_dir / "optimizer.pt",
    )
    write_json(
        checkpoint_dir / "state.json",
        {
            "epoch_index": int(epoch_index),
            "next_question_index": int(next_question_index),
            "global_step": int(global_step),
            "optimizer_step": int(optimizer_step),
            "best_metric": best_metric,
            "best_summary": best_summary,
            "history_rows": history_rows,
            "eval_history_rows": eval_history_rows,
        },
    )


def plot_training_history(
    output_dir: Path,
    *,
    history_rows: Sequence[dict[str, Any]],
    eval_history_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history_rows and not eval_history_rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if history_rows:
        train_steps = [row["question_step"] for row in history_rows]
        axes[0].plot(
            train_steps,
            [row["loss"] for row in history_rows],
            color="#2563eb",
            alpha=0.35,
            linewidth=1,
            label="training loss",
        )
        axes[1].plot(
            train_steps,
            [row["class_margin"] for row in history_rows],
            color="#16a34a",
            alpha=0.35,
            linewidth=1,
            label="training margin",
        )
    if eval_history_rows:
        eval_steps = [row["question_step"] for row in eval_history_rows]
        axes[0].plot(
            eval_steps,
            [row["val_loss"] for row in eval_history_rows],
            color="#dc2626",
            marker="o",
            linewidth=1.5,
            label="validation loss",
        )
        axes[1].plot(
            eval_steps,
            [row["class_margin"] for row in eval_history_rows],
            color="#ca8a04",
            marker="o",
            linewidth=1.5,
            label="validation margin",
        )
    axes[0].set_title("DPO loss")
    axes[1].set_title("Class margin")
    for axis in axes:
        axis.set_xlabel("Question step")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    resume_state: dict[str, Any] | None = None
    resume_dir: Path | None = None
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        resume_state_path = resume_dir / "state.json"
        if not resume_state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is missing {resume_state_path}.")
        resume_state = dict(read_json(resume_state_path))
        args.model_name = str(resume_dir / "model")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("This trainer currently expects a CUDA device.")

    set_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows = load_examples(Path(args.question_classes_with_reference_jsonl))
    if args.validation_question_classes_with_reference_jsonl:
        validation_rows = load_examples(Path(args.validation_question_classes_with_reference_jsonl))
    else:
        train_rows, validation_rows = split_train_validation(
            train_rows,
            ratio=float(args.validation_ratio),
            seed=int(args.split_seed),
        )

    if args.limit_train_questions is not None:
        train_rows = train_rows[: max(0, int(args.limit_train_questions))]
    if args.limit_validation_questions is not None:
        validation_rows = validation_rows[: max(0, int(args.limit_validation_questions))]
    if not train_rows:
        raise ValueError("No training questions available.")
    if not validation_rows:
        raise ValueError("No validation questions available.")

    output_dir = Path(args.output_dir)
    save_model_dir = Path(args.save_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    model, tokenizer = load_policy_model(args, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    if resume_dir is not None:
        optimizer_path = resume_dir / "optimizer.pt"
        if not optimizer_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is missing {optimizer_path}.")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))

    history_rows: list[dict[str, Any]] = [] if resume_state is None else list(resume_state.get("history_rows", []))
    eval_history_rows: list[dict[str, Any]] = [] if resume_state is None else list(resume_state.get("eval_history_rows", []))
    best_metric: float | None = None if resume_state is None else resume_state.get("best_metric")
    best_summary: dict[str, Any] | None = None if resume_state is None else resume_state.get("best_summary")
    best_eval_rows: list[dict[str, Any]] | None = None
    if resume_state is not None:
        best_eval_path = output_dir / "best_validation_per_question.jsonl"
        if best_eval_path.exists():
            best_eval_rows = [dict(row) for row in load_json_records(best_eval_path)]
    bad_evals = 0
    global_step = 0 if resume_state is None else int(resume_state.get("global_step", 0))
    optimizer_step = 0 if resume_state is None else int(resume_state.get("optimizer_step", 0))
    resume_epoch_index = 0 if resume_state is None else int(resume_state.get("epoch_index", 0))
    resume_question_index = 0 if resume_state is None else int(resume_state.get("next_question_index", 0))
    stop_training = False

    for epoch_index in range(int(math.ceil(float(args.num_train_epochs)))):
        if epoch_index < resume_epoch_index:
            continue
        epoch_rows = list(train_rows)
        epoch_rng = random.Random(int(args.seed) + epoch_index)
        epoch_rng.shuffle(epoch_rows)
        optimizer.zero_grad(set_to_none=True)
        question_offset = resume_question_index if epoch_index == resume_epoch_index else 0

        progress = tqdm(
            enumerate(epoch_rows[question_offset:], start=question_offset + 1),
            total=len(epoch_rows),
            desc=f"Epoch {epoch_index + 1}/{int(math.ceil(float(args.num_train_epochs)))}",
            unit="question",
            dynamic_ncols=True,
        )
        for question_index, example in progress:
            optimizer_updated = False
            if str(args.method) == "class_dpo":
                (
                    policy_scores,
                    reference_scores,
                    loss,
                    gap,
                    within_kl,
                ) = memory_efficient_class_dpo_backward(
                    example,
                    tokenizer=tokenizer,
                    model=model,
                    beta=float(args.beta),
                    max_seq_length=int(args.max_seq_length),
                    device=device,
                    scoring_batch_size=int(args.scoring_batch_size),
                    gradient_scale=1.0 / max(1, int(args.gradient_accumulation_steps)),
                )
                scaled_loss = None
            else:
                policy_scores, reference_scores = select_training_scores(
                    example,
                    method=str(args.method),
                    tokenizer=tokenizer,
                    model=model,
                    max_seq_length=int(args.max_seq_length),
                    device=device,
                    scoring_batch_size=int(args.scoring_batch_size),
                    epoch_rng=epoch_rng,
                )
                loss, gap, within_kl = class_objective(
                    policy_scores,
                    reference_scores,
                    method=str(args.method),
                    beta=float(args.beta),
                    rp_lambda=float(args.rp_lambda),
                )
                scaled_loss = loss / max(1, int(args.gradient_accumulation_steps))
                scaled_loss.backward()
            global_step += 1

            if global_step % max(1, int(args.gradient_accumulation_steps)) == 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                optimizer_updated = True

            history_rows.append(
                {
                    "epoch": epoch_index + 1,
                    "question_step": global_step,
                    "optimizer_step": optimizer_step,
                    "question_id": example.question_id,
                    "loss": float(loss.detach().cpu()),
                    "class_margin": float(gap.detach().cpu()),
                    "within_kl": None if within_kl is None else float(within_kl.detach().cpu()),
                    "method": args.method,
                }
            )
            progress.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                margin=f"{float(gap.detach().cpu()):.2f}",
                step=global_step,
            )
            del policy_scores, reference_scores, loss, gap, within_kl, scaled_loss

            should_eval = (
                int(args.eval_every_steps) > 0
                and global_step % int(args.eval_every_steps) == 0
            )
            reached_max_steps = int(args.max_steps) > 0 and global_step >= int(args.max_steps)
            if should_eval or reached_max_steps:
                summary, eval_rows = evaluate_model(
                    validation_rows,
                    model=model,
                    tokenizer=tokenizer,
                    method=str(args.method),
                    beta=float(args.beta),
                    rp_lambda=float(args.rp_lambda),
                    max_seq_length=int(args.max_seq_length),
                    device=device,
                    scoring_batch_size=int(args.scoring_batch_size),
                    compute_fixed_slate_mrr=not bool(args.skip_fixed_slate_mrr),
                )
                summary.update(
                    {
                        "epoch": epoch_index + 1,
                        "question_step": global_step,
                        "optimizer_step": optimizer_step,
                        "method": args.method,
                    }
                )
                eval_history_rows.append(summary)
                plot_training_history(
                    output_dir,
                    history_rows=history_rows,
                    eval_history_rows=eval_history_rows,
                )

                current_metric = metric_value(summary, str(args.selection_metric))
                if is_better(str(args.selection_metric), current_metric, best_metric):
                    best_metric = current_metric
                    best_summary = dict(summary)
                    best_eval_rows = eval_rows
                    bad_evals = 0
                    save_model_and_tokenizer(model, tokenizer, best_model_dir)
                    write_jsonl(output_dir / "best_validation_per_question.jsonl", best_eval_rows)
                    write_json(output_dir / "best_validation_summary.json", best_summary)
                else:
                    bad_evals += 1

                if int(args.early_stopping_patience) > 0 and bad_evals >= int(args.early_stopping_patience):
                    stop_training = True
                    break
                gc.collect()
                torch.cuda.empty_cache()

            if optimizer_updated:
                save_training_checkpoint(
                    output_dir / "checkpoint-last",
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    epoch_index=epoch_index,
                    next_question_index=question_index,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    history_rows=history_rows,
                    eval_history_rows=eval_history_rows,
                    best_metric=best_metric,
                    best_summary=best_summary,
                    best_eval_rows=best_eval_rows,
                )

            if reached_max_steps:
                stop_training = True
                break

        if not stop_training and global_step % max(1, int(args.gradient_accumulation_steps)) != 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        if stop_training:
            break

    if best_summary is None or best_eval_rows is None:
        best_summary, best_eval_rows = evaluate_model(
            validation_rows,
            model=model,
            tokenizer=tokenizer,
            method=str(args.method),
            beta=float(args.beta),
            rp_lambda=float(args.rp_lambda),
            max_seq_length=int(args.max_seq_length),
            device=device,
            scoring_batch_size=int(args.scoring_batch_size),
            compute_fixed_slate_mrr=not bool(args.skip_fixed_slate_mrr),
        )
        best_summary.update(
            {
                "epoch": None,
                "question_step": global_step,
                "optimizer_step": optimizer_step,
                "method": args.method,
            }
        )
        save_model_and_tokenizer(model, tokenizer, best_model_dir)
        write_jsonl(output_dir / "best_validation_per_question.jsonl", best_eval_rows)
        write_json(output_dir / "best_validation_summary.json", best_summary)

    final_train_summary, _ = evaluate_model(
        train_rows,
        model=model,
        tokenizer=tokenizer,
        method=str(args.method),
        beta=float(args.beta),
        rp_lambda=float(args.rp_lambda),
        max_seq_length=int(args.max_seq_length),
        device=device,
        scoring_batch_size=int(args.scoring_batch_size),
        compute_fixed_slate_mrr=False,
    )
    final_train_summary["split"] = "train"

    save_model_and_tokenizer(model, tokenizer, save_model_dir)

    write_jsonl(output_dir / "training_history.jsonl", history_rows)
    write_jsonl(output_dir / "evaluation_history.jsonl", eval_history_rows)
    plot_training_history(
        output_dir,
        history_rows=history_rows,
        eval_history_rows=eval_history_rows,
    )

    manifest = {
        "method": args.method,
        "model_name": args.model_name,
        "question_classes_with_reference_jsonl": args.question_classes_with_reference_jsonl,
        "validation_question_classes_with_reference_jsonl": args.validation_question_classes_with_reference_jsonl,
        "selection_metric": args.selection_metric,
        "rp_lambda": float(args.rp_lambda),
        "beta": float(args.beta),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_train_epochs": float(args.num_train_epochs),
        "max_steps": int(args.max_steps),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_seq_length": int(args.max_seq_length),
        "scoring_batch_size": int(args.scoring_batch_size),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "train_question_count": len(train_rows),
        "validation_question_count": len(validation_rows),
        "fixed_slate_mrr_enabled": not bool(args.skip_fixed_slate_mrr),
        "best_model_dir": str(best_model_dir),
        "final_model_dir": str(save_model_dir),
        "best_validation_summary_path": str(output_dir / "best_validation_summary.json"),
        "best_validation_per_question_path": str(output_dir / "best_validation_per_question.jsonl"),
        "training_history_path": str(output_dir / "training_history.jsonl"),
        "evaluation_history_path": str(output_dir / "evaluation_history.jsonl"),
        "training_curves_path": str(output_dir / "training_curves.png"),
    }
    final_summary = {
        "method": args.method,
        "train_summary": final_train_summary,
        "best_validation_summary": best_summary,
        "training_loss_distribution": summarize_numeric(
            row["loss"] for row in history_rows if isinstance(row.get("loss"), (int, float))
        ),
        "manifest": manifest,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", final_summary)
    write_json(output_dir / "training_complete.json", {"completed": True, "summary_path": str(output_dir / "summary.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
