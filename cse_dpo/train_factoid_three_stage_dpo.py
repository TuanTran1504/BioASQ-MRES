"""Sequential three-stage pairwise DPO for the strict-extractive factoid model.

This trainer intentionally consumes the staged JSONL pair files directly.  It does
not use the evidence-support fields as a filter: class labels have already been
used to construct the three preference stages.  At each stage the policy is
initialized from the previous stage's adapter and the reference scores are
frozen before any update in that stage.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
import argparse
import gc
import hashlib
from dataclasses import asdict, dataclass, replace
from dataclasses import field
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel, prepare_model_for_kbit_training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_first400_c3_c1_v1_gold_c3/staged_curriculum_pairs"
DEFAULT_BASE = ROOT / "models/Qwen2.5-0.5B-Instruct"
DEFAULT_BASE_3B = ROOT / "models/Qwen2.5-3B-Instruct"
DEFAULT_INITIAL = ROOT / "Artifacts/Factoid_SFT/models/evidence_grounded_per_supported_alias_qwen25_05b_lora_dropout_005_strict_extractive/adapter_best_evidence_mrr"
DEFAULT_INITIAL_3B = ROOT / "Artifacts/Factoid_SFT/models/evidence_grounded_per_supported_alias_qwen25_3b_lora_dropout_005_strict_extractive/adapter_best_evidence_mrr"
SPLIT_SFT80_DPO20_INITIAL_05B = ROOT / "Artifacts/Factoid_SFT/models/evidence_grounded_per_supported_alias_qwen25_05b_sft80_dpo20_seed3407_lora_r32_alpha32_dropout_005_strict_extractive/adapter_best_evidence_mrr"
SPLIT_SFT80_DPO20_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/qwen25_05b_sft80_dpo20_r32_step175_dpo226_s10_t07_seed3407_gold_c3/staged_curriculum_pairs"
SPLIT_SFT80_DPO20_STAGE1 = SPLIT_SFT80_DPO20_STAGED_ROOT / "dpo_stage1_concept_learning_all_pairs.jsonl"
SPLIT_SFT80_DPO20_STAGE2 = SPLIT_SFT80_DPO20_STAGED_ROOT / "dpo_stage2_format_alignment_all_pairs.jsonl"
FULL_GOLD_SUPPORTED_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported/merged/staged_curriculum_pairs"
STRICT_EQUIVALENCE_V2_1130_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v2_strict_equivalence_gold_c3_gold_supported_1130/merged/staged_curriculum_pairs"
STRICT_EQUIVALENCE_V2_1130_STAGE1_JACCARD0_WEAKNORM_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v2_strict_equivalence_gold_c3_gold_supported_1130/merged/staged_curriculum_pairs_stage1_jaccard0_weaknorm"
FIRST400_DPO_D_TIE_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_first400_c3_c1_v1_gold_c3/staged_curriculum_pairs_dpo_d_ties_weak_surface_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "Artifacts/cse_dpo/three_stage_dpo_qwen25_05b_c3_c1_curriculum_v1_full"
MODEL_PRESETS = {
    "qwen25_05b": {
        "base_model": DEFAULT_BASE,
        "initial_adapter": DEFAULT_INITIAL,
        "output_root": DEFAULT_OUTPUT_ROOT,
    },
    "qwen25_3b": {
        "base_model": DEFAULT_BASE_3B,
        "initial_adapter": DEFAULT_INITIAL_3B,
        "output_root": ROOT / "Artifacts/cse_dpo/three_stage_dpo_qwen25_3b_c3_c1_curriculum_v1_full",
    },
    "qwen25_05b_sft80_dpo20_r32": {
        "base_model": DEFAULT_BASE,
        "initial_adapter": SPLIT_SFT80_DPO20_INITIAL_05B,
        "output_root": ROOT / "Artifacts/cse_dpo/three_stage_dpo_qwen25_05b_sft80_dpo20_r32_curriculum_full",
    },
}
STAGED_DATASET_PRESETS = {
    "legacy_first400": DEFAULT_STAGED_ROOT,
    "gold_supported_full_merged": FULL_GOLD_SUPPORTED_STAGED_ROOT,
    "gold_supported_strict_equivalence_v2_1130": STRICT_EQUIVALENCE_V2_1130_STAGED_ROOT,
    "gold_supported_strict_equivalence_v2_1130_stage1_jaccard0": STRICT_EQUIVALENCE_V2_1130_STAGE1_JACCARD0_WEAKNORM_STAGED_ROOT,
    "legacy_first400_dpo_d_ties": FIRST400_DPO_D_TIE_STAGED_ROOT,
    "sft80_dpo20_r32_step175_dpo226_gpt": SPLIT_SFT80_DPO20_STAGED_ROOT,
}
SUPPORTED_OPTIMIZERS = {
    "rmsprop",
    "adamw",
    "adam",
    "sgd",
    "adafactor",
    "paged_adamw_8bit",
    "adamw_8bit",
}
SUPPORTED_OBJECTIVES = {"dpo", "dpo_d", "apo_zero", "apo_down", "dpo_adaptive_nll", "cal_dpo"}
SUPPORTED_SELECTION_METRICS = {
    "dev_mrr",
    "dpo_eval_loss",
    "objective_eval_loss",
    "semantic_accuracy",
    "semantic_supported_accuracy",
    "retention_score",
}


@dataclass
class Config:
    model_preset: str = "qwen25_05b"
    dataset_preset: str = "legacy_first400"
    staged_root: str = str(DEFAULT_STAGED_ROOT)
    base_model: str = str(DEFAULT_BASE)
    initial_adapter: str = str(DEFAULT_INITIAL)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    seed: int = 3407
    beta: float = 0.1
    objective: str = "dpo"  # Options: dpo, dpo_d, apo_zero, apo_down, dpo_adaptive_nll, cal_dpo.
    dpo_d_nu: float = 1.0  # Davidson tie mass; nu=1 gives P(tie)=0.5 at zero reward margin.
    tie_label_field: str = "preference_label"  # Values "win" and "tie" for objective=dpo_d.
    adaptive_nll_weight: float = 0.05  # RISE-style chosen-answer NLL weight used by dpo_adaptive_nll.
    # Optional token-mean LM loss for a second, diagnostic prompt attached to
    # each preference row. This keeps the deployed task answer-only while
    # teaching the same adapter to identify why a rejected answer is wrong.
    auxiliary_diagnostic_weight: float = 0.0
    learning_rate: float = 5e-6
    stage3_learning_rate: float = 1e-6
    optimizer: str = "RMSprop"
    weight_decay: float = 0.01
    epochs: int = 8
    batch_size: int = 1
    gradient_accumulation: int = 8
    max_length: int = 8192
    train_backprop_max_length: int | None = None  # If set, skip train pairs above this token length before backward.
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    eval_fraction: float = 0.2
    eval_seed: int = 3407
    eval_every_updates: int = 0  # 0 means evaluate at the end of every epoch.
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    selection_metric: str = "dev_mrr"
    retention_loss_penalty: float = 2.0  # retention_score = newly_correct - penalty * newly_wrong.
    retention_min_rate: float = 0.98  # Checkpoints below this baseline-correct retention are ineligible.
    retention_anchor_weight: float = 0.0  # Sequence log-probability matching to the incoming stage policy.
    retention_anchor_max_examples: int | None = None
    l2sp_weight: float = 0.0  # Anchor trainable parameters to their incoming-stage values.
    allow_truncated_examples: bool = False
    drop_truncated_examples: bool = True
    append_eos_to_completions: bool = True
    evaluate_generated_dev: bool = True
    dev_eval_questions: int | None = None  # Set by smoke mode; None means all dev questions.
    dev_source_input: str = str(ROOT / "data/BioASQ_factoid_sft_prepared/single_answer_full_resources_qwen25_05b/eval_prepared.json")
    prompt_registry: str = str(ROOT / "prompts/factoid_single_answer_aligned.json")
    prompt_ref: str = "factoid-single-answer-extractive-v1"
    # Keep generated-dev scoring independent of the stage training/backprop
    # budget.  SFT checkpoint selection uses the same 4096-token protocol.
    generated_eval_max_seq_length: int | None = None
    generated_eval_max_new_tokens: int = 64
    semantic_judge_enabled: bool = False
    semantic_judge_model: str = "gpt-4.1-mini-2025-04-14"
    semantic_judge_endpoint: str = "https://api.openai.com/v1/chat/completions"
    semantic_judge_api_key_file: str = str(ROOT / "open_ai_api.txt")
    semantic_judge_max_new_calls: int | None = None
    semantic_judge_retries: int = 2
    semantic_judge_timeout_seconds: int = 120
    semantic_judge_cache_root: str | None = None
    attn_implementation: str = "sdpa"
    lora_dropout: float | None = None  # None keeps the dropout stored in the loaded adapter.
    stage_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    smoke_test: bool = True
    smoke_pairs: int = 8
    stop_after_stage: str | None = None  # Inclusive stage name, e.g. "concept_learning" for a stage-1-only run.
    skip_stages: list[str] = field(default_factory=list)  # Stage names to bypass while keeping the current adapter.
    stage_output_suffixes: dict[str, str] = field(default_factory=dict)  # Optional per-stage folder suffix for retry runs.
    allow_mixed_objective_resume: bool = False  # Reuse completed old stages while changing objective for later stages.
    include_auxiliary_c1_over_c0: bool = False
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        if self.model_preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown model_preset {self.model_preset!r}; expected one of {sorted(MODEL_PRESETS)}")
        if self.dataset_preset not in STAGED_DATASET_PRESETS:
            raise ValueError(
                f"Unknown dataset_preset {self.dataset_preset!r}; "
                f"expected one of {sorted(STAGED_DATASET_PRESETS)}"
            )
        preset = MODEL_PRESETS[self.model_preset]
        if self.staged_root == str(DEFAULT_STAGED_ROOT):
            self.staged_root = str(STAGED_DATASET_PRESETS[self.dataset_preset])
        if self.base_model == str(DEFAULT_BASE):
            self.base_model = str(preset["base_model"])
        if self.initial_adapter == str(DEFAULT_INITIAL):
            self.initial_adapter = str(preset["initial_adapter"])
        if self.output_root == str(DEFAULT_OUTPUT_ROOT):
            self.output_root = str(preset["output_root"])


STAGES = (
    ("concept_learning", "dpo_stage1_concept_learning_all_pairs.jsonl"),
    ("format_alignment", "dpo_stage2_format_alignment_all_pairs.jsonl"),
    ("hierarchical_ranking", "dpo_stage3_hierarchical_ranking_all_pairs.jsonl"),
)
AUXILIARY = ("auxiliary_legacy_c1_over_c0", "dpo_auxiliary_legacy_c1_over_c0_all_pairs.jsonl")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def load_stage_rows(cfg: Config) -> dict[str, list[dict[str, Any]]]:
    root = Path(cfg.staged_root)
    stage_specs = list(STAGES) + ([AUXILIARY] if cfg.include_auxiliary_c1_over_c0 else [])
    result: dict[str, list[dict[str, Any]]] = {}
    for name, filename in stage_specs:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _jsonl(path)
        for row in rows:
            for field in ("pair_id", "question_id", "prompt", "chosen", "rejected"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{name}: missing non-empty {field} in {row.get('pair_id')}")
            if row.get("split", "train") != "train":
                raise ValueError(f"{name}: non-train pair {row['pair_id']}")
        if cfg.smoke_test:
            rows = rows[: cfg.smoke_pairs]
        if not rows:
            print(f"[{name}] no pairs available; skipping this stage")
            continue
        result[name] = rows
    return result


def _load_token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _encode_pair(
    tokenizer,
    row: dict[str, Any],
    max_length: int,
    *,
    append_eos: bool = True,
    allow_truncation: bool = False,
) -> dict[str, Any]:
    prompt_ids = _load_token_ids(tokenizer, row["prompt"])
    chosen_ids = _load_token_ids(tokenizer, row["chosen"])
    rejected_ids = _load_token_ids(tokenizer, row["rejected"])
    if append_eos and tokenizer.eos_token_id is not None:
        if not chosen_ids or chosen_ids[-1] != tokenizer.eos_token_id:
            chosen_ids = chosen_ids + [tokenizer.eos_token_id]
        if not rejected_ids or rejected_ids[-1] != tokenizer.eos_token_id:
            rejected_ids = rejected_ids + [tokenizer.eos_token_id]
    if not chosen_ids or not rejected_ids:
        raise ValueError(f"Empty completion for {row['pair_id']}")

    def one(prefix_ids, completion_ids, *, sequence_name: str):
        if len(prefix_ids) + len(completion_ids) > max_length and not allow_truncation:
            raise ValueError(
                f"{row['pair_id']}: tokenized {sequence_name} length "
                f"{len(prefix_ids) + len(completion_ids)} exceeds max_length={max_length}. "
                "Increase max_length, shorten prompts, or set allow_truncated_examples=True."
            )
        # If explicitly allowed, preserve the prompt prefix and truncate only
        # the prompt tail. The default path above prevents silent evidence loss.
        keep_prompt = max(1, max_length - len(completion_ids))
        p = prefix_ids[:keep_prompt]
        ids = p + completion_ids
        labels = [-100] * len(p) + completion_ids
        return ids[:max_length], labels[:max_length]

    c_ids, c_labels = one(prompt_ids, chosen_ids, sequence_name="chosen answer")
    r_ids, r_labels = one(prompt_ids, rejected_ids, sequence_name="rejected answer")
    encoded = {"row": row, "chosen_ids": c_ids, "chosen_labels": c_labels,
               "rejected_ids": r_ids, "rejected_labels": r_labels,
               "prompt_tokens": len(prompt_ids), "chosen_tokens": len(chosen_ids),
               "rejected_tokens": len(rejected_ids)}

    auxiliary_prompt = row.get("auxiliary_prompt")
    auxiliary_target = row.get("auxiliary_target")
    if bool(str(auxiliary_prompt or "").strip()) != bool(str(auxiliary_target or "").strip()):
        raise ValueError(
            f"{row['pair_id']}: auxiliary_prompt and auxiliary_target must either both be set or both be absent"
        )
    if str(auxiliary_prompt or "").strip():
        auxiliary_prompt_ids = _load_token_ids(tokenizer, str(auxiliary_prompt))
        auxiliary_target_ids = _load_token_ids(tokenizer, str(auxiliary_target))
        if append_eos and tokenizer.eos_token_id is not None:
            if not auxiliary_target_ids or auxiliary_target_ids[-1] != tokenizer.eos_token_id:
                auxiliary_target_ids = auxiliary_target_ids + [tokenizer.eos_token_id]
        if not auxiliary_target_ids:
            raise ValueError(f"Empty auxiliary target for {row['pair_id']}")
        auxiliary_ids, auxiliary_labels = one(
            auxiliary_prompt_ids,
            auxiliary_target_ids,
            sequence_name="auxiliary diagnostic",
        )
        encoded.update({
            "auxiliary_ids": auxiliary_ids,
            "auxiliary_labels": auxiliary_labels,
            "auxiliary_prompt_tokens": len(auxiliary_prompt_ids),
            "auxiliary_target_tokens": len(auxiliary_target_ids),
        })
    return encoded


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": [x["row"] for x in batch]}
    auxiliary_presence = ["auxiliary_ids" in item for item in batch]
    if any(auxiliary_presence) and not all(auxiliary_presence):
        raise ValueError("A batch cannot mix rows with and without auxiliary diagnostic targets")
    sides = ["chosen", "rejected"] + (["auxiliary"] if all(auxiliary_presence) else [])
    for side in sides:
        ids = [x[f"{side}_ids"] for x in batch]
        labels = [x[f"{side}_labels"] for x in batch]
        width = max(map(len, ids))
        out[f"{side}_input_ids"] = torch.tensor([a + [pad_id] * (width - len(a)) for a in ids], dtype=torch.long)
        out[f"{side}_attention_mask"] = torch.tensor([[1] * len(a) + [0] * (width - len(a)) for a in ids], dtype=torch.long)
        out[f"{side}_labels"] = torch.tensor([a + [-100] * (width - len(a)) for a in labels], dtype=torch.long)
    return out


def _device(model):
    return next(model.parameters()).device


def _sequence_logps(model, input_ids, attention_mask, labels, train: bool):
    """Score completion tokens without materializing prompt-token logits.

    The standard DPO notebook uses this ``logits_to_keep`` path.  Full
    vocabulary logits for an 8k-token prompt can consume several GiB during
    backward even with gradient checkpointing.
    """
    if input_ids.size(0) != 1:
        # The target positions can differ after padding. Keep the safe memory
        # path by scoring each example independently when a larger batch is set.
        return torch.cat([
            _sequence_logps(model, input_ids[i:i + 1], attention_mask[i:i + 1], labels[i:i + 1], train)
            for i in range(input_ids.size(0))
        ], dim=0)
    device = _device(model)
    ids = input_ids.to(device)
    mask = attention_mask.to(device)
    target_labels = labels.to(device)
    target_positions = torch.nonzero(target_labels[0] != -100, as_tuple=False).squeeze(-1)
    if target_positions.numel() == 0:
        raise ValueError("Completion contains no scored tokens")
    logit_positions = target_positions - 1
    with torch.set_grad_enabled(train):
        result = model(input_ids=ids, attention_mask=mask, logits_to_keep=logit_positions, use_cache=False)
        logits = result.logits
        targets = target_labels[:, target_positions].contiguous()
        seq_logps = torch.zeros(1, dtype=torch.float32, device=logits.device)
        for start in range(0, logits.size(1), 128):
            end = min(start + 128, logits.size(1))
            chunk_logits = logits[:, start:end, :]
            chunk_targets = targets[:, start:end]

            def token_logps_fn(chunk, target):
                chunk = chunk.float()
                selected = torch.gather(chunk, -1, target.unsqueeze(-1)).squeeze(-1)
                return selected - torch.logsumexp(chunk, dim=-1)

            if train:
                token_logps = checkpoint(token_logps_fn, chunk_logits, chunk_targets, use_reentrant=False)
            else:
                token_logps = token_logps_fn(chunk_logits, chunk_targets)
            seq_logps = seq_logps + token_logps.sum(dim=-1).float()
        return seq_logps


@torch.no_grad()
def _reference_scores(model, encoded, tokenizer, cfg: Config):
    model.eval()
    loader = DataLoader(encoded, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id))
    scores = []
    for batch in loader:
        c = _sequence_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"], False)
        r = _sequence_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"], False)
        scores.extend(zip(c.float().cpu().tolist(), r.float().cpu().tolist()))
    return scores


@torch.no_grad()
def _reference_chosen_scores(model, encoded, tokenizer, cfg: Config):
    model.eval()
    loader = DataLoader(
        encoded,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    scores = []
    for batch in loader:
        chosen = _sequence_logps(
            model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
            False,
        )
        scores.extend(chosen.float().cpu().tolist())
    return scores


def _global_eval_question_ids(stages: dict[str, list[dict[str, Any]]], cfg: Config) -> set[str]:
    """Create one question-level held-out set shared by every stage."""
    if cfg.eval_fraction <= 0:
        return set()
    qids = sorted({row["question_id"] for rows in stages.values() for row in rows})
    if len(qids) < 2:
        return set()
    rng = random.Random(f"{cfg.eval_seed}:global")
    rng.shuffle(qids)
    n_eval = max(1, min(len(qids) - 1, round(len(qids) * cfg.eval_fraction)))
    return set(qids[:n_eval])


def _split_stage_rows(rows: list[dict[str, Any]], cfg: Config, stage: str, eval_qids: set[str] | None = None):
    """Split by a global question-level holdout shared across stages."""
    if cfg.eval_fraction <= 0 or len(rows) < 2:
        return rows, []
    if eval_qids is None:
        eval_qids = _global_eval_question_ids({stage: rows}, cfg)
    train = [row for row in rows if row["question_id"] not in eval_qids]
    valid = [row for row in rows if row["question_id"] in eval_qids]
    if not train:
        raise ValueError(f"{stage}: global holdout consumed all training pairs; reduce eval_fraction")
    return train, valid


def _setting(cfg: Config, stage: str, key: str, default):
    return cfg.stage_settings.get(stage, {}).get(key, default)


def _normalize_optimizer_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _make_optimizer(parameters, cfg: Config, stage: str):
    optimizer_name = _normalize_optimizer_name(cfg.optimizer)
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer for stage {stage}: {cfg.optimizer!r}. "
            f"Choose one of {sorted(SUPPORTED_OPTIMIZERS)}."
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if optimizer_name == "adafactor":
        return Adafactor(
            parameters,
            lr=cfg.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    if optimizer_name in {"paged_adamw_8bit", "adamw_8bit"}:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(
                f"{cfg.optimizer!r} requires bitsandbytes to be installed in the training environment."
            ) from exc
        optimizer_cls = (
            bnb.optim.PagedAdamW8bit if optimizer_name == "paged_adamw_8bit"
            else bnb.optim.AdamW8bit
        )
        return optimizer_cls(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise AssertionError(f"Unhandled optimizer: {optimizer_name}")


def _build_dev_eval_rows(cfg: Config, tokenizer):
    from cse_dpo.generated_bioasq_eval import build_generated_eval_rows, load_gold_examples
    from src.prompt_registry import resolve_prompt_bundle
    from src.utility.config import QUESTION_INSTRUCTIONS
    from src.utility.eval_dataset import load_eval_examples, render_prompt

    path = Path(cfg.dev_source_input)
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = resolve_prompt_bundle(Path(cfg.prompt_registry), cfg.prompt_ref, QUESTION_INSTRUCTIONS)
    args = argparse.Namespace(
        question_types=["factoid"], max_resources=0, max_resource_chars=0,
        resource_selection="first", resource_granularity="document", resource_window_mode="single",
        resource_reranker_model=None, resource_reranker_article_model=None,
        resource_reranker_device=None, resource_reranker_batch_size=1,
        local_files_only=True, limit=None,
    )
    examples = load_eval_examples([path], args, bundle["instructions"])
    if cfg.dev_eval_questions is not None:
        examples = examples[: int(cfg.dev_eval_questions)]
    gold = load_gold_examples([path])
    prompt_rows = [{
        "question_id": example.question_id,
        "prompt": render_prompt(tokenizer, example, chat_template=bundle.get("chat_template", "qwen-2.5"), prompt_format="chat"),
    } for example in examples]
    rows, missing = build_generated_eval_rows(prompt_rows, gold)
    if missing:
        raise ValueError(f"Generated evaluation is missing {len(missing)} gold questions; first IDs: {missing[:5]}")
    return rows, bundle["prompt_id"]


def _row_is_tie(row: dict[str, Any], cfg: Config) -> bool:
    """Read an explicit tie label without weakening punctuation or span boundaries."""
    if "is_tie" in row:
        return bool(row["is_tie"])
    value = row.get(cfg.tie_label_field, row.get("tie_label", "win"))
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"tie", "tied", "equal", "equivalent", "no_preference"}:
        return True
    if normalized in {"win", "clear_win", "preference", "preferred", "0", ""}:
        return False
    raise ValueError(
        f"Unsupported tie label {value!r} for pair {row.get('pair_id')!r}; "
        "expected 'win' or 'tie'."
    )


def _tie_targets(rows: list[dict[str, Any]], cfg: Config, *, device, dtype):
    return torch.tensor([_row_is_tie(row, cfg) for row in rows], device=device, dtype=dtype)


def _preference_losses(policy_c, policy_r, refs, cfg: Config, tie_targets=None):
    chosen_logratios = policy_c - refs[:, 0]
    rejected_logratios = policy_r - refs[:, 1]
    if cfg.objective == "dpo":
        delta = chosen_logratios - rejected_logratios
        return -torch.nn.functional.logsigmoid(cfg.beta * delta)
    if cfg.objective == "dpo_d":
        # Davidson DPO, Chen et al. (NeurIPS 2025), Eqs. 12-16.  Clear-win
        # rows maximize P(chosen > rejected); tie rows maximize P(chosen ~
        # rejected).  At nu=1 and zero reward margin, P(tie)=1/2.
        if cfg.dpo_d_nu <= 0:
            raise ValueError("dpo_d requires dpo_d_nu > 0")
        if tie_targets is None:
            tie_targets = torch.zeros_like(policy_c)
        d = cfg.beta * (chosen_logratios - rejected_logratios)
        log_two_nu = math.log(2.0 * float(cfg.dpo_d_nu))
        # log(1 + exp(-d) + 2*nu*exp(-d/2)), evaluated stably.
        log_denom = torch.logsumexp(
            torch.stack((torch.zeros_like(d), -d, torch.full_like(d, log_two_nu) - 0.5 * d)),
            dim=0,
        )
        win_loss = log_denom
        tie_loss = log_denom - log_two_nu + 0.5 * d
        return torch.where(tie_targets.to(dtype=torch.bool), tie_loss, win_loss)
    if cfg.objective == "dpo_adaptive_nll":
        # RISE-style subtle-error-aware DPO: keep the standard DPO contrast,
        # and add chosen-answer NLL only when the policy has pushed the chosen
        # completion below the stage reference probability. This is useful for
        # near-neighbor pairs such as C3>C2, where plain DPO can improve the
        # pair margin while damaging the chosen answer likelihood.
        delta = chosen_logratios - rejected_logratios
        dpo = -torch.nn.functional.logsigmoid(cfg.beta * delta)
        chosen_nll = -policy_c
        adaptive_gate = (chosen_logratios < 0).to(dtype=dpo.dtype).detach()
        return dpo + float(cfg.adaptive_nll_weight) * adaptive_gate * chosen_nll
    if cfg.objective == "cal_dpo":
        # Cal-DPO (Calibrated Direct Preference Optimization) calibrates the
        # implicit rewards log(pi_theta/pi_ref) toward +1/(2 beta) for chosen
        # responses and -1/(2 beta) for rejected responses, while keeping the
        # BT contrastive loss on the unscaled reward gap.
        if cfg.beta <= 0:
            raise ValueError("cal_dpo requires beta > 0")
        target = 0.5 / float(cfg.beta)
        contrastive = -torch.nn.functional.logsigmoid(chosen_logratios - rejected_logratios)
        calibration = (chosen_logratios - target).pow(2) + (rejected_logratios + target).pow(2)
        return contrastive + calibration
    if cfg.objective == "apo_zero":
        # TRL APO-zero / Eq. 7: increase chosen likelihood and decrease rejected likelihood.
        return (1 - torch.sigmoid(cfg.beta * chosen_logratios)) + torch.sigmoid(cfg.beta * rejected_logratios)
    if cfg.objective == "apo_down":
        # TRL APO-down / Eq. 8: decrease both, with stronger pressure on rejected.
        delta = chosen_logratios - rejected_logratios
        return torch.sigmoid(cfg.beta * chosen_logratios) + (1 - torch.sigmoid(cfg.beta * delta))
    raise ValueError(f"Unsupported objective {cfg.objective!r}; expected one of {sorted(SUPPORTED_OBJECTIVES)}")


def _preference_margin(policy_c, policy_r, refs):
    return (policy_c - refs[:, 0]) - (policy_r - refs[:, 1])


@torch.no_grad()
def _dpo_eval_loss(model, encoded, reference, tokenizer, cfg: Config):
    if not encoded:
        return {"question_count": 0, "pair_count": 0, "dpo_eval_loss": None, "pairwise_accuracy": None}
    model.eval()
    ref_by_id = {item["row"]["pair_id"]: reference[i] for i, item in enumerate(encoded)}
    loader = DataLoader(encoded, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id))
    losses, auxiliary_losses, correct, tie_margins = [], [], [], []
    for batch in loader:
        c = _sequence_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"], False)
        r = _sequence_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"], False)
        refs = torch.tensor([ref_by_id[row["pair_id"]] for row in batch["rows"]], device=c.device, dtype=c.dtype)
        margin = _preference_margin(c, r, refs)
        ties = _tie_targets(batch["rows"], cfg, device=c.device, dtype=c.dtype)
        objective_losses = _preference_losses(c, r, refs, cfg, ties)
        losses.extend(objective_losses.float().cpu().tolist())
        if cfg.auxiliary_diagnostic_weight > 0:
            if "auxiliary_input_ids" not in batch:
                raise ValueError("Auxiliary diagnostic loss is enabled but the eval batch has no target")
            auxiliary_logps = _sequence_logps(
                model,
                batch["auxiliary_input_ids"],
                batch["auxiliary_attention_mask"],
                batch["auxiliary_labels"],
                False,
            )
            auxiliary_counts = (batch["auxiliary_labels"] != -100).sum(dim=1).to(
                device=auxiliary_logps.device,
                dtype=auxiliary_logps.dtype,
            ).clamp_min(1)
            auxiliary_losses.extend(
                (-auxiliary_logps / auxiliary_counts).float().cpu().tolist()
            )
        for value, is_tie in zip(margin.float().cpu().tolist(), ties.bool().cpu().tolist()):
            if is_tie:
                tie_margins.append(abs(value))
            else:
                correct.append(float(value > 0))
    mean_loss = float(sum(losses) / len(losses))
    auxiliary_mean = (
        float(sum(auxiliary_losses) / len(auxiliary_losses))
        if auxiliary_losses
        else None
    )
    joint_mean = (
        mean_loss + float(cfg.auxiliary_diagnostic_weight) * auxiliary_mean
        if auxiliary_mean is not None
        else mean_loss
    )
    return {"question_count": len({item["row"]["question_id"] for item in encoded}),
            "pair_count": len(encoded), "dpo_eval_loss": mean_loss,
            "auxiliary_diagnostic_eval_loss": auxiliary_mean,
            "objective_eval_loss": joint_mean, "objective": cfg.objective,
            "clear_win_pair_count": len(correct),
            "tie_pair_count": len(tie_margins),
            "pairwise_accuracy": float(sum(correct) / len(correct)) if correct else None,
            "mean_abs_tie_margin": float(sum(tie_margins) / len(tie_margins)) if tie_margins else None}



def _semantic_judge_output_root(cfg: Config, stage_dir: Path) -> Path:
    if cfg.semantic_judge_cache_root:
        return Path(cfg.semantic_judge_cache_root)
    return stage_dir / "semantic_judge_cache"


def _gold_aliases(example) -> list[str]:
    from src.utility.bioasq_format import exact_answer_groups
    aliases: list[str] = []
    seen: set[str] = set()
    for group in exact_answer_groups(example, example.question_type):
        for alias in group:
            alias = str(alias).strip()
            if alias and alias not in seen:
                seen.add(alias)
                aliases.append(alias)
    return aliases


def _candidate_from_prediction(prediction: str, question_type: str) -> tuple[str, bool, int]:
    from src.utility.bioasq_format import parse_prediction_items
    items = parse_prediction_items(prediction, question_type)
    if len(items) == 1:
        return str(items[0]).strip(), True, 1
    fallback = str(prediction or "").strip()
    return fallback, False, len(items)


def _score_generated_semantics(
    cfg: Config,
    stage_dir: Path,
    dev_rows: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    *,
    step: int,
) -> dict[str, Any]:
    """Classify generated dev predictions as C3/C2/C1 for selection.

    C3 is deterministic exact accepted-alias match.  Non-exact predictions are
    sent to the same C3/C2/C1 judge used for candidate-bank annotation.
    ``semantic_accuracy`` is the fraction of dev questions classified as C3 or
    C2.  The metric is answer equivalence, not an official BioASQ score.
    """
    if not eval_metrics:
        return {
            "semantic_accuracy": None,
            "semantic_correct_count": 0,
            "semantic_evaluated_count": 0,
            "semantic_judge_status": "no_eval_metrics",
        }
    from cse_dpo.candidate_bank_class_judge import CandidateBankClassJudge, exact_norm, extract_snippets, format_answer

    examples = {row["example"].question_id: row["example"] for row in dev_rows}
    records: list[dict[str, Any]] = []
    preclassified: list[dict[str, Any]] = []
    for metric in eval_metrics:
        qid = str(metric.get("question_id") or "")
        example = examples.get(qid)
        if example is None:
            continue
        aliases = _gold_aliases(example)
        candidate, valid_single, parsed_count = _candidate_from_prediction(
            str(metric.get("prediction") or ""), example.question_type
        )
        exact = valid_single and any(exact_norm(candidate) == exact_norm(alias) for alias in aliases)
        base = {
            "question_id": qid,
            "question": example.body,
            "gold_aliases": aliases,
            "candidate": candidate,
            "candidate_output": format_answer(candidate),
            "source_model": f"generated_dev_step_{step}",
            "response_id": f"{qid}__step_{step}",
            "sample_id": step,
            "prediction": metric.get("prediction"),
            "valid_single_answer": valid_single,
            "parsed_answer_count": parsed_count,
            "snippets": extract_snippets(list(example.resources)),
        }
        if exact:
            preclassified.append({
                **base,
                "exact_match": True,
                "class": "C3",
                "semantic_correct": True,
                "related": True,
                "evidence_support": "not_judged",
                "evidence_ids": [],
                "error_type": "none",
                "confidence": "deterministic",
                "basis": "Exact accepted-alias match.",
                "origin": "deterministic_exact",
            })
        elif not valid_single:
            preclassified.append({
                **base,
                "exact_match": False,
                "class": "C1",
                "raw_class": "invalid_format",
                "semantic_correct": False,
                "related": False,
                "evidence_support": "insufficient",
                "evidence_ids": [],
                "error_type": "invalid_single_answer_format",
                "confidence": "deterministic",
                "basis": "Prediction did not parse as exactly one factoid answer.",
                "origin": "deterministic_invalid_format",
            })
        else:
            records.append(base)

    output_root = _semantic_judge_output_root(cfg, stage_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    judged: list[dict[str, Any]] = []
    judge_summary: dict[str, Any] = {"status": "not_needed", "new_api_calls": 0}
    if records:
        judge = CandidateBankClassJudge(
            records=records,
            output_root=output_root,
            api_key_file=Path(cfg.semantic_judge_api_key_file),
            judge_model=cfg.semantic_judge_model,
            judge_endpoint=cfg.semantic_judge_endpoint,
            max_new_judge_calls=cfg.semantic_judge_max_new_calls,
            max_retries=cfg.semantic_judge_retries,
            timeout_seconds=cfg.semantic_judge_timeout_seconds,
        )
        judged, judge_summary = judge.run()

    all_judgments = preclassified + judged
    all_judgments.sort(key=lambda row: row["question_id"])
    step_dir = stage_dir / "semantic_generated_eval" / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "semantic_judgments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_judgments),
        encoding="utf-8",
    )
    class_counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    for row in all_judgments:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
        origin_counts[row["origin"]] = origin_counts.get(row["origin"], 0) + 1
    correct = sum(1 for row in all_judgments if row["class"] in {"C3", "C2"})
    evaluated = len(all_judgments)
    supported_correct = sum(
        1 for row in all_judgments
        if row["class"] in {"C3", "C2"} and row.get("evidence_support") in {"supported", "not_judged"}
    )
    summary = {
        "semantic_accuracy": (correct / evaluated) if evaluated else None,
        "semantic_correct_count": correct,
        "semantic_evaluated_count": evaluated,
        "semantic_supported_accuracy": (supported_correct / evaluated) if evaluated else None,
        "semantic_supported_correct_count": supported_correct,
        "semantic_class_counts": class_counts,
        "semantic_origin_counts": origin_counts,
        "semantic_judge_status": judge_summary.get("status"),
        "semantic_judge_new_api_calls": judge_summary.get("new_api_calls", 0),
        "semantic_judge_model": cfg.semantic_judge_model,
        "semantic_judgments_file": str((step_dir / "semantic_judgments.jsonl").resolve()),
    }
    _save_json(step_dir / "summary.json", summary)
    return summary

def _load_policy(cfg: Config, adapter: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trainer")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, device_map="auto",
        torch_dtype=compute_dtype, attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code)
    base = prepare_model_for_kbit_training(base)
    policy = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    _set_lora_dropout(policy, cfg.lora_dropout)
    policy.config.use_cache = False
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable()
    return policy


def _set_lora_dropout(model, dropout: float | None) -> None:
    if dropout is None:
        return
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"lora_dropout must be in [0, 1), got {dropout}")
    changed = 0
    for name, module in model.named_modules():
        if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
            module.p = dropout
            changed += 1
    for config in getattr(model, "peft_config", {}).values():
        if hasattr(config, "lora_dropout"):
            config.lora_dropout = dropout
    if changed == 0:
        raise RuntimeError("No LoRA dropout modules were found to update")


def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_config(cfg: Config) -> dict[str, Any]:
    config = asdict(cfg)
    # Before this option existed, generated evaluation used ``max_length``.
    # Omitting the redundant form preserves resumability for runs whose
    # effective evaluation length is unchanged.
    if config.get("generated_eval_max_seq_length") in {None, config.get("max_length")}:
        config.pop("generated_eval_max_seq_length", None)
    # Runtime controls for stopping/resuming the curriculum should not change the
    # dataset/model fingerprint. This lets a stage-1-only run continue later by
    # setting stop_after_stage=None with the same output_root. skip_stages is kept
    # in the fingerprint because it changes the trained model sequence.
    config.pop("stop_after_stage", None)
    # Folder naming is only an artifact-routing decision. Excluding it lets a
    # retry stage live beside the original stage while still reusing earlier
    # completed stages from the same parent output_root.
    config.pop("stage_output_suffixes", None)
    config.pop("allow_mixed_objective_resume", None)
    return config


def _safe_stage_suffix(value: str) -> str:
    value = str(value or "").strip().strip("_")
    if not value:
        return ""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        raise ValueError("stage output suffix became empty after sanitization")
    return value


def _stage_output_dir(output_root: Path, stage_index: int, stage_name: str, cfg: Config) -> Path:
    base = f"stage_{stage_index}_{stage_name}"
    suffix = _safe_stage_suffix(cfg.stage_output_suffixes.get(stage_name, ""))
    return output_root / (f"{base}_{suffix}" if suffix else base)


def _experiment_fingerprint(cfg: Config, stages: dict[str, list[dict[str, Any]]], global_eval_qids: set[str]) -> str:
    return hashlib.sha256(json.dumps({
        "config": _fingerprint_config(cfg),
        "stage_row_digests": {name: _rows_digest(rows) for name, rows in sorted(stages.items())},
        "global_eval_qids": sorted(global_eval_qids),
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _save_training_plot(output_root: Path, results: list[dict[str, Any]]):
    """Write a compact cross-stage plot when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, axes = plt.subplots(1, 5, figsize=(25, 4), constrained_layout=True)
    for result in results:
        stage = result.get("stage", "stage")
        histories = result.get("history", [])
        evaluations = result.get("eval_history", [])
        label = stage.capitalize()
        train = [(x.get("step", 0), x.get("train_loss")) for x in histories
                 if x.get("train_loss") is not None and not x.get("epoch_summary")]
        ev_loss = [(x.get("step", 0), x.get("dpo_eval_loss")) for x in evaluations
                   if x.get("dpo_eval_loss") is not None]
        mrr = [(x.get("step", 0), x.get("dev_mrr")) for x in evaluations
               if x.get("dev_mrr") is not None]
        semantic = [(x.get("step", 0), x.get("semantic_accuracy")) for x in evaluations
                    if x.get("semantic_accuracy") is not None]
        retention = [(x.get("step", 0), x.get("retention_rate")) for x in evaluations
                     if x.get("retention_rate") is not None]
        if train:
            axes[0].plot([x[0] for x in train], [x[1] for x in train], marker=".", label=label)
        if ev_loss:
            axes[1].plot([x[0] for x in ev_loss], [x[1] for x in ev_loss], marker="o", label=label)
        if mrr:
            axes[2].plot([x[0] for x in mrr], [x[1] for x in mrr], marker="o", label=label)
        if semantic:
            axes[3].plot([x[0] for x in semantic], [x[1] for x in semantic], marker="o", label=label)
        if retention:
            axes[4].plot([x[0] for x in retention], [x[1] for x in retention], marker="o", label=label)
    axes[0].set_title("Training DPO loss")
    axes[1].set_title("Held-out DPO eval loss")
    axes[2].set_title("Official BioASQ dev MRR")
    axes[3].set_title("LLM-judged semantic accuracy")
    axes[4].set_title("Baseline-correct retention")
    for axis in axes:
        axis.set_xlabel("Stage optimizer step")
        axis.grid(alpha=0.25)
        axis.legend()
    path = output_root / "training_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_adapter_checkpoint(model, tokenizer, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
    return path


def _resolve_stage_config(cfg: Config, stage: str) -> Config:
    """Return the effective Config for one stage after applying overrides."""
    return replace(
        cfg,
        beta=float(_setting(cfg, stage, "beta", cfg.beta)),
        objective=str(_setting(cfg, stage, "objective", cfg.objective)),
        dpo_d_nu=float(_setting(cfg, stage, "dpo_d_nu", cfg.dpo_d_nu)),
        tie_label_field=str(_setting(cfg, stage, "tie_label_field", cfg.tie_label_field)),
        adaptive_nll_weight=float(_setting(cfg, stage, "adaptive_nll_weight", cfg.adaptive_nll_weight)),
        auxiliary_diagnostic_weight=float(
            _setting(
                cfg,
                stage,
                "auxiliary_diagnostic_weight",
                cfg.auxiliary_diagnostic_weight,
            )
        ),
        learning_rate=float(_setting(cfg, stage, "learning_rate", cfg.stage3_learning_rate if stage in {"exactness", "hierarchical_ranking"} else cfg.learning_rate)),
        epochs=int(_setting(cfg, stage, "epochs", cfg.epochs)),
        batch_size=int(_setting(cfg, stage, "batch_size", cfg.batch_size)),
        gradient_accumulation=int(_setting(cfg, stage, "gradient_accumulation", cfg.gradient_accumulation)),
        max_grad_norm=float(_setting(cfg, stage, "max_grad_norm", cfg.max_grad_norm)),
        max_length=int(_setting(cfg, stage, "max_length", cfg.max_length)),
        generated_eval_max_seq_length=(
            None
            if _setting(
                cfg,
                stage,
                "generated_eval_max_seq_length",
                cfg.generated_eval_max_seq_length,
            ) is None
            else int(
                _setting(
                    cfg,
                    stage,
                    "generated_eval_max_seq_length",
                    cfg.generated_eval_max_seq_length,
                )
            )
        ),
        generated_eval_max_new_tokens=int(
            _setting(
                cfg,
                stage,
                "generated_eval_max_new_tokens",
                cfg.generated_eval_max_new_tokens,
            )
        ),
        train_backprop_max_length=(
            None if _setting(cfg, stage, "train_backprop_max_length", cfg.train_backprop_max_length) is None
            else int(_setting(cfg, stage, "train_backprop_max_length", cfg.train_backprop_max_length))
        ),
        warmup_fraction=float(_setting(cfg, stage, "warmup_fraction", cfg.warmup_fraction)),
        eval_every_updates=int(_setting(cfg, stage, "eval_every_updates", cfg.eval_every_updates)),
        early_stopping_patience=int(_setting(cfg, stage, "early_stopping_patience", cfg.early_stopping_patience)),
        early_stopping_min_delta=float(_setting(cfg, stage, "early_stopping_min_delta", cfg.early_stopping_min_delta)),
        selection_metric=str(_setting(cfg, stage, "selection_metric", cfg.selection_metric)),
        retention_loss_penalty=float(
            _setting(cfg, stage, "retention_loss_penalty", cfg.retention_loss_penalty)
        ),
        retention_min_rate=float(_setting(cfg, stage, "retention_min_rate", cfg.retention_min_rate)),
        retention_anchor_weight=float(
            _setting(cfg, stage, "retention_anchor_weight", cfg.retention_anchor_weight)
        ),
        retention_anchor_max_examples=(
            None
            if _setting(
                cfg, stage, "retention_anchor_max_examples", cfg.retention_anchor_max_examples
            ) is None
            else int(
                _setting(
                    cfg, stage, "retention_anchor_max_examples", cfg.retention_anchor_max_examples
                )
            )
        ),
        l2sp_weight=float(_setting(cfg, stage, "l2sp_weight", cfg.l2sp_weight)),
        allow_truncated_examples=bool(_setting(cfg, stage, "allow_truncated_examples", cfg.allow_truncated_examples)),
        drop_truncated_examples=bool(_setting(cfg, stage, "drop_truncated_examples", cfg.drop_truncated_examples)),
        append_eos_to_completions=bool(_setting(cfg, stage, "append_eos_to_completions", cfg.append_eos_to_completions)),
        semantic_judge_enabled=bool(_setting(cfg, stage, "semantic_judge_enabled", cfg.semantic_judge_enabled)),
        semantic_judge_model=str(_setting(cfg, stage, "semantic_judge_model", cfg.semantic_judge_model)),
        semantic_judge_endpoint=str(_setting(cfg, stage, "semantic_judge_endpoint", cfg.semantic_judge_endpoint)),
        semantic_judge_api_key_file=str(_setting(cfg, stage, "semantic_judge_api_key_file", cfg.semantic_judge_api_key_file)),
        semantic_judge_max_new_calls=_setting(cfg, stage, "semantic_judge_max_new_calls", cfg.semantic_judge_max_new_calls),
        semantic_judge_retries=int(_setting(cfg, stage, "semantic_judge_retries", cfg.semantic_judge_retries)),
        semantic_judge_timeout_seconds=int(_setting(cfg, stage, "semantic_judge_timeout_seconds", cfg.semantic_judge_timeout_seconds)),
        semantic_judge_cache_root=_setting(cfg, stage, "semantic_judge_cache_root", cfg.semantic_judge_cache_root),
        optimizer=str(_setting(cfg, stage, "optimizer", cfg.optimizer)),
        weight_decay=float(_setting(cfg, stage, "weight_decay", cfg.weight_decay)),
        lora_dropout=(
            None if _setting(cfg, stage, "lora_dropout", cfg.lora_dropout) is None
            else float(_setting(cfg, stage, "lora_dropout", cfg.lora_dropout))
        ),
    )


def _validate_stage_config(name: str, cfg: Config, stage_cfg: Config) -> None:
    invalid = (
        stage_cfg.batch_size <= 0
        or stage_cfg.gradient_accumulation <= 0
        or stage_cfg.max_grad_norm <= 0
        or stage_cfg.epochs <= 0
        or stage_cfg.early_stopping_patience <= 0
        or (
            stage_cfg.generated_eval_max_seq_length is not None
            and stage_cfg.generated_eval_max_seq_length <= 0
        )
        or stage_cfg.generated_eval_max_new_tokens <= 0
        or stage_cfg.objective not in SUPPORTED_OBJECTIVES
        or stage_cfg.dpo_d_nu <= 0
        or stage_cfg.adaptive_nll_weight < 0
        or stage_cfg.auxiliary_diagnostic_weight < 0
        or stage_cfg.retention_loss_penalty < 0
        or not 0 <= stage_cfg.retention_min_rate <= 1
        or stage_cfg.retention_anchor_weight < 0
        or (
            stage_cfg.retention_anchor_max_examples is not None
            and stage_cfg.retention_anchor_max_examples <= 0
        )
        or stage_cfg.l2sp_weight < 0
        or not 0 <= stage_cfg.warmup_fraction <= 1
        or stage_cfg.selection_metric not in SUPPORTED_SELECTION_METRICS
    )
    if invalid:
        raise ValueError(f"Invalid settings for stage {name}: {asdict(stage_cfg)}")
    if stage_cfg.selection_metric in {"semantic_accuracy", "semantic_supported_accuracy"}:
        if not cfg.evaluate_generated_dev:
            raise ValueError(f"{name}: {stage_cfg.selection_metric} requires evaluate_generated_dev=True")
        if not stage_cfg.semantic_judge_enabled:
            raise ValueError(f"{name}: {stage_cfg.selection_metric} requires semantic_judge_enabled=True")
        if not Path(stage_cfg.semantic_judge_api_key_file).exists():
            raise FileNotFoundError(stage_cfg.semantic_judge_api_key_file)
    if stage_cfg.selection_metric == "retention_score" and not cfg.evaluate_generated_dev:
        raise ValueError(f"{name}: retention_score requires evaluate_generated_dev=True")


def _prepare_stage_dir(stage_dir: Path, name: str) -> None:
    """Create a clean stage directory, preserving incomplete prior attempts."""
    if stage_dir.exists():
        if (stage_dir / "manifest.json").exists():
            raise FileExistsError(f"Completed stage directory already exists: {stage_dir}")
        interrupted = stage_dir.with_name(f"{stage_dir.name}_interrupted_{int(time.time())}")
        stage_dir.rename(interrupted)
        print(f"[{name}] preserved incomplete directory at {interrupted}; restarting stage cleanly")
    stage_dir.mkdir(parents=True, exist_ok=False)


def _encode_stage_pairs(tokenizer, rows: list[dict[str, Any]], stage_cfg: Config, stage_dir: Path, name: str):
    encoded_all = []
    dropped_truncated: list[dict[str, Any]] = []
    for row in rows:
        try:
            encoded_all.append(
                _encode_pair(
                    tokenizer,
                    row,
                    stage_cfg.max_length,
                    append_eos=stage_cfg.append_eos_to_completions,
                    allow_truncation=stage_cfg.allow_truncated_examples,
                )
            )
        except ValueError as exc:
            if stage_cfg.drop_truncated_examples and "exceeds max_length" in str(exc):
                dropped_truncated.append({
                    "pair_id": row["pair_id"],
                    "question_id": row["question_id"],
                    "stage": name,
                    "reason": str(exc),
                })
                continue
            raise
    if not encoded_all:
        raise ValueError(f"{name}: all pairs were dropped by token-length filtering")
    if dropped_truncated:
        _save_json(stage_dir / "dropped_truncated_pairs.json", dropped_truncated)
    return encoded_all, dropped_truncated


def _split_encoded_by_train_ids(encoded_all: list[dict[str, Any]], train_rows: list[dict[str, Any]]):
    train_ids = {row["pair_id"] for row in train_rows}
    encoded = [item for item in encoded_all if item["row"]["pair_id"] in train_ids]
    encoded_eval = [item for item in encoded_all if item["row"]["pair_id"] not in train_ids]
    return encoded, encoded_eval


def _apply_backprop_budget(encoded: list[dict[str, Any]], stage_cfg: Config, stage_dir: Path, name: str):
    if stage_cfg.train_backprop_max_length is None:
        return encoded, []
    kept_encoded = []
    dropped_backprop: list[dict[str, Any]] = []
    budget = int(stage_cfg.train_backprop_max_length)
    for item in encoded:
        chosen_len = len(item["chosen_ids"])
        rejected_len = len(item["rejected_ids"])
        auxiliary_len = len(item.get("auxiliary_ids", []))
        train_len = max(chosen_len, rejected_len, auxiliary_len)
        if train_len <= budget:
            kept_encoded.append(item)
            continue
        row = item["row"]
        dropped_backprop.append({
            "pair_id": row["pair_id"],
            "question_id": row["question_id"],
            "stage": name,
            "prompt_tokens": item["prompt_tokens"],
            "chosen_tokens": item["chosen_tokens"],
            "rejected_tokens": item["rejected_tokens"],
            "chosen_sequence_tokens": chosen_len,
            "rejected_sequence_tokens": rejected_len,
            "auxiliary_sequence_tokens": auxiliary_len or None,
            "train_sequence_tokens": train_len,
            "train_backprop_max_length": budget,
            "reason": f"max(chosen_sequence_tokens, rejected_sequence_tokens, auxiliary_sequence_tokens)={train_len} exceeds train_backprop_max_length={budget}",
        })
    if not kept_encoded:
        raise ValueError(f"{name}: all training pairs were dropped by train_backprop_max_length={stage_cfg.train_backprop_max_length}")
    if dropped_backprop:
        _save_json(stage_dir / "dropped_backprop_pairs.json", dropped_backprop)
    return kept_encoded, dropped_backprop


def _prepare_retention_anchors(
    tokenizer,
    rows: list[dict[str, Any]],
    stage_cfg: Config,
    global_eval_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build unique Stage-1 chosen completions used to preserve the incoming policy.

    Dev/held-out questions are excluded so the anchor cannot train on model-selection
    examples.  The rejected completion is encoded only to reuse the normal collator;
    the retention loss scores the chosen completion alone.
    """
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row["question_id"] in global_eval_qids:
            continue
        key = (row["question_id"], row["chosen"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    rng = random.Random(f"{stage_cfg.seed}:retention-anchors")
    rng.shuffle(unique_rows)
    if stage_cfg.retention_anchor_max_examples is not None:
        unique_rows = unique_rows[: stage_cfg.retention_anchor_max_examples]

    encoded, dropped = [], []
    budget = stage_cfg.train_backprop_max_length
    for row in unique_rows:
        try:
            item = _encode_pair(
                tokenizer,
                row,
                stage_cfg.max_length,
                append_eos=stage_cfg.append_eos_to_completions,
                allow_truncation=stage_cfg.allow_truncated_examples,
            )
        except ValueError as exc:
            if stage_cfg.drop_truncated_examples and "exceeds max_length" in str(exc):
                dropped.append({
                    "pair_id": row["pair_id"],
                    "question_id": row["question_id"],
                    "reason": str(exc),
                })
                continue
            raise
        if budget is not None and len(item["chosen_ids"]) > int(budget):
            dropped.append({
                "pair_id": row["pair_id"],
                "question_id": row["question_id"],
                "chosen_sequence_tokens": len(item["chosen_ids"]),
                "train_backprop_max_length": int(budget),
                "reason": "chosen retention sequence exceeds train_backprop_max_length",
            })
            continue
        encoded.append(item)
    return encoded, dropped


def _save_stage_data_summary(
    stage_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    encoded_all: list[dict[str, Any]],
    encoded_train: list[dict[str, Any]],
    dropped_truncated: list[dict[str, Any]],
    dropped_backprop: list[dict[str, Any]],
    global_eval_qids: set[str],
    stage_cfg: Config,
) -> None:
    optimizer_name = _normalize_optimizer_name(stage_cfg.optimizer)
    _save_json(stage_dir / "data_summary.json", {
        "stage": name, "pairs": len(rows), "questions": len({r['question_id'] for r in rows}),
        "train_pairs": len(train_rows), "train_questions": len({r['question_id'] for r in train_rows}),
        "eval_pairs": len(eval_rows), "eval_questions": len({r['question_id'] for r in eval_rows}),
        "encoded_pairs": len(encoded_all),
        "encoded_train_pairs_after_backprop_filter": len(encoded_train),
        "dropped_truncated_pairs": len(dropped_truncated),
        "dropped_backprop_pairs": len(dropped_backprop),
        "train_backprop_max_length": stage_cfg.train_backprop_max_length,
        "global_eval_question_count": len(global_eval_qids),
        "optimizer": optimizer_name,
        "objective": stage_cfg.objective,
        "dpo_d_nu": stage_cfg.dpo_d_nu,
        "tie_label_field": stage_cfg.tie_label_field,
        "clear_win_pairs": sum(not _row_is_tie(row, stage_cfg) for row in rows),
        "tie_pairs": sum(_row_is_tie(row, stage_cfg) for row in rows),
        "adaptive_nll_weight": stage_cfg.adaptive_nll_weight,
        "auxiliary_diagnostic_weight": stage_cfg.auxiliary_diagnostic_weight,
        "auxiliary_pair_count": sum("auxiliary_ids" in item for item in encoded_all),
        "beta": stage_cfg.beta,
        "learning_rate": stage_cfg.learning_rate,
        "weight_decay": stage_cfg.weight_decay,
        "retention_loss_penalty": stage_cfg.retention_loss_penalty,
        "retention_min_rate": stage_cfg.retention_min_rate,
        "retention_anchor_weight": stage_cfg.retention_anchor_weight,
        "retention_anchor_max_examples": stage_cfg.retention_anchor_max_examples,
        "l2sp_weight": stage_cfg.l2sp_weight,
        "warmup_fraction": stage_cfg.warmup_fraction,
        "max_length": stage_cfg.max_length,
        "generated_eval_max_seq_length": (
            stage_cfg.generated_eval_max_seq_length or stage_cfg.max_length
        ),
        "generated_eval_max_new_tokens": stage_cfg.generated_eval_max_new_tokens,
        "append_eos_to_completions": stage_cfg.append_eos_to_completions,
        "allow_truncated_examples": stage_cfg.allow_truncated_examples,
        "drop_truncated_examples": stage_cfg.drop_truncated_examples,
        "prompt_tokens_max": max(x["prompt_tokens"] for x in encoded_all),
        "auxiliary_prompt_tokens_max": max(
            (x.get("auxiliary_prompt_tokens", 0) for x in encoded_all),
            default=0,
        ),
        "auxiliary_target_tokens_max": max(
            (x.get("auxiliary_target_tokens", 0) for x in encoded_all),
            default=0,
        ),
        "train_sequence_tokens_max_before_backprop_filter": max(
            max(
                len(x["chosen_ids"]),
                len(x["rejected_ids"]),
                len(x.get("auxiliary_ids", [])),
            )
            for x in encoded_all
        ),
        "train_sequence_tokens_max_after_backprop_filter": max(
            max(
                len(x["chosen_ids"]),
                len(x["rejected_ids"]),
                len(x.get("auxiliary_ids", [])),
            )
            for x in encoded_train
        ),
        "truncated_pairs": sum(
            max(
                x["prompt_tokens"] + max(x["chosen_tokens"], x["rejected_tokens"]),
                x.get("auxiliary_prompt_tokens", 0) + x.get("auxiliary_target_tokens", 0),
            ) > stage_cfg.max_length
            for x in encoded_all
        ),
    })


def _metric_improved(metric: str, current: float, best: float | None, min_delta: float) -> bool:
    if metric in {"dpo_eval_loss", "objective_eval_loss"}:
        return best is None or float(current) < float(best) - min_delta
    return best is None or float(current) > float(best) + min_delta


def _retention_metrics(
    baseline_metrics: list[dict[str, Any]],
    current_metrics: list[dict[str, Any]],
    loss_penalty: float,
    min_rate: float,
) -> dict[str, Any]:
    """Compare exact BioASQ correctness with the incoming stage checkpoint."""
    baseline = {
        row["question_id"]: float(row.get("mrr", row.get("primary_score", 0.0)) or 0.0) > 0
        for row in baseline_metrics
    }
    current = {
        row["question_id"]: float(row.get("mrr", row.get("primary_score", 0.0)) or 0.0) > 0
        for row in current_metrics
    }
    if set(baseline) != set(current):
        missing = sorted(set(baseline) - set(current))
        extra = sorted(set(current) - set(baseline))
        raise ValueError(
            "Generated-dev question IDs changed during a stage; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    baseline_correct = {qid for qid, correct in baseline.items() if correct}
    baseline_wrong = set(baseline) - baseline_correct
    retained = sum(current[qid] for qid in baseline_correct)
    lost = len(baseline_correct) - retained
    gained = sum(current[qid] for qid in baseline_wrong)
    rate = retained / len(baseline_correct) if baseline_correct else 1.0
    score = float(gained) - float(loss_penalty) * float(lost)
    return {
        "baseline_correct_count": len(baseline_correct),
        "baseline_wrong_count": len(baseline_wrong),
        "retained_correct_count": retained,
        "lost_correct_count": lost,
        "newly_correct_count": gained,
        "retention_rate": rate,
        "retention_score": score,
        "retention_constraint_met": rate >= min_rate,
        "retention_min_rate": min_rate,
        "retention_loss_penalty": loss_penalty,
    }


def _train_stage(
    name: str,
    rows: list[dict[str, Any]],
    cfg: Config,
    adapter: Path,
    stage_dir: Path,
    global_eval_qids: set[str],
    experiment_fingerprint: str,
    retention_rows: list[dict[str, Any]] | None = None,
):
    _prepare_stage_dir(stage_dir, name)
    stage_cfg = _resolve_stage_config(cfg, name)
    _validate_stage_config(name, cfg, stage_cfg)
    if stage_cfg.auxiliary_diagnostic_weight > 0:
        missing_auxiliary = [
            row.get("pair_id")
            for row in rows
            if not str(row.get("auxiliary_prompt") or "").strip()
            or not str(row.get("auxiliary_target") or "").strip()
        ]
        if missing_auxiliary:
            raise ValueError(
                f"{name}: auxiliary_diagnostic_weight={stage_cfg.auxiliary_diagnostic_weight} "
                f"but {len(missing_auxiliary)} rows lack auxiliary_prompt/auxiliary_target; "
                f"first IDs: {missing_auxiliary[:5]}"
            )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=stage_cfg.trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, eval_rows = _split_stage_rows(rows, cfg, name, global_eval_qids)
    encoded_all, dropped_truncated = _encode_stage_pairs(tokenizer, rows, stage_cfg, stage_dir, name)
    encoded, encoded_eval = _split_encoded_by_train_ids(encoded_all, train_rows)
    encoded, dropped_backprop = _apply_backprop_budget(encoded, stage_cfg, stage_dir, name)
    _save_stage_data_summary(
        stage_dir, name, rows, train_rows, eval_rows, encoded_all, encoded,
        dropped_truncated, dropped_backprop, global_eval_qids, stage_cfg
    )
    model = _load_policy(stage_cfg, adapter)
    reference = _reference_scores(model, encoded, tokenizer, stage_cfg)
    reference_eval = _reference_scores(model, encoded_eval, tokenizer, stage_cfg) if encoded_eval else []
    retention_encoded: list[dict[str, Any]] = []
    retention_reference: list[float] = []
    retention_dropped: list[dict[str, Any]] = []
    if stage_cfg.retention_anchor_weight > 0:
        retention_encoded, retention_dropped = _prepare_retention_anchors(
            tokenizer, retention_rows or [], stage_cfg, global_eval_qids
        )
        if not retention_encoded:
            raise ValueError(
                f"{name}: retention_anchor_weight={stage_cfg.retention_anchor_weight} "
                "but no eligible retention rows were available"
            )
        retention_reference = _reference_chosen_scores(
            model, retention_encoded, tokenizer, stage_cfg
        )
        _save_json(stage_dir / "retention_anchor_summary.json", {
            "source": "unique chosen completions from concept_learning training questions",
            "anchor_examples": len(retention_encoded),
            "dropped_examples": len(retention_dropped),
            "weight": stage_cfg.retention_anchor_weight,
            "mode": "mean_completion_logp_mse_to_incoming_stage_policy",
            "excluded_global_eval_questions": True,
        })
        if retention_dropped:
            _save_json(stage_dir / "dropped_retention_anchors.json", retention_dropped)
    loader = DataLoader(list(zip(encoded, reference)), batch_size=stage_cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(stage_cfg.seed),
                        collate_fn=lambda b: _collate([x[0] for x in b], tokenizer.pad_token_id))
    trainable = [p for p in model.parameters() if p.requires_grad]
    l2sp_reference = (
        [parameter.detach().clone() for parameter in trainable]
        if stage_cfg.l2sp_weight > 0
        else []
    )
    lr = stage_cfg.learning_rate
    epochs = stage_cfg.epochs
    eval_every_updates = stage_cfg.eval_every_updates
    optimizer = _make_optimizer(trainable, stage_cfg, name)
    total_updates = max(1, math.ceil(len(loader) / stage_cfg.gradient_accumulation) * epochs)
    warmup = int(total_updates * stage_cfg.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup)) if warmup else None
    history = []
    eval_history = []
    eval_metrics = []
    baseline_dev_metrics: list[dict[str, Any]] = []
    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dev_rows, eval_prompt_id = ([], None)
    if cfg.evaluate_generated_dev:
        dev_rows, eval_prompt_id = _build_dev_eval_rows(cfg, tokenizer)
    ref_by_id = {item["row"]["pair_id"]: reference[i] for i, item in enumerate(encoded)}

    def evaluate_now(epoch, phase):
        nonlocal eval_metrics, baseline_dev_metrics
        summary = _dpo_eval_loss(model, encoded_eval, reference_eval, tokenizer, stage_cfg)
        summary.update({"stage": name, "epoch": epoch, "step": step, "evaluation_phase": phase,
                        "eval_prompt_id": eval_prompt_id, "dev_eval_question_count": len(dev_rows)})
        if cfg.evaluate_generated_dev and dev_rows:
            from cse_dpo.generated_bioasq_eval import evaluate_generated_bioasq
            generated_summary, eval_metrics = evaluate_generated_bioasq(
                model, tokenizer, dev_rows,
                max_seq_length=(stage_cfg.generated_eval_max_seq_length or stage_cfg.max_length),
                max_new_tokens=stage_cfg.generated_eval_max_new_tokens, do_sample=False,
                empty_cuda_cache_per_generation=True, progress_desc=f"{name} {phase} BioASQ dev",
                official_output_dir=stage_dir / "official_generated_eval" / f"step_{step}",
                official_model_label=f"three-stage-dpo-{name}-step-{step}",
            )
            summary.update({"dev_mrr": generated_summary.get("mrr"),
                            "dev_strict_accuracy": generated_summary.get("strict_accuracy"),
                            "dev_lenient_accuracy": generated_summary.get("lenient_accuracy"),
                            "scoring_backend": generated_summary.get("scoring_backend"),
                            "generation_protocol": {
                                "max_seq_length": (
                                    stage_cfg.generated_eval_max_seq_length or stage_cfg.max_length
                                ),
                                "max_new_tokens": stage_cfg.generated_eval_max_new_tokens,
                                "num_generations": 1,
                                "do_sample": False,
                                "use_cache": False,
                            }})
            if not baseline_dev_metrics:
                baseline_dev_metrics = [dict(row) for row in eval_metrics]
            summary.update(
                _retention_metrics(
                    baseline_dev_metrics,
                    eval_metrics,
                    stage_cfg.retention_loss_penalty,
                    stage_cfg.retention_min_rate,
                )
            )
            if stage_cfg.semantic_judge_enabled:
                summary.update(_score_generated_semantics(stage_cfg, stage_dir, dev_rows, eval_metrics, step=step))
        eval_history.append(summary)
        _save_json(stage_dir / "eval_history.json", eval_history)
        _save_json(stage_dir / f"evaluation_metrics_step_{step}.json", eval_metrics)
        model.train()
        return summary

    checkpoints_root = stage_dir / "checkpoints"
    best_mrr = None
    best_loss = None
    best_mrr_adapter = None
    best_loss_adapter = None
    best_selected = None
    best_selected_adapter = None
    bad_evaluations = 0
    evaluation_count = 0
    stopped_early = False

    def consider_checkpoint(summary):
        """Save the selected metric optimum; return whether patience is exhausted."""
        nonlocal best_mrr, best_loss, best_mrr_adapter, best_loss_adapter
        nonlocal best_selected, best_selected_adapter
        nonlocal bad_evaluations, evaluation_count
        evaluation_count += 1
        current_mrr = summary.get("dev_mrr")
        current_loss = summary.get("dpo_eval_loss")
        current_selected = summary.get(stage_cfg.selection_metric)
        if not isinstance(current_selected, (int, float)):
            raise ValueError(
                f"{name}: selection_metric={stage_cfg.selection_metric!r} is unavailable in "
                f"evaluation summary. Enable the required evaluation or choose another metric."
            )
        delta = stage_cfg.early_stopping_min_delta
        improved_mrr = isinstance(current_mrr, (int, float)) and _metric_improved("dev_mrr", float(current_mrr), best_mrr, delta)
        improved_loss = isinstance(current_loss, (int, float)) and _metric_improved("dpo_eval_loss", float(current_loss), best_loss, delta)
        retention_eligible = (
            stage_cfg.selection_metric != "retention_score"
            or bool(summary.get("retention_constraint_met"))
        )
        improved_selected = retention_eligible and _metric_improved(
            stage_cfg.selection_metric, float(current_selected), best_selected, delta
        )
        if improved_selected or improved_mrr or improved_loss:
            checkpoint_path = checkpoints_root / f"step_{step}"
            _save_adapter_checkpoint(model, tokenizer, checkpoint_path)
            summary["checkpoint"] = str(checkpoint_path.resolve())
        else:
            checkpoint_path = None
        if improved_selected:
            best_selected = float(current_selected)
            best_selected_adapter = checkpoint_path
        if improved_mrr:
            best_mrr = float(current_mrr)
            best_mrr_adapter = checkpoint_path
        if improved_loss:
            best_loss = float(current_loss)
            best_loss_adapter = checkpoint_path
        if improved_selected:
            bad_evaluations = 0
        else:
            bad_evaluations += 1
        summary.update({"improved_dev_mrr": improved_mrr, "improved_dpo_eval_loss": improved_loss,
                        "selection_metric": stage_cfg.selection_metric,
                        "selected_metric_value": float(current_selected),
                        "improved_selected_metric": improved_selected,
                        "selected_metric_eligible": retention_eligible,
                        "bad_evaluations": bad_evaluations, "evaluation_count": evaluation_count})
        _save_json(stage_dir / "eval_history.json", eval_history)
        return bad_evaluations >= stage_cfg.early_stopping_patience

    # Record and checkpoint the starting point before any update.
    initial_summary = evaluate_now(0, "initial")
    consider_checkpoint(initial_summary)
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_updates = 0
        update_loss_total = 0.0
        update_preference_loss_total = 0.0
        update_auxiliary_loss_total = 0.0
        update_retention_loss_total = 0.0
        update_l2sp_loss_total = 0.0
        update_micro_count = 0
        for batch_idx, batch in enumerate(loader):
            group_start = (batch_idx // stage_cfg.gradient_accumulation) * stage_cfg.gradient_accumulation
            group_size = min(stage_cfg.gradient_accumulation, len(loader) - group_start)
            policy_c = _sequence_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"], True)
            policy_r = _sequence_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"], True)
            refs = torch.tensor([ref_by_id[r["pair_id"]][0:2] for r in batch["rows"]], device=policy_c.device, dtype=policy_c.dtype)
            ties = _tie_targets(batch["rows"], stage_cfg, device=policy_c.device, dtype=policy_c.dtype)
            preference_loss = _preference_losses(policy_c, policy_r, refs, stage_cfg, ties).mean()
            (preference_loss / group_size).backward()

            auxiliary_loss = torch.zeros((), device=policy_c.device, dtype=policy_c.dtype)
            if stage_cfg.auxiliary_diagnostic_weight > 0:
                if "auxiliary_input_ids" not in batch:
                    raise ValueError(
                        "Auxiliary diagnostic loss is enabled but the training batch has no target"
                    )
                auxiliary_logps = _sequence_logps(
                    model,
                    batch["auxiliary_input_ids"],
                    batch["auxiliary_attention_mask"],
                    batch["auxiliary_labels"],
                    True,
                )
                auxiliary_counts = (batch["auxiliary_labels"] != -100).sum(dim=1).to(
                    device=auxiliary_logps.device,
                    dtype=auxiliary_logps.dtype,
                ).clamp_min(1)
                # Token-mean CE prevents longer diagnostic explanations from
                # receiving disproportionately large weight.
                auxiliary_loss = (-auxiliary_logps / auxiliary_counts).mean()
                (
                    stage_cfg.auxiliary_diagnostic_weight * auxiliary_loss / group_size
                ).backward()

            retention_loss = torch.zeros((), device=policy_c.device, dtype=policy_c.dtype)
            if retention_encoded:
                anchor_idx = batch_idx % len(retention_encoded)
                anchor_item = retention_encoded[anchor_idx]
                anchor_batch = _collate([anchor_item], tokenizer.pad_token_id)
                policy_anchor = _sequence_logps(
                    model,
                    anchor_batch["chosen_input_ids"],
                    anchor_batch["chosen_attention_mask"],
                    anchor_batch["chosen_labels"],
                    True,
                )
                completion_tokens = max(1, anchor_item["chosen_tokens"])
                reference_anchor = torch.tensor(
                    [retention_reference[anchor_idx]],
                    device=policy_anchor.device,
                    dtype=policy_anchor.dtype,
                )
                retention_loss = (
                    policy_anchor / completion_tokens - reference_anchor / completion_tokens
                ).pow(2).mean()
                (
                    stage_cfg.retention_anchor_weight * retention_loss / group_size
                ).backward()

            l2sp_loss = torch.zeros((), device=policy_c.device, dtype=policy_c.dtype)
            if l2sp_reference:
                l2sp_loss = torch.stack([
                    (parameter - initial).float().pow(2).mean()
                    for parameter, initial in zip(trainable, l2sp_reference)
                ]).mean()
                (stage_cfg.l2sp_weight * l2sp_loss / group_size).backward()

            loss = (
                preference_loss.detach()
                + stage_cfg.auxiliary_diagnostic_weight * auxiliary_loss.detach()
                + stage_cfg.retention_anchor_weight * retention_loss.detach()
                + stage_cfg.l2sp_weight * l2sp_loss.detach()
            )
            loss_value = float(loss.cpu())
            preference_loss_value = float(preference_loss.detach().cpu())
            auxiliary_loss_value = float(auxiliary_loss.detach().cpu())
            retention_loss_value = float(retention_loss.detach().cpu())
            l2sp_loss_value = float(l2sp_loss.detach().cpu())
            epoch_loss += loss_value
            update_loss_total += loss_value
            update_preference_loss_total += preference_loss_value
            update_auxiliary_loss_total += auxiliary_loss_value
            update_retention_loss_total += retention_loss_value
            update_l2sp_loss_total += l2sp_loss_value
            update_micro_count += 1
            if (batch_idx + 1) % stage_cfg.gradient_accumulation == 0 or batch_idx + 1 == len(loader):
                clip_grad_norm_(trainable, stage_cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                epoch_updates += 1
                history.append({"stage": name, "epoch": epoch + 1, "step": step,
                                "train_loss": update_loss_total / update_micro_count,
                                "preference_loss": update_preference_loss_total / update_micro_count,
                                "auxiliary_diagnostic_loss": update_auxiliary_loss_total / update_micro_count,
                                "auxiliary_diagnostic_weight": stage_cfg.auxiliary_diagnostic_weight,
                                "retention_anchor_loss": update_retention_loss_total / update_micro_count,
                                "l2sp_loss": update_l2sp_loss_total / update_micro_count,
                                "learning_rate": lr, "micro_batches": update_micro_count})
                update_loss_total = 0.0
                update_preference_loss_total = 0.0
                update_auxiliary_loss_total = 0.0
                update_retention_loss_total = 0.0
                update_l2sp_loss_total = 0.0
                update_micro_count = 0
                if eval_every_updates > 0 and step % eval_every_updates == 0:
                    summary = evaluate_now(epoch + 1, "scheduled")
                    if consider_checkpoint(summary):
                        stopped_early = True
                        break
        if stopped_early:
            history.append({"stage": name, "epoch": epoch + 1, "step": step,
                            "train_loss": epoch_loss / max(1, batch_idx + 1),
                            "learning_rate": lr, "epoch_summary": True,
                            "optimizer_updates": epoch_updates, "stopped_early": True})
            _save_json(stage_dir / "history.json", history)
            break
        if eval_every_updates <= 0 or not eval_history or eval_history[-1]["step"] != step:
            summary = evaluate_now(epoch + 1, "epoch_end")
            if consider_checkpoint(summary):
                stopped_early = True
        history.append({"stage": name, "epoch": epoch + 1, "step": step,
                        "train_loss": epoch_loss / len(loader), "learning_rate": lr,
                        "epoch_summary": True, "optimizer_updates": epoch_updates,
                        "stopped_early": stopped_early})
        _save_json(stage_dir / "history.json", history)
        print(f"[{name}] epoch {epoch + 1}/{epochs}: train_loss={epoch_loss / len(loader):.6f}; eval={eval_history[-1]}")
        if stopped_early:
            break
    adapter_out = stage_dir / "adapter"
    _save_adapter_checkpoint(model, tokenizer, adapter_out)
    if best_selected_adapter is None:
        best_selected_adapter = adapter_out
    if best_mrr_adapter is None:
        best_mrr_adapter = best_selected_adapter
    if best_loss_adapter is None:
        best_loss_adapter = best_selected_adapter
    _save_training_plot(stage_dir, [{"stage": name, "history": history, "eval_history": eval_history}])
    _save_json(stage_dir / "manifest.json", {"stage": name, "adapter": str(adapter_out.resolve()),
        "best_selected_adapter": str(best_selected_adapter.resolve()),
        "objective": stage_cfg.objective,
        "selection_metric": stage_cfg.selection_metric,
        "best_selected_metric": best_selected,
        "best_mrr_adapter": str(best_mrr_adapter.resolve()), "best_loss_adapter": str(best_loss_adapter.resolve()),
        "best_dev_mrr": best_mrr, "best_dpo_eval_loss": best_loss,
        "stopped_early": stopped_early, "config": asdict(cfg), "stage_config": asdict(stage_cfg),
        "experiment_fingerprint": experiment_fingerprint,
        "rows": len(rows), "train_rows": len(train_rows), "eval_rows": len(eval_rows),
        "history": history, "eval_history": eval_history, "eval_prompt_id": eval_prompt_id})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_selected_adapter, history, eval_history


def _validate_run_plan(cfg: Config) -> tuple[list[str], set[str]]:
    stage_order = [name for name, _ in STAGES]
    if cfg.stop_after_stage is not None and cfg.stop_after_stage not in stage_order:
        raise ValueError(f"Unknown stop_after_stage {cfg.stop_after_stage!r}; expected one of {stage_order}")
    unknown_skips = sorted(set(cfg.skip_stages) - set(stage_order))
    if unknown_skips:
        raise ValueError(f"Unknown skip_stages {unknown_skips!r}; expected stage names from {stage_order}")
    unknown_suffixes = sorted(set(cfg.stage_output_suffixes) - set(stage_order))
    if unknown_suffixes:
        raise ValueError(f"Unknown stage_output_suffixes keys {unknown_suffixes!r}; expected stage names from {stage_order}")
    return stage_order, set(cfg.skip_stages)


def _has_stage_dirs(output_root: Path, cfg: Config) -> bool:
    return any(
        (output_root / f"stage_{idx}_{name}").exists()
        or _stage_output_dir(output_root, idx, name, cfg).exists()
        for idx, (name, _) in enumerate(STAGES, 1)
    )


def _prepare_output_root(cfg: Config, experiment_fingerprint: str, global_eval_qids: set[str]) -> Path:
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "config.json"
    has_stage_dirs = _has_stage_dirs(output_root, cfg)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        old_fingerprint = existing.get("experiment_fingerprint")
        if old_fingerprint is None and has_stage_dirs:
            raise RuntimeError(
                f"{output_root} contains stage directories from an older run without a fingerprint. "
                "Use a new output_root or remove the old run deliberately."
            )
        if old_fingerprint is not None and old_fingerprint != experiment_fingerprint:
            if cfg.allow_mixed_objective_resume and has_stage_dirs:
                print(
                    f"[resume] {output_root} has a different prior experiment fingerprint; "
                    "completed stages will be reused, and new stages will use the current config."
                )
            else:
                raise RuntimeError(
                    f"{output_root} already contains a different experiment configuration or dataset. "
                    "Use a new output_root, or set allow_mixed_objective_resume=True to deliberately continue from completed stages."
                )
    _save_json(output_root / "config.json", {
        "config": asdict(cfg),
        "experiment_fingerprint": experiment_fingerprint,
        "global_eval_question_count": len(global_eval_qids),
        "global_eval_question_ids": sorted(global_eval_qids),
    })
    return output_root


def _load_completed_stage_manifest(name: str, stage_dir: Path, cfg: Config, experiment_fingerprint: str) -> dict[str, Any] | None:
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    old_cfg = manifest.get("config", {})
    if "stage_config" not in manifest or "eval_history" not in manifest:
        raise RuntimeError(
            f"{stage_dir} is from the earlier trainer without evaluation history; "
            "use a new output_root for the metrics-enabled run."
        )
    if old_cfg.get("smoke_test") != cfg.smoke_test or old_cfg.get("smoke_pairs") != cfg.smoke_pairs:
        raise RuntimeError(
            f"{stage_dir} was created with a different smoke configuration; "
            "use a new output_root for the full run."
        )
    if manifest.get("experiment_fingerprint") != experiment_fingerprint:
        if cfg.allow_mixed_objective_resume:
            print(
                f"[{name}] completed stage has a different prior fingerprint; "
                "reusing its best adapter because allow_mixed_objective_resume=True"
            )
        else:
            raise RuntimeError(
                f"{stage_dir} was created from a different dataset/configuration. "
                "Use a new output_root, or set allow_mixed_objective_resume=True to deliberately continue from completed stages."
            )
    return manifest


def run_three_stage(cfg: Config | None = None):
    cfg = cfg or Config()
    stage_order, skip_stages = _validate_run_plan(cfg)
    set_seed(cfg.seed)
    random.seed(cfg.seed)

    stages = load_stage_rows(cfg)
    global_eval_qids = _global_eval_question_ids(stages, cfg)
    experiment_fingerprint = _experiment_fingerprint(cfg, stages, global_eval_qids)
    output_root = _prepare_output_root(cfg, experiment_fingerprint, global_eval_qids)

    adapter = Path(cfg.initial_adapter)
    if not adapter.exists():
        raise FileNotFoundError(adapter)

    results = []
    stopped_after_requested_stage = False
    for stage_index, (name, _) in enumerate(STAGES, 1):
        if name not in stages:
            continue

        stage_dir = _stage_output_dir(output_root, stage_index, name, cfg)
        if name in skip_stages:
            skipped = {
                "stage": name,
                "status": "skipped",
                "reason": "listed in cfg.skip_stages",
                "adapter": str(adapter.resolve()),
                "stage_dir": str(stage_dir),
                "rows": len(stages.get(name, [])),
            }
            results.append(skipped)
            print(f"[{name}] skipped; carrying forward adapter {adapter}")
            if cfg.stop_after_stage == name:
                stopped_after_requested_stage = True
                break
            continue

        manifest = _load_completed_stage_manifest(name, stage_dir, cfg, experiment_fingerprint)
        if manifest is not None:
            adapter = Path(manifest.get("best_selected_adapter", manifest.get("best_mrr_adapter", manifest["adapter"])))
            results.append(manifest)
            print(f"[{name}] already complete; reusing {adapter}")
            if cfg.stop_after_stage == name:
                stopped_after_requested_stage = True
                break
            continue

        adapter, history, eval_history = _train_stage(
            name,
            stages[name],
            cfg,
            adapter,
            stage_dir,
            global_eval_qids,
            experiment_fingerprint,
            retention_rows=(stages.get("concept_learning", []) if name != "concept_learning" else None),
        )
        results.append({"stage": name, "adapter": str(adapter.resolve()), "history": history, "eval_history": eval_history})
        # Persist a cumulative plot immediately. If a later stage fails, the
        # completed earlier stages remain visualized at the run root.
        _save_training_plot(output_root, results)
        if cfg.stop_after_stage == name:
            stopped_after_requested_stage = True
            break

    plot_path = _save_training_plot(output_root, results)
    _save_json(output_root / "manifest.json", {
        "status": "partial" if stopped_after_requested_stage and cfg.stop_after_stage != stage_order[-1] else "complete",
        "stop_after_stage": cfg.stop_after_stage,
        "stopped_after_requested_stage": stopped_after_requested_stage,
        "stages": results,
        "config": asdict(cfg),
        "experiment_fingerprint": experiment_fingerprint,
        "global_eval_question_count": len(global_eval_qids),
        "global_eval_question_ids": sorted(global_eval_qids),
        "training_plot": str(plot_path) if plot_path else None,
    })
    return results


if __name__ == "__main__":
    run_three_stage(Config(smoke_test=False))
