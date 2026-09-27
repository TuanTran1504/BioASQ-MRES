#!/usr/bin/env python
"""Run the controlled Qwen2.5-0.5B rationale-SFT -> rationale-DPO comparison.

Arms
----
1. continued_rationale_sft:
   the old answer-only SFT adapter continued on the accepted rationale format.
2. from_base_rationale_sft:
   a fresh rank-32 LoRA adapter trained from the Qwen2.5-0.5B base on the same
   rationale/replay records.
3. old_answer_only_sft:
   the already-trained answer-only SFT adapter, unchanged.

All arms are evaluated with the same rationale prompt and then trained on the
same frozen rationale DPO pairs. Completed SFT, DPO, and evaluation artifacts
are detected by manifests and skipped. No annotation API is called.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = Path("/home/dinh-tuan/miniconda3/envs/bioasq/bin/python")
BASE_MODEL = ROOT / "models/Qwen2.5-0.5B-Instruct"
OLD_SFT_ADAPTER = (
    ROOT
    / "Artifacts/Factoid_SFT/models/"
    "evidence_grounded_per_supported_alias_qwen25_05b_lora_dropout_005_strict_extractive/"
    "adapter_best_evidence_mrr"
)
CONTINUED_SFT_ROOT = (
    ROOT
    / "Artifacts/Factoid_SFT/models/"
    "qwen25_05b_grounded_rationale_sft_positive1111_replay20_v1"
)
CONTINUED_SFT_ADAPTER = CONTINUED_SFT_ROOT / "adapter_final"
FROM_BASE_SFT_ROOT = (
    ROOT
    / "Artifacts/Factoid_SFT/models/"
    "qwen25_05b_grounded_rationale_sft_from_base_positive1111_replay20_v1"
)
FROM_BASE_SFT_ADAPTER = FROM_BASE_SFT_ROOT / "adapter_final"
SFT_DATA_DIR = CONTINUED_SFT_ROOT / "prepared_data"
SFT_MIXED_FILE = SFT_DATA_DIR / "mixed_train.jsonl"
RATIONALE_BANK = (
    ROOT
    / "Artifacts/cse_dpo/gold_answer_rationales/"
    "gpt41mini_gold_supported_1130_v2/gold_answer_rationales.jsonl"
)
C1_RATIONALE_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/c1_answer_rationales/"
    "gpt41mini_strict_equivalence_v2_positive1111_rubric_v3"
)
C1_RATIONALES = C1_RATIONALE_ROOT / "c1_answer_rationales.jsonl"
SOURCE_TRAIN = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "evidence_grounded_single_answer_qwen25_05b/train_prepared.json"
)
DEV_INPUT = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "single_answer_full_resources_qwen25_05b/eval_prepared.json"
)
PROMPT_FILE = ROOT / "prompts/factoid_grounded_rationale_sft.json"
DPO_DATA_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/rationale_dpo_pairs/"
    "strict_equivalence_v2_positive1111_c1_v3"
)
COMPARISON_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/"
    "rationale_sft_dpo_comparison_qwen25_05b_v1"
)

SEED = 3407
MAX_LENGTH = 4096
MAX_NEW_TOKENS = 192
ANSWER_ONLY_REPLAY_FRACTION = 0.20
SFT_EARLY_STOPPING_PATIENCE = 2
SFT_EARLY_STOPPING_MIN_DELTA = 0.0

ARM_NAMES = (
    "continued_rationale_sft",
    "from_base_rationale_sft",
    "old_answer_only_sft",
)


@dataclass(frozen=True)
class Arm:
    name: str
    sft_adapter: Path
    sft_root: Path | None
    train_sft: bool
    from_base: bool
    sft_epochs: float
    sft_learning_rate: float


ARMS = {
    "continued_rationale_sft": Arm(
        "continued_rationale_sft",
        CONTINUED_SFT_ADAPTER,
        CONTINUED_SFT_ROOT,
        True,
        False,
        8.0,
        2e-5,
    ),
    "from_base_rationale_sft": Arm(
        "from_base_rationale_sft",
        FROM_BASE_SFT_ADAPTER,
        FROM_BASE_SFT_ROOT,
        True,
        True,
        8.0,
        6e-4,
    ),
    "old_answer_only_sft": Arm(
        "old_answer_only_sft",
        OLD_SFT_ADAPTER,
        None,
        False,
        False,
        0.0,
        0.0,
    ),
}


def ensure_project_python() -> None:
    """Restart under the tested project environment when launched from Conda base."""
    current = Path(sys.executable).resolve()
    expected = PROJECT_PYTHON.resolve()
    if current == expected:
        return
    if not PROJECT_PYTHON.exists():
        raise RuntimeError(
            f"This script requires the bioasq environment, but {PROJECT_PYTHON} is missing. "
            "Run `conda activate bioasq` before launching it."
        )
    print(
        f"[environment] Restarting with {PROJECT_PYTHON} instead of {sys.executable}",
        flush=True,
    )
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def resources_from_row(row: dict[str, Any]) -> list[str]:
    resources: list[tuple[int, str]] = []
    for key, value in row.items():
        match = re.fullmatch(r"input_(\d+)", str(key))
        if match and int(match.group(1)) >= 2 and str(value or "").strip():
            resources.append((int(match.group(1)), str(value).strip()))
    return [value for _, value in sorted(resources)]


def parse_aliases(output: Any) -> list[str]:
    values = [
        normalized_one_line(match)
        for match in re.findall(r"\[BE\]\s*(.*?)\s*\[EE\]", str(output), flags=re.I | re.S)
    ]
    return list(dict.fromkeys(value for value in values if value))


def load_prompt_spec() -> dict[str, Any]:
    spec = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    required = {"prompt_id", "grounded_rationale_system", "answer_only_replay_system"}
    if not required <= set(spec):
        raise ValueError(f"{PROMPT_FILE} is missing {sorted(required - set(spec))}")
    return spec


def format_user_context(source_row: dict[str, Any], snippets: list[dict[str, Any]]) -> str:
    lines = [
        f"Snippet {snippet['snippet_id']} (PubMed {snippet['pubmed_id']}): "
        f"{normalized_one_line(snippet['text'])}"
        for snippet in snippets
    ]
    return (
        f"Question: {normalized_one_line(source_row['input_1'])}\n\n"
        "PubMed snippets:\n"
        + "\n".join(lines)
    )


def chat_token_length(
    tokenizer: Any,
    system: str,
    user: str,
    assistant_completions: tuple[str, ...] = (),
    reserve_generation_tokens: int = 0,
) -> int:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant_completions:
        return max(
            len(
                tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": completion}],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False,
                )
            )
            for completion in assistant_completions
        )
    return (
        len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        + reserve_generation_tokens
    )


def select_whole_snippets_to_fit(
    tokenizer: Any,
    source_row: dict[str, Any],
    system: str,
    *,
    assistant_completions: tuple[str, ...] = (),
    reserve_generation_tokens: int = 0,
    priority_ids: Iterable[str] = (),
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Preserve required evidence and add whole distractors in original order."""
    from cse_dpo.candidate_bank_class_judge import extract_snippets

    all_snippets = extract_snippets(resources_from_row(source_row))
    if not all_snippets:
        raise ValueError(f"{source_row['id']}: no snippets")
    required = {str(value) for value in priority_ids}
    known = {snippet["snippet_id"] for snippet in all_snippets}
    if not required <= known:
        raise ValueError(f"{source_row['id']}: unknown required snippet IDs {sorted(required-known)}")
    selected_ids = set(required)

    def ordered(ids: set[str]) -> list[dict[str, Any]]:
        return [snippet for snippet in all_snippets if snippet["snippet_id"] in ids]

    initial_user = format_user_context(source_row, ordered(selected_ids))
    if chat_token_length(
        tokenizer,
        system,
        initial_user,
        assistant_completions,
        reserve_generation_tokens,
    ) > MAX_LENGTH:
        raise ValueError(f"{source_row['id']}: required evidence/completions exceed {MAX_LENGTH}")

    for snippet in all_snippets:
        snippet_id = snippet["snippet_id"]
        if snippet_id in selected_ids:
            continue
        proposed = selected_ids | {snippet_id}
        user = format_user_context(source_row, ordered(proposed))
        if chat_token_length(
            tokenizer,
            system,
            user,
            assistant_completions,
            reserve_generation_tokens,
        ) <= MAX_LENGTH:
            selected_ids.add(snippet_id)

    selected = ordered(selected_ids)
    user = format_user_context(source_row, selected)
    dropped = [
        snippet["snippet_id"]
        for snippet in all_snippets
        if snippet["snippet_id"] not in selected_ids
    ]
    return user, selected, dropped


def encode_sft_record(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = tokenizer.apply_chat_template(
        row["messages"][:2], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False, return_dict=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError(f"Chat-template prefix mismatch: {row['id']}")
    if len(full_ids) > MAX_LENGTH:
        raise ValueError(f"{row['id']}: {len(full_ids)} tokens exceed {MAX_LENGTH}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
    }


def build_sft_data(tokenizer: Any, prompt_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Build or validate the exact 1,111-rationale + 278-replay training set."""
    from cse_dpo.candidate_bank_class_judge import candidate_is_extractive, extract_snippets

    rationale_path = SFT_DATA_DIR / "rationale_sft_accepted1111.jsonl"
    replay_path = SFT_DATA_DIR / "answer_only_replay.jsonl"
    frozen_paths = (rationale_path, replay_path, SFT_MIXED_FILE)
    if all(path.exists() for path in frozen_paths):
        rationale_examples = read_jsonl(rationale_path)
        replay = read_jsonl(replay_path)
        mixed = read_jsonl(SFT_MIXED_FILE)
        if (
            len(rationale_examples) != 1111
            or len(replay) != 278
            or len(mixed) != 1389
            or len({row["id"] for row in mixed}) != 1389
        ):
            raise ValueError("Existing frozen rationale SFT files have unexpected counts or duplicate IDs")
        if {row["id"] for row in mixed} != {
            row["id"] for row in [*rationale_examples, *replay]
        }:
            raise ValueError("Existing mixed SFT file does not match its rationale/replay components")
        dev_ids = {
            str(row["id"])
            for row in json.loads(DEV_INPUT.read_text(encoding="utf-8"))
        }
        if {row["question_id"] for row in mixed} & dev_ids:
            raise ValueError("Frozen rationale SFT data overlaps the 160-question dev split")
        for row in rationale_examples:
            if row.get("mode") != "grounded_rationale":
                raise ValueError(f"Unexpected rationale mode: {row.get('id')}")
            if row["messages"][0]["content"] != prompt_spec["grounded_rationale_system"]:
                raise ValueError(f"Rationale prompt changed: {row['id']}")
            if not re.fullmatch(
                r"Reason: .+\nAnswer: \[BE\][^\n]+\[EE\]",
                row["messages"][2]["content"],
            ):
                raise ValueError(f"Invalid rationale target: {row['id']}")
        for row in replay:
            if row.get("mode") != "answer_only_replay":
                raise ValueError(f"Unexpected replay mode: {row.get('id')}")
            if row["messages"][0]["content"] != prompt_spec["answer_only_replay_system"]:
                raise ValueError(f"Replay prompt changed: {row['id']}")
            if not re.fullmatch(
                r"Answer: \[BE\][^\n]+\[EE\]", row["messages"][2]["content"]
            ):
                raise ValueError(f"Invalid replay target: {row['id']}")
        encoded = [encode_sft_record(tokenizer, row) for row in mixed]
        summary = {
            "version": "factoid-grounded-rationale-sft-data-v1",
            "prompt_id": prompt_spec["prompt_id"],
            "rationale_examples": len(rationale_examples),
            "answer_only_replay_examples": len(replay),
            "mixed_examples": len(mixed),
            "unique_questions": len({row["question_id"] for row in mixed}),
            "answer_only_fraction": len(replay) / len(mixed),
            "max_tokens": max(len(item["input_ids"]) for item in encoded),
            "max_target_tokens": max(
                sum(token != -100 for token in item["labels"]) for item in encoded
            ),
            "truncated_examples": 0,
            "train_dev_disjoint": True,
            "mixed_sha256": sha256_file(SFT_MIXED_FILE),
            "reuse_status": "validated_existing_frozen_export",
        }
        write_json(SFT_DATA_DIR / "comparison_driver_preflight.json", summary)
        print("[data] rationale SFT:", json.dumps(summary, indent=2))
        return mixed

    if any(path.exists() for path in frozen_paths):
        raise FileNotFoundError(
            "The rationale SFT export is only partially present. Restore all three files or use a new output root."
        )

    source_rows = json.loads(SOURCE_TRAIN.read_text(encoding="utf-8"))
    source_by_id = {str(row["id"]): row for row in source_rows}
    dev_ids = {
        str(row["id"])
        for row in json.loads(DEV_INPUT.read_text(encoding="utf-8"))
    }
    positive_rows = [
        row for row in read_jsonl(RATIONALE_BANK) if row.get("status") == "accepted"
    ]
    if len(positive_rows) != 1111 or len({row["question_id"] for row in positive_rows}) != 1111:
        raise ValueError("Expected exactly 1,111 unique accepted positive rationales")
    if {row["question_id"] for row in positive_rows} & dev_ids:
        raise ValueError("Rationale SFT data overlaps the 160-question dev split")

    rationale_examples: list[dict[str, Any]] = []
    replay_candidates: list[dict[str, Any]] = []
    for annotation in positive_rows:
        qid = str(annotation["question_id"])
        source = source_by_id[qid]
        all_snippets = extract_snippets(resources_from_row(source))
        evidence_ids = [str(value) for value in annotation["evidence_ids"]]
        cited = [
            snippet for snippet in all_snippets if snippet["snippet_id"] in set(evidence_ids)
        ]
        answer = normalized_one_line(annotation["chosen_answer"])
        reason = normalized_one_line(annotation["reason"])
        if not evidence_ids or not candidate_is_extractive(answer, cited):
            raise ValueError(f"{qid}: positive answer is not extractive from cited evidence")
        chosen = f"Reason: {reason}\nAnswer: [BE]{answer}[EE]"
        user, selected, dropped = select_whole_snippets_to_fit(
            tokenizer,
            source,
            prompt_spec["grounded_rationale_system"],
            assistant_completions=(chosen,),
            priority_ids=evidence_ids,
        )
        metadata = {
            "answer": answer,
            "evidence_ids": evidence_ids,
            "included_snippet_ids": [snippet["snippet_id"] for snippet in selected],
            "dropped_snippet_ids": dropped,
        }
        rationale_examples.append(
            {
                "id": f"{qid}::rationale",
                "question_id": qid,
                "mode": "grounded_rationale",
                "messages": [
                    {"role": "system", "content": prompt_spec["grounded_rationale_system"]},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": chosen},
                ],
                "metadata": metadata,
            }
        )
        replay_candidates.append(
            {
                "id": f"{qid}::answer_only_replay",
                "question_id": qid,
                "mode": "answer_only_replay",
                "messages": [
                    {"role": "system", "content": prompt_spec["answer_only_replay_system"]},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": f"Answer: [BE]{answer}[EE]"},
                ],
                "metadata": metadata,
            }
        )

    replay_count = round(
        len(rationale_examples)
        * ANSWER_ONLY_REPLAY_FRACTION
        / (1 - ANSWER_ONLY_REPLAY_FRACTION)
    )
    rng = random.Random(SEED)
    replay = rng.sample(replay_candidates, replay_count)
    mixed = rationale_examples + replay
    rng.shuffle(mixed)
    encoded = [encode_sft_record(tokenizer, row) for row in mixed]
    if len(mixed) != 1389 or len(replay) != 278:
        raise ValueError("Unexpected rationale/replay record count")

    generated_files = {
        "rationale_sft_accepted1111.jsonl": rationale_examples,
        "answer_only_replay.jsonl": replay,
        "mixed_train.jsonl": mixed,
    }
    SFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in generated_files.items():
        path = SFT_DATA_DIR / name
        if path.exists() and read_jsonl(path) != rows:
            raise ValueError(f"Frozen SFT data changed: {path}")
        if not path.exists():
            write_jsonl(path, rows)
    summary = {
        "version": "factoid-grounded-rationale-sft-data-v1",
        "prompt_id": prompt_spec["prompt_id"],
        "rationale_examples": len(rationale_examples),
        "answer_only_replay_examples": len(replay),
        "mixed_examples": len(mixed),
        "unique_questions": len({row["question_id"] for row in mixed}),
        "answer_only_fraction": len(replay) / len(mixed),
        "max_tokens": max(len(item["input_ids"]) for item in encoded),
        "max_target_tokens": max(
            sum(token != -100 for token in item["labels"]) for item in encoded
        ),
        "truncated_examples": 0,
        "train_dev_disjoint": True,
        "source_hashes": {
            "rationale_bank": sha256_file(RATIONALE_BANK),
            "source_train": sha256_file(SOURCE_TRAIN),
            "dev_input": sha256_file(DEV_INPUT),
            "prompt_file": sha256_file(PROMPT_FILE),
        },
        "mixed_sha256": sha256_file(SFT_MIXED_FILE),
    }
    write_json(SFT_DATA_DIR / "comparison_driver_preflight.json", summary)
    print("[data] rationale SFT:", json.dumps(summary, indent=2))
    return mixed


def build_dpo_data(tokenizer: Any, prompt_spec: dict[str, Any]) -> Path:
    """Freeze accepted C3+rationale > C1+rationale pairs under the new prompt."""
    source_rows = json.loads(SOURCE_TRAIN.read_text(encoding="utf-8"))
    source_by_id = {str(row["id"]): row for row in source_rows}
    dev_ids = {
        str(row["id"])
        for row in json.loads(DEV_INPUT.read_text(encoding="utf-8"))
    }
    annotations = [
        row for row in read_jsonl(C1_RATIONALES) if row.get("status") == "accepted"
    ]
    if len(annotations) != 466 or len({row["pair_id"] for row in annotations}) != 466:
        raise ValueError("Expected exactly 466 unique accepted C1 rationale annotations")
    if {row["question_id"] for row in annotations} & dev_ids:
        raise ValueError("Rationale DPO pairs overlap the 160-question dev split")

    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        qid = str(annotation["question_id"])
        source = source_by_id[qid]
        positive_ids = [str(value) for value in annotation["positive_evidence_ids"]]
        negative_ids = [str(value) for value in annotation["evidence_ids"]]
        chosen = (
            f"Reason: {normalized_one_line(annotation['positive_reason'])}\n"
            f"Answer: [BE]{normalized_one_line(annotation['positive_answer'])}[EE]"
        )
        rejected = (
            f"Reason: {normalized_one_line(annotation['reason'])}\n"
            f"Answer: [BE]{normalized_one_line(annotation['c1_answer'])}[EE]"
        )
        user, selected, dropped = select_whole_snippets_to_fit(
            tokenizer,
            source,
            prompt_spec["grounded_rationale_system"],
            assistant_completions=(chosen, rejected),
            priority_ids=[*positive_ids, *negative_ids],
        )
        messages = [
            {"role": "system", "content": prompt_spec["grounded_rationale_system"]},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # The production DPO trainer appends EOS to each completion.
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        chosen_tokens = len(tokenizer(chosen, add_special_tokens=False)["input_ids"]) + 1
        rejected_tokens = len(tokenizer(rejected, add_special_tokens=False)["input_ids"]) + 1
        total = prompt_tokens + max(chosen_tokens, rejected_tokens)
        if total > MAX_LENGTH:
            raise ValueError(f"{annotation['pair_id']}: pair has {total} tokens")
        rows.append(
            {
                "pair_id": str(annotation["pair_id"]),
                "question_id": qid,
                "split": "train",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "positive_answer": annotation["positive_answer"],
                "c1_answer": annotation["c1_answer"],
                "positive_evidence_ids": positive_ids,
                "c1_evidence_ids": negative_ids,
                "relation_type": annotation.get("relation_type"),
                "error_type": annotation.get("error_type"),
                "c1_source_model": annotation.get("c1_source_model"),
                "included_snippet_ids": [snippet["snippet_id"] for snippet in selected],
                "dropped_snippet_ids": dropped,
                "prompt_tokens": prompt_tokens,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "max_sequence_tokens": total,
                "prompt_id": prompt_spec["prompt_id"],
            }
        )

    DPO_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    stage1 = DPO_DATA_ROOT / "dpo_stage1_concept_learning_all_pairs.jsonl"
    if stage1.exists() and read_jsonl(stage1) != rows:
        raise ValueError(f"Frozen rationale DPO pairs changed: {stage1}")
    if not stage1.exists():
        write_jsonl(stage1, rows)
    for filename in (
        "dpo_stage2_format_alignment_all_pairs.jsonl",
        "dpo_stage3_hierarchical_ranking_all_pairs.jsonl",
    ):
        path = DPO_DATA_ROOT / filename
        if not path.exists():
            path.write_text("", encoding="utf-8")
        elif path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Expected an empty unused stage file: {path}")
    summary = {
        "version": "factoid-answer-rationale-dpo-pairs-v1",
        "prompt_id": prompt_spec["prompt_id"],
        "pairs": len(rows),
        "questions": len({row["question_id"] for row in rows}),
        "max_sequence_tokens": max(row["max_sequence_tokens"] for row in rows),
        "pairs_requiring_context_selection": sum(
            bool(row["dropped_snippet_ids"]) for row in rows
        ),
        "max_length": MAX_LENGTH,
        "truncated_pairs": 0,
        "pair_file_sha256": sha256_file(stage1),
        "c1_annotation_sha256": sha256_file(C1_RATIONALES),
        "prompt_sha256": sha256_file(PROMPT_FILE),
        "source_train_sha256": sha256_file(SOURCE_TRAIN),
        "train_dev_disjoint": True,
        "old_source_prompt_ignored": True,
        "context_policy": (
            "preserve positive and C1 cited snippets; add whole distractors in source order; "
            "never truncate"
        ),
    }
    write_json(DPO_DATA_ROOT / "summary.json", summary)
    print("[data] rationale DPO:", json.dumps(summary, indent=2))
    return stage1


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch = {
            key: torch.tensor(
                [feature[key] + [pad] * (width - len(feature[key])) for feature in features],
                dtype=torch.long,
            )
            for key, pad in (
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }
        shifted = torch.nn.functional.pad(batch["labels"][:, 1:], (0, 1), value=-100)
        positions = torch.where(shifted.ne(-100).any(dim=0))[0]
        if positions.numel() == 0:
            raise ValueError("Batch contains no supervised tokens")
        batch["logits_to_keep"] = positions
        batch["shift_labels"] = shifted[:, positions].contiguous()
        return batch


def adapter_is_complete(adapter: Path, completion_marker: Path | None = None) -> bool:
    valid = (adapter / "adapter_config.json").exists() and (
        adapter / "adapter_model.safetensors"
    ).exists()
    if completion_marker is not None:
        valid = valid and completion_marker.exists()
    return valid


def train_sft_arm(
    arm: Arm,
    tokenizer: Any,
    mixed_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    *,
    preflight_only: bool,
) -> Path:
    if not arm.train_sft:
        if not adapter_is_complete(arm.sft_adapter):
            raise FileNotFoundError(f"Missing existing old SFT adapter: {arm.sft_adapter}")
        print(f"[{arm.name}] existing old SFT adapter; no retraining: {arm.sft_adapter}")
        return arm.sft_adapter

    assert arm.sft_root is not None
    marker = arm.sft_root / "training_complete.json"
    if adapter_is_complete(arm.sft_adapter, marker):
        print(f"[{arm.name}] SFT already complete; skipping: {arm.sft_adapter}")
        return arm.sft_adapter
    if preflight_only:
        print(
            f"[{arm.name}] would train SFT for at most {arm.sft_epochs:g} epochs from "
            f"{'base model' if arm.from_base else OLD_SFT_ADAPTER}; evaluate official "
            f"dev MRR at epoch 0 and after every epoch; stop after "
            f"{SFT_EARLY_STOPPING_PATIENCE} non-improving epochs; save best to "
            f"{arm.sft_adapter}"
        )
        return arm.sft_adapter

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    class DevMRREarlyStoppingCallback(TrainerCallback):
        """Generate the full dev split each epoch and retain the best-MRR adapter."""

        def __init__(self) -> None:
            self.history_path = arm.sft_root / "sft_dev_mrr_history.json"
            self.eval_root = arm.sft_root / "dev_mrr_evaluations"
            self.history: list[dict[str, Any]] = []
            self.best_mrr: float | None = None
            self.best_epoch: int | None = None
            self.bad_epochs = 0
            if self.history_path.exists():
                saved = json.loads(self.history_path.read_text(encoding="utf-8"))
                self.history = list(saved.get("history", []))
                self.best_mrr = saved.get("best_mrr")
                self.best_epoch = saved.get("best_epoch")
                self.bad_epochs = int(saved.get("bad_epochs", 0))

        def _write_state(self, *, stopped_early: bool = False) -> None:
            write_json(
                self.history_path,
                {
                    "selection_metric": "official_bioasq_dev_mrr",
                    "mode": "max",
                    "max_epochs": arm.sft_epochs,
                    "patience": SFT_EARLY_STOPPING_PATIENCE,
                    "min_delta": SFT_EARLY_STOPPING_MIN_DELTA,
                    "best_mrr": self.best_mrr,
                    "best_epoch": self.best_epoch,
                    "bad_epochs": self.bad_epochs,
                    "stopped_early": stopped_early,
                    "best_adapter": str(arm.sft_adapter.resolve()),
                    "history": self.history,
                },
            )

        def _save_best(self, model: Any) -> None:
            if arm.sft_adapter.exists():
                shutil.rmtree(arm.sft_adapter)
            arm.sft_adapter.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(arm.sft_adapter)
            tokenizer.save_pretrained(arm.sft_adapter)

        def _evaluate_epoch(self, epoch: int, model: Any, global_step: int) -> None:
            output_dir = self.eval_root / f"epoch_{epoch:03d}"
            summary = evaluate_adapter(
                arm.sft_adapter,
                tokenizer,
                dev_records,
                output_dir,
                f"{arm.name}-sft-epoch-{epoch}",
                prompt_spec,
                preflight_only=False,
                loaded_model=model,
                adapter_hash_override=f"in-memory-epoch-{epoch:03d}",
                adapter_label=f"{arm.name}:in-memory-epoch-{epoch:03d}",
            )
            assert summary is not None
            mrr = float(summary["official_metrics"]["mrr"])
            improved = (
                self.best_mrr is None
                or mrr > self.best_mrr + SFT_EARLY_STOPPING_MIN_DELTA
            )
            if improved:
                self.best_mrr = mrr
                self.best_epoch = epoch
                self.bad_epochs = 0
                self._save_best(model)
            else:
                self.bad_epochs += 1
            self.history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "dev_mrr": mrr,
                    "format_valid_rate": summary["format_valid_rate"],
                    "citation_id_valid_rate": summary["citation_id_valid_rate"],
                    "answer_extractive_rate": summary["answer_extractive_rate"],
                    "improved": improved,
                    "bad_epochs_after_evaluation": self.bad_epochs,
                    "evaluation_summary": str(
                        (output_dir / "evaluation_summary.json").resolve()
                    ),
                }
            )
            self._write_state()
            print(
                f"[{arm.name}] epoch {epoch} dev MRR={mrr:.6f}; "
                f"best={self.best_mrr:.6f} at epoch {self.best_epoch}; "
                f"non-improving epochs={self.bad_epochs}/"
                f"{SFT_EARLY_STOPPING_PATIENCE}",
                flush=True,
            )

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if not self.history:
                self._evaluate_epoch(0, kwargs["model"], int(state.global_step))
            else:
                print(
                    f"[{arm.name}] restored SFT MRR history through epoch "
                    f"{self.history[-1]['epoch']}; best={self.best_mrr}",
                    flush=True,
                )
            return control

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            epoch = int(round(float(state.epoch or 0.0)))
            if epoch <= 0 or any(int(row["epoch"]) == epoch for row in self.history):
                return control
            self._evaluate_epoch(epoch, kwargs["model"], int(state.global_step))
            if self.bad_epochs >= SFT_EARLY_STOPPING_PATIENCE:
                control.should_training_stop = True
                self._write_state(stopped_early=True)
                print(
                    f"[{arm.name}] early stopping after epoch {epoch}: no dev MRR "
                    f"improvement for {self.bad_epochs} consecutive epochs.",
                    flush=True,
                )
            return control

        def summary(self) -> dict[str, Any]:
            return {
                "selection_metric": "official_bioasq_dev_mrr",
                "max_epochs": arm.sft_epochs,
                "patience": SFT_EARLY_STOPPING_PATIENCE,
                "min_delta": SFT_EARLY_STOPPING_MIN_DELTA,
                "best_mrr": self.best_mrr,
                "best_epoch": self.best_epoch,
                "bad_epochs": self.bad_epochs,
                "evaluated_epochs": [int(row["epoch"]) for row in self.history],
                "stopped_early": self.bad_epochs >= SFT_EARLY_STOPPING_PATIENCE,
                "history_file": str(self.history_path.resolve()),
            }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT training")
    encoded = [encode_sft_record(tokenizer, row) for row in mixed_records]
    set_seed(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if arm.from_base:
        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model, OLD_SFT_ADAPTER, is_trainable=True, local_files_only=True
        )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    checkpoint_dir = arm.sft_root / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=arm.sft_epochs,
        learning_rate=arm.sft_learning_rate,
        weight_decay=0.01,
        warmup_steps=5 if arm.from_base else 0,
        warmup_ratio=0.0 if arm.from_base else 0.05,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
    )
    mrr_callback = DevMRREarlyStoppingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(encoded),
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        callbacks=[mrr_callback],
    )
    resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    result = trainer.train(resume_from_checkpoint=resume)
    if not adapter_is_complete(arm.sft_adapter):
        raise RuntimeError(
            f"MRR selection did not produce a complete best adapter: {arm.sft_adapter}"
        )
    early_stopping = mrr_callback.summary()
    manifest = {
        "status": "complete",
        "arm": arm.name,
        "from_base": arm.from_base,
        "initial_adapter": None if arm.from_base else str(OLD_SFT_ADAPTER.resolve()),
        "final_adapter": str(arm.sft_adapter.resolve()),
        "final_adapter_policy": "best official BioASQ dev MRR across epoch 0 and epoch ends",
        "train_file": str(SFT_MIXED_FILE.resolve()),
        "train_file_sha256": sha256_file(SFT_MIXED_FILE),
        "epochs_requested": arm.sft_epochs,
        "epochs_completed": trainer.state.epoch,
        "learning_rate": arm.sft_learning_rate,
        "gradient_accumulation_steps": 32,
        "max_length": MAX_LENGTH,
        "early_stopping": early_stopping,
        "train_metrics": result.metrics,
    }
    write_json(arm.sft_root / "run_manifest.json", manifest)
    write_json(
        marker,
        {
            "status": "complete",
            "final_adapter": str(arm.sft_adapter.resolve()),
            "best_epoch": early_stopping["best_epoch"],
            "best_dev_mrr": early_stopping["best_mrr"],
            "stopped_early": early_stopping["stopped_early"],
        },
    )
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return arm.sft_adapter

def build_dev_records(
    tokenizer: Any, prompt_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    dev_rows = json.loads(DEV_INPUT.read_text(encoding="utf-8"))
    records = []
    for row in dev_rows:
        user, snippets, dropped = select_whole_snippets_to_fit(
            tokenizer,
            row,
            prompt_spec["grounded_rationale_system"],
            reserve_generation_tokens=MAX_NEW_TOKENS,
        )
        messages = [
            {"role": "system", "content": prompt_spec["grounded_rationale_system"]},
            {"role": "user", "content": user},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(prompt_ids) + MAX_NEW_TOKENS > MAX_LENGTH:
            raise ValueError(f"{row['id']}: dev prompt exceeds generation budget")
        records.append(
            {
                "question_id": str(row["id"]),
                "question": normalized_one_line(row["input_1"]),
                "gold_aliases": parse_aliases(row["output"]),
                "prompt_ids": prompt_ids,
                "snippets": snippets,
                "included_snippet_ids": [snippet["snippet_id"] for snippet in snippets],
                "dropped_snippet_ids": dropped,
            }
        )
    if len(records) != 160 or len({row["question_id"] for row in records}) != 160:
        raise ValueError("Expected exactly 160 unique dev questions")
    return records


def evaluate_adapter(
    adapter: Path,
    tokenizer: Any,
    dev_records: list[dict[str, Any]],
    output_dir: Path,
    label: str,
    prompt_spec: dict[str, Any],
    *,
    preflight_only: bool,
    loaded_model: Any | None = None,
    adapter_hash_override: str | None = None,
    adapter_label: str | None = None,
) -> dict[str, Any] | None:
    marker = output_dir / "evaluation_summary.json"
    adapter_hash = (
        adapter_hash_override
        if adapter_hash_override is not None
        else sha256_file(adapter / "adapter_model.safetensors")
        if adapter.exists()
        else None
    )
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (
            existing.get("adapter_sha256") == adapter_hash
            and existing.get("prompt_sha256") == sha256_file(PROMPT_FILE)
        ):
            print(f"[{label}] evaluation already complete; skipping: {marker}")
            return existing
        raise ValueError(f"Evaluation inputs changed; use a new output directory: {output_dir}")
    if preflight_only:
        print(f"[{label}] would evaluate {adapter} on 160 dev questions")
        return None
    if loaded_model is None and not adapter_is_complete(adapter):
        raise FileNotFoundError(adapter)

    import torch
    from src.utility.bioasq_official import evaluate_with_bioasq_java
    from src.utility.eval_types import EvalExample

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generated evaluation")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    owns_model = loaded_model is None
    base = None
    if owns_model:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            local_files_only=True,
            torch_dtype=dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
        )
        model = PeftModel.from_pretrained(
            base, adapter, is_trainable=False, local_files_only=True
        )
    else:
        model = loaded_model
    was_training = bool(model.training)
    prior_use_cache = bool(model.config.use_cache)
    model.eval()
    model.config.use_cache = True
    predictions = []
    from cse_dpo.candidate_bank_class_judge import candidate_is_extractive

    for index, row in enumerate(dev_records, 1):
        inputs = torch.tensor([row["prompt_ids"]], device=model.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                logits_to_keep=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output[0, inputs.shape[1] :].cpu(), skip_special_tokens=True
        ).strip()
        aliases = parse_aliases(raw)
        answer = aliases[-1] if aliases else ""
        reason_match = re.match(r"^Reason:\s*(.*?)\nAnswer:", raw, flags=re.S)
        reason = reason_match.group(1).strip() if reason_match else ""
        cited_ids = list(dict.fromkeys(re.findall(r"\b\d+\.\d+\b", reason)))
        format_valid = bool(
            re.fullmatch(r"Reason: .+\nAnswer: \[BE\][^\n]+\[EE\]", raw, flags=re.S)
        )
        citation_ids_valid = bool(cited_ids) and len(cited_ids) <= 2 and set(cited_ids) <= set(
            row["included_snippet_ids"]
        )
        answer_extractive = bool(answer) and candidate_is_extractive(answer, row["snippets"])
        predictions.append(
            {
                "question_id": row["question_id"],
                "question_type": "factoid",
                "body": row["question"],
                "source_path": str(DEV_INPUT),
                "prediction": f"[BE]{answer}[EE]" if answer else "",
                "answer": answer,
                "reason": reason,
                "cited_ids": cited_ids,
                "raw_prediction": raw,
                "format_valid": format_valid,
                "citation_ids_valid": citation_ids_valid,
                "answer_extractive": answer_extractive,
                "dropped_snippet_ids": row["dropped_snippet_ids"],
            }
        )
        if index % 20 == 0:
            print(f"[{label}] generated {index}/{len(dev_records)}", flush=True)

    examples: dict[tuple[str, str], Any] = {}
    for row in dev_records:
        raw_question = {
            "id": row["question_id"],
            "type": "factoid",
            "body": row["question"],
            "exact_answer": [list(row["gold_aliases"])],
        }
        examples[(row["question_id"], "factoid")] = EvalExample(
            row["question_id"],
            "factoid",
            row["question"],
            "",
            (),
            f"[BE]{row['gold_aliases'][0]}[EE]",
            str(DEV_INPUT),
            raw_question,
        )
    official = evaluate_with_bioasq_java(
        prediction_rows=predictions,
        examples_by_key=examples,
        model_label=label,
        model_dir=output_dir / "official_bioasq",
        args=argparse.Namespace(
            bioasq_java_jar=str(
                ROOT
                / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/"
                "BioASQEvaluation.jar"
            ),
            bioasq_java_heap="512m",
            bioasq_java_version=5,
        ),
        include_per_question=True,
    )
    by_id = {row["question_id"]: row for row in official["per_question"]}
    for row in predictions:
        row.update(
            {
                key: value
                for key, value in by_id[row["question_id"]].items()
                if key not in {"question_id", "question_type"}
            }
        )
    metrics = official["aggregate"]["by_type"]["factoid"]["metrics"]
    summary = {
        "status": "complete",
        "label": label,
        "adapter": adapter_label or str(adapter.resolve()),
        "adapter_sha256": adapter_hash,
        "prompt_id": prompt_spec["prompt_id"],
        "prompt_sha256": sha256_file(PROMPT_FILE),
        "question_count": len(predictions),
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "official_metrics": metrics,
        "format_valid_rate": sum(row["format_valid"] for row in predictions)
        / len(predictions),
        "citation_id_valid_rate": sum(row["citation_ids_valid"] for row in predictions)
        / len(predictions),
        "answer_extractive_rate": sum(row["answer_extractive"] for row in predictions)
        / len(predictions),
        "questions_requiring_context_selection": sum(
            bool(row["dropped_snippet_ids"]) for row in predictions
        ),
        "rationale_semantic_correctness": "not automatically judged",
    }
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(marker, summary)
    print(f"[{label}]", json.dumps(summary, indent=2))
    model.config.use_cache = prior_use_cache
    if was_training:
        model.train()
    if owns_model:
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    return summary


def run_dpo_arm(
    arm: Arm,
    initial_adapter: Path,
    *,
    preflight_only: bool,
) -> Path:
    output_root = COMPARISON_ROOT / "dpo_runs" / arm.name
    stage_dir = output_root / "stage_1_concept_learning"
    stage_manifest = stage_dir / "manifest.json"
    if stage_manifest.exists():
        manifest = json.loads(stage_manifest.read_text(encoding="utf-8"))
        adapter = Path(manifest["best_selected_adapter"])
        if not adapter_is_complete(adapter):
            raise FileNotFoundError(f"Completed DPO manifest points to missing adapter: {adapter}")
        print(f"[{arm.name}] rationale DPO already complete; skipping: {adapter}")
        return adapter
    if preflight_only:
        print(f"[{arm.name}] would train standard rationale DPO from {initial_adapter}")
        return output_root / "stage_1_concept_learning/checkpoints/step_0"
    if not adapter_is_complete(initial_adapter):
        raise FileNotFoundError(initial_adapter)

    from cse_dpo.train_factoid_three_stage_dpo import Config, run_three_stage

    cfg = Config(
        model_preset="qwen25_05b",
        dataset_preset="legacy_first400",
        staged_root=str(DPO_DATA_ROOT),
        base_model=str(BASE_MODEL),
        initial_adapter=str(initial_adapter),
        output_root=str(output_root),
        seed=SEED,
        beta=0.05,
        objective="dpo",
        learning_rate=2e-6,
        optimizer="AdamW",
        weight_decay=0.01,
        epochs=8,
        batch_size=1,
        gradient_accumulation=8,
        max_length=MAX_LENGTH,
        train_backprop_max_length=MAX_LENGTH,
        max_grad_norm=1.0,
        warmup_fraction=0.05,
        eval_fraction=0.2,
        eval_seed=SEED,
        eval_every_updates=0,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        selection_metric="dpo_eval_loss",
        allow_truncated_examples=False,
        drop_truncated_examples=True,
        append_eos_to_completions=True,
        evaluate_generated_dev=False,
        lora_dropout=0.05,
        smoke_test=False,
        stop_after_stage="concept_learning",
        skip_stages=[],
        include_auxiliary_c1_over_c0=False,
    )
    run_three_stage(cfg)
    manifest = json.loads(stage_manifest.read_text(encoding="utf-8"))
    adapter = Path(manifest["best_selected_adapter"])
    if not adapter_is_complete(adapter):
        raise FileNotFoundError(adapter)
    return adapter


def arm_plan(arms: list[Arm]) -> dict[str, Any]:
    return {
        "version": "rationale-sft-dpo-three-arm-comparison-v2",
        "base_model": str(BASE_MODEL),
        "prompt_file": str(PROMPT_FILE),
        "sft_data": str(SFT_MIXED_FILE),
        "dpo_data": str(DPO_DATA_ROOT / "dpo_stage1_concept_learning_all_pairs.jsonl"),
        "arms": {
            arm.name: {
                "sft_adapter": str(arm.sft_adapter),
                "train_sft": arm.train_sft,
                "from_base": arm.from_base,
                "sft_epochs": arm.sft_epochs,
                "sft_learning_rate": arm.sft_learning_rate,
                "sft_selection_metric": (
                    "official_bioasq_dev_mrr" if arm.train_sft else None
                ),
                "sft_evaluate_at_epoch_zero": arm.train_sft,
                "sft_eval_strategy": "epoch" if arm.train_sft else None,
                "sft_early_stopping_patience": (
                    SFT_EARLY_STOPPING_PATIENCE if arm.train_sft else None
                ),
                "sft_early_stopping_min_delta": (
                    SFT_EARLY_STOPPING_MIN_DELTA if arm.train_sft else None
                ),
                "sft_best_adapter": (
                    str(arm.sft_adapter) if arm.train_sft else None
                ),
                "dpo_output": str(COMPARISON_ROOT / "dpo_runs" / arm.name),
            }
            for arm in arms
        },
        "shared_dpo": {
            "objective": "dpo",
            "beta": 0.05,
            "learning_rate": 2e-6,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "epochs": 8,
            "gradient_accumulation": 8,
            "eval_fraction_by_question": 0.2,
            "selection_metric": "dpo_eval_loss",
            "early_stopping_patience": 3,
            "max_length": MAX_LENGTH,
            "reference_policy": "the corresponding arm's SFT adapter at DPO initialization",
        },
        "generated_evaluation": {
            "split": "160-question dev",
            "prompt": "shared rationale prompt",
            "decoding": "greedy",
            "max_length": MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "scorer": "official BioASQ Java v5",
        },
    }


def main() -> None:
    ensure_project_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_NAMES,
        default=list(ARM_NAMES),
        help="Subset of the three controlled arms to run.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build/validate frozen data and print the plan without loading model weights.",
    )
    parser.add_argument("--skip-sft-eval", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--skip-dpo-eval", action="store_true")
    args = parser.parse_args()

    required = [
        BASE_MODEL,
        OLD_SFT_ADAPTER,
        RATIONALE_BANK,
        C1_RATIONALES,
        SOURCE_TRAIN,
        DEV_INPUT,
        PROMPT_FILE,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompt_spec = load_prompt_spec()
    selected_arms = [ARMS[name] for name in args.arms]
    plan = arm_plan(selected_arms)
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(COMPARISON_ROOT / "experiment_plan.json", plan)
    print(json.dumps(plan, indent=2))

    mixed = build_sft_data(tokenizer, prompt_spec)
    build_dpo_data(tokenizer, prompt_spec)
    dev_records = build_dev_records(tokenizer, prompt_spec)
    dev_preflight = {
        "questions": len(dev_records),
        "max_prompt_tokens": max(len(row["prompt_ids"]) for row in dev_records),
        "max_total_with_generation": max(len(row["prompt_ids"]) for row in dev_records)
        + MAX_NEW_TOKENS,
        "questions_requiring_context_selection": sum(
            bool(row["dropped_snippet_ids"]) for row in dev_records
        ),
    }
    write_json(COMPARISON_ROOT / "dev_preflight.json", dev_preflight)
    print("[data] dev:", json.dumps(dev_preflight, indent=2))

    if args.preflight_only:
        for arm in selected_arms:
            train_sft_arm(
                arm,
                tokenizer,
                mixed,
                dev_records,
                prompt_spec,
                preflight_only=True,
            )
            run_dpo_arm(arm, arm.sft_adapter, preflight_only=True)
        print("Preflight complete. No model training or generation was run.")
        return

    results: dict[str, Any] = {}
    for arm in selected_arms:
        sft_adapter = train_sft_arm(
            arm,
            tokenizer,
            mixed,
            dev_records,
            prompt_spec,
            preflight_only=False,
        )
        sft_eval = None
        if not args.skip_sft_eval:
            sft_eval = evaluate_adapter(
                sft_adapter,
                tokenizer,
                dev_records,
                COMPARISON_ROOT / "sft_evaluations" / arm.name,
                f"{arm.name}-before-dpo",
                prompt_spec,
                preflight_only=False,
            )
        dpo_adapter = None
        dpo_eval = None
        if not args.skip_dpo:
            dpo_adapter = run_dpo_arm(arm, sft_adapter, preflight_only=False)
            if not args.skip_dpo_eval:
                dpo_eval = evaluate_adapter(
                    dpo_adapter,
                    tokenizer,
                    dev_records,
                    COMPARISON_ROOT / "dpo_evaluations" / arm.name,
                    f"{arm.name}-after-rationale-dpo",
                    prompt_spec,
                    preflight_only=False,
                )
        sft_mrr = (
            sft_eval.get("official_metrics", {}).get("mrr") if sft_eval else None
        )
        dpo_mrr = (
            dpo_eval.get("official_metrics", {}).get("mrr") if dpo_eval else None
        )
        results[arm.name] = {
            "sft_adapter": str(sft_adapter.resolve()),
            "sft_evaluation": sft_eval,
            "dpo_adapter": str(dpo_adapter.resolve()) if dpo_adapter else None,
            "dpo_evaluation": dpo_eval,
            "dev_mrr_change_after_dpo": (
                dpo_mrr - sft_mrr
                if isinstance(dpo_mrr, (int, float))
                and isinstance(sft_mrr, (int, float))
                else None
            ),
        }
        write_json(
            COMPARISON_ROOT / "comparison_summary.json",
            {"status": "running", "plan": plan, "results": results},
        )

    write_json(
        COMPARISON_ROOT / "comparison_summary.json",
        {"status": "complete", "plan": plan, "results": results},
    )
    print(f"Comparison complete: {COMPARISON_ROOT / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
