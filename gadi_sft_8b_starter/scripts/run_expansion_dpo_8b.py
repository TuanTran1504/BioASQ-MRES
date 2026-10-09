#!/usr/bin/env python3
"""Generate training-only response banks or train standard whole-response DPO."""

import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

from expansion_dpo_data import (DEFAULT_CONFIG, MODELS, response_id, split_rows,
                                strict_answers, validate_pairs)
from prepare_matched_8b_sft import ROOT, digest, read, write
from run_matched_8b_evaluation import validate_training_pair
from run_extractive_expansion_8b import (configure_job_local_compiler_cache, input_token_count,
                                       load_model, prepare_inference_execution, tokenize_text,
                                       write_jsonl)
from run_single_answer_sampling import sample_seed


def source_for(config, model):
    evaluation = read(ROOT / config["evaluation_config"])
    return validate_training_pair(evaluation, model)["expansion"]


def render_preference(tokenizer, row, kwargs, maximum):
    """Native chat serialization, exact prefix check, and no silent truncation."""
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                          add_generation_prompt=True, **kwargs)
    result = {}
    for side in ("chosen", "rejected"):
        ids = tokenizer.apply_chat_template(row["prompt"] + [{"role": "assistant", "content": row[side]}],
                                            tokenize=True, add_generation_prompt=False, **kwargs)
        if not isinstance(ids, list) or ids[:len(prompt)] != prompt:
            raise ValueError(f"{row['question_id']}: native assistant tokens do not extend the generation prompt")
        if len(ids) > maximum or len(ids) <= len(prompt):
            raise ValueError(f"{row['question_id']}: completion missing or sequence exceeds {maximum}; refusing truncation")
        result[side + "_input_ids"] = ids
        result[side + "_labels"] = [-100] * len(prompt) + ids[len(prompt):]
    result["question_id"] = row["question_id"]
    result["pair_ids"] = [row["chosen_id"], row["rejected_id"]]
    return result


def collate_pairs(rows, pad_id, torch):
    result = {}
    for side in ("chosen", "rejected"):
        maximum = max(len(r[side + "_input_ids"]) for r in rows)
        ids, labels, masks = [], [], []
        for row in rows:
            length = len(row[side + "_input_ids"])
            ids.append(row[side + "_input_ids"] + [pad_id] * (maximum - length))
            labels.append(row[side + "_labels"] + [-100] * (maximum - length))
            masks.append([1] * length + [0] * (maximum - length))
        result[side + "_input_ids"] = torch.tensor(ids, dtype=torch.long)
        result[side + "_labels"] = torch.tensor(labels, dtype=torch.long)
        result[side + "_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        if all("ref_" + side in r for r in rows):
            result["ref_" + side] = torch.tensor([r["ref_" + side] for r in rows], dtype=torch.float32)
    return result


def completion_logps(logits, labels, torch):
    """Sum next-token log probabilities only over the native assistant completion."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError("DPO requires full sequence logits; enable Unsloth return_logits for this loader")
    totals = []
    for scores, targets in zip(logits[:, :-1], labels[:, 1:]):
        mask = targets != -100
        if not bool(mask.any()):
            raise ValueError("No completion tokens contribute to DPO")
        scores, targets = scores[mask], targets[mask]
        # Avoid a full sequence x vocabulary FP32 log-softmax allocation.
        total = scores.new_zeros((), dtype=torch.float32)
        for offset in range(0, len(targets), 128):
            chunk = scores[offset:offset + 128].float()
            chosen = chunk.gather(1, targets[offset:offset + 128, None]).squeeze(1)
            total = total + (chosen - torch.logsumexp(chunk, dim=-1)).sum()
        totals.append(total)
    return torch.stack(totals)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta, torch):
    margin = beta * ((chosen - ref_chosen) - (rejected - ref_rejected))
    return -torch.nn.functional.logsigmoid(margin).mean()


def generate(args, config, source, output, state):
    panels = split_rows(config, args.full_dataset)
    if args.smoke_test:
        panels = {s: rows[:2] for s, rows in panels.items()}
    state["question_ids"] = {s: [r["question_id"] for r in rows] for s, rows in panels.items()}
    write(output / "manifest.json", state)
    activate = prepare_inference_execution(source)
    import torch
    activate(torch)
    model, tokenizer = load_model(source["adapter"], config["max_seq_length"], False, source["model_loader"])
    configure_job_local_compiler_cache()
    device = next(model.parameters()).device
    prepared = []
    for split, rows in panels.items():
        for row in rows:
            prompt = row["messages"][:2]
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                                 **source["chat_template_kwargs"])
            count = input_token_count(tokenize_text(tokenizer, text, add_special_tokens=True))
            if count > config["max_seq_length"] - config["max_new_tokens"]:
                raise ValueError(f"{row['question_id']}: all snippets exceed generation budget; refusing truncation")
            prepared.append((split, row["question_id"], prompt, text, count))
    responses = []
    path = output / "responses.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for split, qid, prompt, text, count in prepared:
            encoded = tokenize_text(tokenizer, text, return_tensors="pt", add_special_tokens=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            for draw in range(config["sampled_responses"] + 1):
                seed = sample_seed(qid, draw + 1, config["seed"])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                options = {"do_sample": draw > 0, "num_beams": 1, "use_cache": True,
                           "max_new_tokens": config["max_new_tokens"],
                           "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
                if draw:
                    options.update(temperature=config["temperature"], top_p=config["top_p"], top_k=0)
                started = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**encoded, **options)
                new_ids = ids[0, encoded["input_ids"].shape[-1]:]
                raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                row = {"model_key": args.model, "split": split, "question_id": qid, "prompt": prompt,
                       "draw": draw, "seed": seed, "decoding": "sampling" if draw else "greedy",
                       "raw_response": raw, "response_id": response_id(args.model, qid, draw, raw),
                       "prompt_tokens": count, "output_tokens": int(new_ids.numel()),
                       "generation_seconds": time.perf_counter() - started}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                responses.append(row)
                del ids, new_ids
            print(f"{args.model} {split} {qid}: saved {len(responses)} responses", flush=True)
    state["responses_sha256"] = digest(path)
    state["response_count"] = len(responses)
    if args.smoke_test:
        for split in panels:
            valid = 0
            for row in responses:
                if row["split"] != split:
                    continue
                try:
                    strict_answers(row["raw_response"])
                    valid += 1
                except (ValueError, TypeError):
                    pass
            if not valid:
                raise ValueError(f"No complete valid {split} expansion response in generation smoke")


def train(args, config, source, output, state):
    preferences = Path(args.preferences).resolve()
    manifest, splits = validate_pairs(preferences, config)
    from matched_8b_provenance import validate_provenance
    recorded = manifest["banks"][args.model]["manifest"]["source"]
    validate_provenance({"expansion": recorded}, {"expansion": source})
    state.update(preferences_manifest_sha256=digest(preferences / "manifest.json"),
                 preference_pair_counts=manifest["pair_counts"],
                 reference="precomputed initial expansion SFT completion log probabilities",
                 checkpoint_selection=config["checkpoint_selection"])
    write(output / "manifest.json", state)
    sys.path.insert(0, str(ROOT))
    from src.utility.training_execution import prepare_execution, activate_execution
    prepare_execution(source["execution_mode"])
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    import unsloth
    import torch
    activate_execution(source["execution_mode"], torch)
    from src.utility.text_tokenizer import text_only_tokenizer
    from src.utility.adapter_save import save_adapter_and_tokenizer
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    loader = unsloth.FastModel if source["model_loader"] == "fast_model" else unsloth.FastLanguageModel
    torch.manual_seed(config["seed"])
    load_options = {"model_name": source["adapter"], "max_seq_length": config["max_seq_length"],
                    "dtype": None, "load_in_4bit": True, "local_files_only": True}
    if source["model_loader"] == "fast_model":
        # FastModel's compile wrapper can otherwise suppress vocabulary logits.
        load_options["return_logits"] = True
    model, tokenizer = loader.from_pretrained(**load_options)
    tokenizer = text_only_tokenizer(tokenizer)
    if set(getattr(model, "peft_config", {})) != {"default"}:
        raise ValueError("Expected the existing expansion SFT adapter named default")
    loader.for_training(model, use_gradient_checkpointing="unsloth")
    model.set_adapter("default")
    trainable = []
    for name, parameter in model.named_parameters():
        active = "lora_" in name and ".default." in name
        parameter.requires_grad_(active)
        if active:
            trainable.append(name)
    if not trainable or any(any(s in n.lower() for s in ("vision", "visual", "projector")) for n in trainable):
        raise ValueError("Expected trainable language-only SFT LoRA parameters")
    # Standard DPO uses deterministic policy and reference likelihoods.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.config.use_cache = False
    configure_job_local_compiler_cache()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = {s: [render_preference(tokenizer, r, source["chat_template_kwargs"], config["max_seq_length"])
                     for r in rows] for s, rows in splits.items()}
    if args.smoke_test:
        # Exercise the longest selected pairs to catch GPU memory failures early.
        tokenized = {s: sorted(rows, key=lambda r: max(len(r[k + "_input_ids"]) for k in ("chosen", "rejected")),
                               reverse=True)[:4] for s, rows in tokenized.items()}
    collator = lambda rows: collate_pairs(rows, tokenizer.pad_token_id, torch)
    device = next(model.parameters()).device
    reference_rows = []
    model.eval()
    for split, rows in tokenized.items():
        for row in rows:
            batch = {k: v.to(device) for k, v in collator([row]).items()}
            for side in ("chosen", "rejected"):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=batch[side + "_input_ids"],
                                   attention_mask=batch[side + "_attention_mask"], use_cache=False).logits
                    value = completion_logps(logits, batch[side + "_labels"], torch).item()
                if not math.isfinite(value):
                    raise ValueError("Non-finite frozen reference log probability")
                row["ref_" + side] = value
                del logits
            reference_rows.append({"split": split, "pair_ids": row["pair_ids"],
                                   "ref_chosen": row["ref_chosen"], "ref_rejected": row["ref_rejected"]})
    write_jsonl(output / "reference_logps.jsonl", reference_rows)
    state["reference_logps_sha256"] = digest(output / "reference_logps.jsonl")
    state["phase"] = "train"
    write(output / "manifest.json", state)

    class ExpansionDPOTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            values = {}
            for side in ("chosen", "rejected"):
                logits = model(input_ids=inputs[side + "_input_ids"],
                               attention_mask=inputs[side + "_attention_mask"], use_cache=False).logits
                values[side] = completion_logps(logits, inputs[side + "_labels"], torch)
            loss = dpo_loss(values["chosen"], values["rejected"], inputs["ref_chosen"],
                            inputs["ref_rejected"], config["beta"], torch)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite DPO loss")
            return (loss, {"margin": (values["chosen"] - values["rejected"]).detach()}) if return_outputs else loss

    arguments = TrainingArguments(output_dir=str(output / "trainer_output"),
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke_test else config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"], max_steps=2 if args.smoke_test else -1,
        learning_rate=config["learning_rate"], warmup_steps=0 if args.smoke_test else 10,
        weight_decay=0.01, fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_8bit",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, report_to="none", logging_steps=1, seed=config["seed"],
        prediction_loss_only=True, label_names=["chosen_labels", "rejected_labels"])
    dataset = {s: Dataset.from_list([{k: v for k, v in r.items() if k not in ("question_id", "pair_ids")}
                                    for r in rows]) for s, rows in tokenized.items()}
    trainer = ExpansionDPOTrainer(model=model, args=arguments, train_dataset=dataset["train"],
                                   eval_dataset=dataset["validation"], data_collator=collator)
    model.train()
    trainer.train()
    state["phase"] = "save_adapter"
    write(output / "manifest.json", state)
    save_adapter_and_tokenizer(model, tokenizer, output / "adapter", save_dtype="float32")
    state.update(best_validation_loss=trainer.state.best_metric,
                 best_checkpoint=trainer.state.best_model_checkpoint,
                 global_steps=trainer.state.global_step,
                 trained_pair_counts={s: len(v) for s, v in tokenized.items()},
                 adapter_files_sha256={p.name: digest(p) for p in (output / "adapter").iterdir() if p.is_file()})
    write(output / "training_log.json", trainer.state.log_history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "generate", "train"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = read(DEFAULT_CONFIG)
    source = source_for(config, args.model)
    split_rows(config, args.full_dataset)
    if args.preferences:
        manifest, _ = validate_pairs(args.preferences.resolve(), config)
        from matched_8b_provenance import validate_provenance
        validate_provenance({"expansion": manifest["banks"][args.model]["manifest"]["source"]}, {"expansion": source})
    if args.mode == "validate":
        return
    if args.mode == "train" and not args.preferences:
        parser.error("Training requires --preferences with reviewed, validated pairs")
    name = args.run_name or f"{args.model}-{args.mode}-{os.environ.get('PBS_JOBID', 'manual')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a directory name")
    output = ROOT / config["output_root"] / name
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "phase": args.mode, "model_key": args.model,
             "source": source, "full_dataset": args.full_dataset, "smoke_test": args.smoke_test,
             "dpo_config_sha256": digest(DEFAULT_CONFIG), "gold_blind_generation": True,
             "package_versions": {n: version(n) for n in ("torch", "transformers", "peft", "unsloth")}}
    write(output / "manifest.json", state)
    configure_job_local_compiler_cache()
    try:
        (generate if args.mode == "generate" else train)(args, config, source, output, state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    print(f"Complete: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
