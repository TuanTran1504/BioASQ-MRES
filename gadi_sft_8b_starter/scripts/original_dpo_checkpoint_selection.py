"""Greedy official MRR on internal validation; no development-set selection."""
from contextlib import contextmanager
import json
import hashlib
import math
from pathlib import Path
import shutil
import time

from prepare_matched_8b_sft import ROOT, read, records, digest, write, load_source
from expansion_dpo_data import split_rows
from run_extractive_expansion_8b import tokenize_text, write_jsonl
from run_single_answer_sampling import prepare_prompts, choose_prepared, generation_options, parse_single_answer, sample_seed

JAR = ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"


def validation_examples(config):
    """All 144 audited internal-validation questions, common to both strategies."""
    from prepare_sft_dual_selection import validation_examples as audited_examples
    evaluation = read(ROOT / config["evaluation_config"])
    matrix = read(ROOT / evaluation["training_matrix"])
    source, _, _ = load_source(matrix)
    audit = source / "audit.jsonl"
    if digest(audit) != read(source / "manifest.json")["output_sha256"]["audit.jsonl"]:
        raise ValueError("Validation alias audit changed")
    panels = split_rows(config, True)
    examples = audited_examples(panels["validation"], records(audit))
    ids = {r["question_id"] for r in examples}
    forbidden = {r["question_id"] for r in panels["train"]} | {
        r["question_id"] for r in records(ROOT / evaluation["input"])}
    if len(examples) != 144 or len(ids) != 144 or ids & forbidden:
        raise ValueError("MRR selection requires 144 unique internal-validation questions, disjoint from fitting/dev")
    prompts = {r["messages"][0]["content"] for r in panels["validation"]}
    if len(prompts) != 1:
        raise ValueError("Validation system prompts differ")
    return examples, prompts.pop()


def check_scoring_runtime():
    from src.utility.bioasq_official import _conda_executable, _official_adapter_classes
    import subprocess
    subprocess.run([*_conda_executable("java"), "-version"], check=True)
    _official_adapter_classes(JAR)


def score_answers(examples, generations, destination):
    """Failures score zero; official MRR is binary with one submitted answer."""
    from src.utility.bioasq_official import _run_official_per_question
    expected = {r["question_id"] for r in examples}
    by_id = {r["question_id"]: r for r in generations}
    if len(expected) != len(examples) or len(by_id) != len(generations) or set(by_id) != expected:
        raise ValueError("Every validation question must be retained exactly once")
    gold, predictions = [], []
    for example in examples:
        raw = by_id[example["question_id"]]["raw_response"]
        try:
            parsed = parse_single_answer(raw)
        except ValueError:
            parsed = None
        if parsed != by_id[example["question_id"]]["answer"]:
            raise ValueError("Validation answer differs from the raw tagged parse")
        if parsed is None:
            continue
        common = {"id": example["question_id"], "type": "factoid", "body": example["question"]}
        gold.append({**common, "exact_answer": [example["gold_aliases"]]})
        predictions.append({**common, "exact_answer": [[parsed]]})
    gold_path, predictions_path = destination / "scorer_gold.json", destination / "scorer_predictions.json"
    write(gold_path, {"questions": gold})
    write(predictions_path, {"questions": predictions})
    scored = _run_official_per_question(gold_path=gold_path, prediction_path=predictions_path,
        jar_path=JAR, challenge_version=8) if predictions else []
    scores = {r["question_id"]: r["mrr"] for r in scored}
    if (len(scores) != len(scored) or set(scores) != {r["id"] for r in predictions}
            or any(v not in (0.0, 1.0) for v in scores.values())):
        raise ValueError("Incomplete or non-binary official single-answer MRR")
    rows = [{"question_id": e["question_id"], "mrr": scores.get(e["question_id"], 0.0),
             "parsed": by_id[e["question_id"]]["answer"] is not None} for e in examples]
    metrics = {"mrr": sum(r["mrr"] for r in rows) / len(rows), "question_count": len(rows),
               "parse_success_count": sum(r["parsed"] for r in rows),
               "correct_count": sum(r["mrr"] for r in rows)}
    write(destination / "scores.json", {"metrics": metrics, "questions": rows})
    return metrics


@contextmanager
def generation_mode(model, torch, restore_for_training=None):
    """Restore training, checkpointing, cache and RNG even when scoring fails."""
    was_training, was_cache = model.training, model.config.use_cache
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    trainability = [(p, p.requires_grad) for p in model.parameters()]
    checkpoint_states = [(m, {k: getattr(m, k) for k in ("gradient_checkpointing", "_gradient_checkpointing_func")
                             if hasattr(m, k)}) for m in model.modules()]
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            if checkpointing:
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
            yield
        finally:
            # Unsloth generate() can leave private generation flags and padding
            # changes behind when called on an already-eval model. Its native
            # for_training transition clears them before the next DPO update.
            if restore_for_training is not None:
                restore_for_training()
            elif checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for module, fields in checkpoint_states:
                for key, value in fields.items():
                    setattr(module, key, value)
            for parameter, trainable in trainability:
                parameter.requires_grad_(trainable)
            model.config.use_cache = was_cache
            model.train(was_training)


class EpochMRRTracker:
    def __init__(self, config, source, output, smoke=False, restore_for_training=None):
        self.config, self.source, self.output, self.smoke = config, source, output, smoke
        self.examples, prompt = validation_examples(config)
        if prompt != source["system_prompt"]:
            raise ValueError("MRR validation must use the original SFT prompt")
        self.settings = {"max_seq_length": config["max_seq_length"], "max_new_tokens": config["max_new_tokens"],
            "chat_template_kwargs": source["chat_template_kwargs"], "mark_snippets": False,
            "require_all_snippets": True, "response_mode": "single_answer_greedy", "temperature": 0,
            "num_generations": 1}
        self.prepared, self.rows = None, []
        self.restore_for_training = restore_for_training

    def evaluate(self, model, tokenizer, epoch, step, loss, torch):
        if not math.isfinite(loss) or any(r["global_step"] == step for r in self.rows):
            raise ValueError("Expected one finite preference loss per epoch checkpoint")
        if self.prepared is None:
            all_inputs = prepare_prompts(tokenizer, self.source["system_prompt"], self.examples, self.settings)
            self.prepared = choose_prepared(all_inputs, 4 if self.smoke else 144, self.smoke)
        destination = self.output / "validation_mrr" / f"checkpoint-{step}"
        destination.mkdir(parents=True, exist_ok=False)
        examples = [e for e, _, _ in self.prepared]
        write_jsonl(destination / "examples.jsonl", examples)
        write(destination / "config.json", {**self.settings, "seed": self.config["seed"],
            "selection_panel": "internal validation, common across concept/form strategies",
            "system_prompt_sha256": hashlib.sha256(self.source["system_prompt"].encode()).hexdigest()})
        generated = []
        device = next(model.parameters()).device
        if self.source.get("model_loader") == "fast_model":
            # Training loads the adapter directly, bypassing load_model()'s
            # Ministral generation setup. Reuse the proven inference backend:
            # FastModel's forced static cache can mix Float and Half values.
            from src.utility.ministral_generation import configure_ministral_generation
            configure_ministral_generation(model)
        with generation_mode(model, torch, self.restore_for_training):
            for example, text, count in self.prepared:
                encoded = tokenize_text(tokenizer, text, return_tensors="pt", add_special_tokens=True)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                seed = sample_seed(example["question_id"], 1, self.config["seed"])
                torch.manual_seed(seed)
                started = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**encoded, **generation_options(self.settings),
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                new_ids = ids[0, encoded["input_ids"].shape[-1]:]
                raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                answer, error = None, None
                try:
                    answer = parse_single_answer(raw)
                except ValueError as exc:
                    error = str(exc)
                generated.append({"question_id": example["question_id"], "raw_response": raw,
                    "answer": answer, "parse_error": error, "prompt_tokens": count,
                    "output_tokens": int(new_ids.numel()), "generation_seconds": time.perf_counter() - started})
                del encoded, ids, new_ids
        write_jsonl(destination / "generations.jsonl", generated)
        metrics = score_answers(examples, generated, destination)
        if self.smoke and not metrics["parse_success_count"]:
            raise ValueError("MRR smoke produced no parseable original-SFT answers")
        row = {"epoch": epoch, "global_step": step, "eval_loss": loss, "validation_mrr": metrics["mrr"],
               "metrics": metrics, "scoring": "official BioASQ Java, one answer; failures retained",
               "artifact_path": str(destination), "files_sha256": {
                   p.name: digest(p) for p in destination.iterdir() if p.is_file()}}
        self.rows.append(row)
        write(self.output / "epoch_selection_scores.json", self.rows)
        return metrics["mrr"]


def choose_winners(rows, expected_epochs):
    if (len(rows) != expected_epochs or [r["epoch"] for r in rows] != list(range(1, expected_epochs + 1))
            or len({r["global_step"] for r in rows}) != len(rows)
            or any(not math.isfinite(r["eval_loss"]) or not 0 <= r["validation_mrr"] <= 1 for r in rows)):
        raise ValueError("Retain and score every complete epoch before selection")
    return {"best_eval_loss": min(rows, key=lambda r: (r["eval_loss"], r["epoch"])),
            "best_validation_mrr": max(rows, key=lambda r: (r["validation_mrr"], -r["epoch"]))}


def export_selected_adapters(output, rows, expected_epochs):
    """Copy each winning epoch's weights, using the saved native tokenizer."""
    result = {}
    for name, winner in choose_winners(rows, expected_epochs).items():
        checkpoint = output / "trainer_output" / f"checkpoint-{winner['global_step']}"
        saved = read(checkpoint / "trainer_state.json")
        if saved["global_step"] != winner["global_step"] or saved["epoch"] != winner["epoch"]:
            raise ValueError("Epoch scores do not belong to this saved checkpoint")
        destination = output / "selected_adapters" / name
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / filename).is_file() or not (checkpoint / filename).stat().st_size:
                raise ValueError("Selected checkpoint weights missing")
            shutil.copy2(checkpoint / filename, destination / filename)
        for path in (output / "adapter").iterdir():
            if path.is_file() and path.name not in {"adapter_config.json", "adapter_model.safetensors", "training_complete.json"}:
                shutil.copy2(path, destination / path.name)
        if not (destination / "tokenizer_config.json").is_file():
            raise ValueError("Native tokenizer missing for selected epoch")
        write(destination / "training_complete.json", {"status": "completed", "selection_name": name,
            "checkpoint": winner, "selection_panel": "internal validation; no dev/test checkpoint selection"})
        result[name] = {"path": str(destination), "epoch": winner["epoch"], "global_step": winner["global_step"],
            "eval_loss": winner["eval_loss"], "validation_mrr": winner["validation_mrr"],
            "files_sha256": {p.name: digest(p) for p in destination.iterdir() if p.is_file()}}
    return result
