#!/usr/bin/env python3
"""Generate single answers or ten expansion responses without gold-guided selection."""

import argparse
import gc
import hashlib
import re
import time
from pathlib import Path

from run_extractive_expansion_8b import (
    ROOT, configure_job_local_compiler_cache, file_sha256, input_token_count,
    load_model, parse_equivalent_response, read_json, read_jsonl, render_prompt,
    tokenize_text, write_json, write_jsonl,
)

PARSER_VERSION = "single-tagged-answer-v2-trained-prefix"


def sample_seed(question_id, draw, seed=3407):
    return int.from_bytes(hashlib.sha256(f"{seed}:{question_id}:{draw}".encode()).digest()[:4], "big")


def parse_single_answer(raw):
    """Accept the tagged answer with the exact prefix used in historical SFT.

    build_sharegpt_conversation trained assistant targets as
    'Answer: [BE] ... [EE]'. This prefix carries no answer content.
    """
    raw = re.sub(r"^\s*Answer:\s*", "", raw, count=1, flags=re.IGNORECASE)
    matches = re.findall(r"\[BE\](.*?)\[EE\]", raw, flags=re.DOTALL)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError("Expected exactly one nonempty [BE] answer [EE]")
    if re.sub(r"\[BE\].*?\[EE\]", "", raw, flags=re.DOTALL).strip():
        raise ValueError("Unexpected text outside the single answer")
    return re.sub(r"\s+", " ", matches[0]).strip()


def unique_candidates(samples):
    seen, result = set(), []
    for row in samples:
        answer = row.get("answer")
        if not answer or answer.casefold() in seen:
            continue
        seen.add(answer.casefold())
        result.append({"answer": answer, "relation_type": "sampled_single_answer",
                       "position": len(result) + 1, "raw_position": row["draw"]})
    return result


def expansion_candidates(samples):
    """Flatten draw order then within-response order, keeping first occurrences."""
    seen, result = set(), []
    for sample in samples:
        for row in sample["answers"]:
            key = re.sub(r"\s+", " ", row["answer"].casefold()).strip()
            if key in seen:
                continue
            seen.add(key)
            result.append({**row, "position": len(result) + 1,
                           "draw": sample["draw"], "within_draw_position": row["raw_position"]})
    return result


def generation_options(config):
    if config.get("response_mode") == "single_answer_greedy":
        if config["num_generations"] != 1 or config["temperature"] != 0:
            raise ValueError("Expected one greedy single-answer generation")
        return {"max_new_tokens": config["max_new_tokens"], "do_sample": False,
                "num_beams": 1, "repetition_penalty": 1.0, "use_cache": True}
    if config["num_generations"] != 10 or config["temperature"] <= 0:
        raise ValueError("Expected ten stochastic samples with positive temperature")
    return {"max_new_tokens": config["max_new_tokens"], "do_sample": True,
            "temperature": config["temperature"], "top_p": config["top_p"],
            "top_k": 0, "num_beams": 1, "repetition_penalty": 1.0, "use_cache": True}


def prompt_example(example, config):
    if not config.get("mark_snippets", True):
        return example
    return {**example, "snippets": [{**s, "text": "[BS] " + s["text"] + " [ES]"}
                                    for s in example["snippets"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=160)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    cache = configure_job_local_compiler_cache()
    config = read_json(args.config)
    options = generation_options(config)
    expansion = config.get("response_mode") == "equivalent_sampling"
    examples = read_jsonl(ROOT / config["input"])[:args.limit]
    prompt_path = ROOT / config["prompt"]
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config.update(model_name=args.model_name, question_count=len(examples),
                  input_sha256=file_sha256(ROOT / config["input"]),
                  prompt_sha256=file_sha256(prompt_path), gold_blind_generation=True,
                  local_files_only=True, selection="first five unique answers in draw order",
                  parser_version="equivalent-expansion-sampling-v1" if expansion else PARSER_VERSION)
    if expansion:
        config["selection"] = "first five unique candidates in draw then within-response order"
    write_json(output / "config.json", config)
    write_jsonl(output / "examples.jsonl", examples)
    state = {"status": "running", "expected_questions": len(examples), "completed_questions": 0}
    generations, candidates, invalid_candidates = [], [], []
    write_json(output / "status.json", state)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Submit sampling through the Gadi GPU queue")
    model = tokenizer = None
    try:
        loader = config.get("model_loader", "fast_language_model")
        model, tokenizer = (load_model(args.model_name, config["max_seq_length"], False)
                            if loader == "fast_language_model" else
                            load_model(args.model_name, config["max_seq_length"], False, loader))
        if cache:
            configure_job_local_compiler_cache()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        device = next(model.parameters()).device
        budget = config["max_seq_length"] - config["max_new_tokens"]
        prepared = []
        for example in examples:
            # Historical adapters used markers; fresh matched 8B SFT did not.
            marked = prompt_example(example, config)
            text = render_prompt(tokenizer, prompt, marked, config.get("chat_template_kwargs", {}))
            count = input_token_count(tokenize_text(tokenizer, text, add_special_tokens=True))
            if count > budget:
                raise ValueError(f"{example['question_id']}: all snippets exceed {budget} prompt tokens")
            prepared.append((example, text, count))
        for example, text, count in prepared:
            qid = example["question_id"]
            encoded = tokenize_text(tokenizer, text, return_tensors="pt", add_special_tokens=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            samples = []
            for draw in range(1, config["num_generations"] + 1):
                seed = sample_seed(qid, draw, config["seed"])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                started = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**encoded, **options,
                                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                         eos_token_id=tokenizer.eos_token_id)
                new_ids = ids[0, encoded["input_ids"].shape[-1]:]
                raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                elapsed = time.perf_counter() - started
                answer, error, details = None, None, {}
                try:
                    if expansion:
                        answers, rejected, issues, compliant = parse_equivalent_response(raw)
                        details = {"answers": answers, "candidate_limit_applied": any(
                            r.get("reason") == "unique_candidate_limit_exceeded" for r in rejected),
                                   "schema_compliant": compliant,
                                   "validation_issues": issues, "invalid_candidate_count": len(rejected)}
                        invalid_candidates.extend({"question_id": qid, "draw": draw, **r} for r in rejected)
                        if not answers:
                            error = "No usable expansion candidates"
                    else:
                        answer = parse_single_answer(raw)
                except ValueError as exc:
                    error = str(exc)
                    if expansion:
                        details = {"answers": [], "schema_compliant": False,
                                   "validation_issues": [], "invalid_candidate_count": 0}
                samples.append({"draw": draw, "seed": seed, "raw_response": raw,
                                **({} if expansion else {"answer": answer}), **details, "parse_error": error,
                                "output_tokens": int(new_ids.numel()), "generation_seconds": elapsed})
                del ids, new_ids
            unique = expansion_candidates(samples) if expansion else unique_candidates(samples)
            candidates.extend({"question_id": qid, **row} for row in unique)
            generations.append({"question_id": qid, "question": example["question"],
                                "samples": samples, "parse_error": None if unique else "All samples failed",
                                "prompt_tokens": count, "included_snippets": len(example["snippets"]),
                                "total_snippets": len(example["snippets"]), "snippets_truncated": False,
                                "request_count": len(samples), "input_tokens": count * len(samples),
                                "output_tokens": sum(s["output_tokens"] for s in samples),
                                "generation_seconds": sum(s["generation_seconds"] for s in samples)})
            state["completed_questions"] = len(generations)
            write_jsonl(output / "generations.jsonl", generations)
            write_jsonl(output / "candidates.jsonl", candidates)
            if expansion:
                write_jsonl(output / "invalid_candidates.jsonl", invalid_candidates)
            write_json(output / "status.json", state)
            print(f"{len(generations)}/{len(examples)}: {len(samples)} draws; {len(unique)} unique answers", flush=True)
            del encoded
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_jsonl(output / "generations.jsonl", generations)
        write_jsonl(output / "candidates.jsonl", candidates)
        if expansion:
            write_jsonl(output / "invalid_candidates.jsonl", invalid_candidates)
        write_json(output / "status.json", state)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
