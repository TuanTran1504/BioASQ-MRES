#!/usr/bin/env python3
"""Check the trained smoke adapter's output format on four validation questions."""

import argparse
import json
from pathlib import Path

from prepare_matched_8b_sft import ROOT, read, records, write
from run_extractive_expansion_8b import (
    configure_job_local_compiler_cache, input_token_count, load_model,
    parse_equivalent_response, render_prompt, tokenize_text)
from run_single_answer_sampling import parse_single_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    status = read(directory / "status.json")
    if status["status"] != "completed" or not status["smoke_test"]:
        raise ValueError("Expected a completed training smoke adapter")
    config = status["configuration"]
    configure_job_local_compiler_cache()
    model, tokenizer = load_model(str(directory / "adapter"), 8192, False, config["model_loader"])
    import torch
    rows = records(ROOT / config["eval_input"])
    prepared = []
    for row in rows:
        user = json.loads(row["messages"][1]["content"])
        example = {"question": user["question"], "snippets": [
            {"snippet_id": snippet["id"], "text": snippet["text"]} for snippet in user["snippets"]]}
        text = render_prompt(tokenizer, row["messages"][0]["content"], example, config["chat_template_kwargs"])
        count = input_token_count(tokenize_text(tokenizer, text, add_special_tokens=True))
        if count > 8192 - 512:
            raise ValueError(f"{row['question_id']}: smoke generation would truncate evidence")
        prepared.append((count, row["question_id"], text))
    results = []
    for _, qid, text in sorted(prepared, reverse=True)[:4]:
        encoded = tokenize_text(tokenizer, text, return_tensors="pt", add_special_tokens=True)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=512, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        try:
            if config["formulation"] == "original":
                answers = [parse_single_answer(raw)]
                compliant = True
            else:
                accepted, _, _, compliant = parse_equivalent_response(raw)
                answers = [item["answer"] for item in accepted]
            results.append({"question_id": qid, "raw_response": raw, "answers": answers,
                            "schema_compliant": compliant})
        except ValueError as exc:
            results.append({"question_id": qid, "raw_response": raw, "answers": [],
                            "schema_compliant": False, "parse_error": str(exc)})
    usable = sum(bool(row["answers"]) for row in results)
    write(directory / "generation_smoke.json", {"questions": results, "usable_questions": usable,
                                                "status": "passed" if usable else "failed"})
    if not usable:
        raise RuntimeError("Trained smoke adapter produced zero parseable answers; full training remains held")
    print(f"Generation smoke passed: {usable}/4 validation questions produced answers")


if __name__ == "__main__":
    main()
