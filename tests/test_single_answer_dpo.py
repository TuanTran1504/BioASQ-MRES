import copy
import json
from pathlib import Path
import sys

import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import single_answer_dpo_data as data
import run_expansion_dpo_8b as runner
from run_single_answer_sampling import sample_seed


def response(answer="TNF-alpha", label="C3", draw=0, model="llama31", split="train", qid="q"):
    raw = f"Answer: [BE]{answer}[EE]"
    prompt = [{"role": "system", "content": "One answer"}, {"role": "user", "content": json.dumps({
        "question": "Which cytokine?", "snippets": [{"id": "s1", "text": "TNF-alpha is the implicated cytokine."}]})}]
    row = {"question_id": qid, "model_key": model, "split": split, "draw": draw, "raw_response": raw,
           "response_id": data.response_id(model, qid, draw, raw), "prompt": prompt,
           "seed": sample_seed(qid, draw + 1), "decoding": "sampling" if draw else "greedy"}
    output = data.template(row)
    output.update(**{"class": label}, official_accepted=label == "C3", supported=label != "C1", confidence="high",
        reviewer="test annotator", review_notes="Question and evidence checked", evidence=[] if label == "C1" else [
            {"snippet_id": "s1", "quote": "TNF-alpha is the implicated cytokine."}])
    return output


def test_class_strategies_are_separate_and_mixed_balanced():
    a = response()
    b = response("IL-6", "C1", 1)
    c = response("tumour necrosis factor alpha", "C2", 2, "qwen3")
    pairs, _ = data.build_pairs([a, b, c])
    assert pairs["concept"]["train"][0]["rejected_class"] == "C1"
    assert pairs["form"]["train"][0]["rejected_class"] == "C2"
    assert {p["rejected_class"] for p in pairs["mixed"]["train"]} == {"C1", "C2"}
    assert all(p["chosen"] == a["raw_response"] for p in pairs["mixed"]["train"])
    assert all(p["prompt"] == a["prompt"] for p in pairs["mixed"]["train"])


def test_no_exact_mismatch_is_automatically_c1():
    row = response("full name", "C2", 1)
    row["class"] = None
    assert not data.build_pairs([response(), row])[0]["concept"]["train"]


@pytest.mark.parametrize("field,value", [("supported", False), ("confidence", "low"), ("evidence", []), ("reviewer", "")])
def test_unreliable_winner_or_c2_is_not_used(field, value):
    a = response()
    a[field] = value
    assert not data.build_pairs([a, response("IL-6", "C1", 1)])[0]["mixed"]["train"]


def test_format_duplicates_do_not_create_preferences_and_partitions_stay_separate():
    a = response()
    duplicate = copy.deepcopy(a)
    duplicate.update(raw_response="Answer: [BE] TNF-alpha [EE]", draw=3)
    duplicate["response_id"] = data.response_id("llama31", "q", 3, duplicate["raw_response"])
    b = response("IL-6", "C1", 1)
    val = response(split="validation", qid="v")
    pairs, _ = data.build_pairs([a, duplicate, b, val])
    assert len(pairs["mixed"]["train"]) == 1 and not pairs["mixed"]["validation"]
    assert data.template({**a, "raw_response": "untagged answer"})["excluded_reason"]


def test_official_class_conflict_or_changed_answer_rejected():
    row = response()
    row["official_accepted"] = False
    with pytest.raises(ValueError, match="C3"):
        data.reliable(row)
    row = response()
    row["answer"] = "IL-6"
    with pytest.raises(ValueError, match="changed"):
        data.reliable(row)


def test_bank_rejects_missing_draws_and_dev_question(monkeypatch):
    config = data.read(data.DEFAULT_CONFIG)
    rows = [response(draw=i) for i in range(config["sampled_responses"] + 1)]
    panel = {"train": [{"question_id": "q", "messages": rows[0]["prompt"] + [{"role": "assistant", "content": "target"}]}], "validation": []}
    monkeypatch.setattr(data, "split_rows", lambda *a: panel)
    manifest = {"status": "complete", "smoke_test": False, "formulation": "original",
        "dpo_config_sha256": data.digest(data.DEFAULT_CONFIG), "full_dataset": True, "model_key": "llama31",
        "question_ids": {"train": ["q"], "validation": []}, "response_count": len(rows)}
    data.validate_responses(manifest, rows, config)
    with pytest.raises(ValueError, match="Missing"):
        data.validate_responses(manifest, rows[:-1], config)
    changed = copy.deepcopy(rows)
    changed[0]["question_id"] = "dev-question"
    with pytest.raises(ValueError, match="partition"):
        data.validate_responses(manifest, changed, config)


def test_original_source_and_shorter_warmup(monkeypatch):
    monkeypatch.setattr(runner, "read", lambda p: {})
    monkeypatch.setattr(runner, "validate_training_pair", lambda *a: {"original": {"adapter": "original"}, "expansion": {"adapter": "expansion"}})
    assert runner.source_for({"evaluation_config": "config", "formulation": "original"}, "qwen3")["adapter"] == "original"
    config = {"num_train_epochs": 2, "gradient_accumulation_steps": 16, "warmup_ratio": 0.05, "minimum_optimizer_steps": 10}
    steps, warmup = runner.training_schedule(config, 159)
    assert steps == 20 and warmup == 1 and steps - warmup == 19
    assert runner.training_schedule(config, 1296) == (162, 9)
    with pytest.raises(ValueError, match="not enough"):
        runner.training_schedule(config, 16)
    assert runner.training_schedule(config, 4, True) == (2, 0)


def test_pair_package_detects_annotation_and_pair_tampering(tmp_path, monkeypatch):
    config = data.read(data.DEFAULT_CONFIG)
    raw, annotated, banks = [], [], {}
    for model in data.MODELS:
        rows = [response(draw=0, model=model), response("IL-6", "C1", 1, model),
                response(model=model, split="validation", qid="v"), response("IL-6", "C1", 1, model, "validation", "v")]
        annotated.extend(rows)
        raw.extend({k: row[k] for k in ("question_id", "model_key", "split", "draw", "raw_response", "response_id", "prompt", "seed", "decoding")} for row in rows)
        banks[model] = {"manifest": {"source": {"adapter": model}}}
    monkeypatch.setattr(data, "validate_responses", lambda *a: None)
    pairs, _ = data.build_pairs(annotated)
    from run_extractive_expansion_8b import write_jsonl
    write_jsonl(tmp_path / "responses.jsonl", raw)
    write_jsonl(tmp_path / "annotations.jsonl", annotated)
    for split, values in pairs["concept"].items():
        write_jsonl(tmp_path / f"concept-{split}.jsonl", values)
    files = ["responses.jsonl", "annotations.jsonl", "concept-train.jsonl", "concept-validation.jsonl"]
    data.write(tmp_path / "manifest.json", {"status": "ready", "banks": banks,
        "dpo_config_sha256": data.digest(data.DEFAULT_CONFIG), "files_sha256": {f: data.digest(tmp_path / f) for f in files}})
    _, verified = data.validate_pairs(tmp_path, config, "concept")
    assert len(verified["train"]) == 1 and len(verified["validation"]) == 1
    (tmp_path / "concept-train.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="Pairs"):
        data.validate_pairs(tmp_path, config, "concept")
