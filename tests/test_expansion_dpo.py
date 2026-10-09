import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gadi_sft_8b_starter/scripts"))
import expansion_dpo_data as data
from run_expansion_dpo_8b import (collate_pairs, completion_logps, dpo_loss,
                                   render_preference)


def response(answers, classes=None, supported=None, equivalent=None, model="llama31", draw=0):
    raw = json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                  for i, a in enumerate(answers)]})
    row = {"raw_response": raw, "prompt": [{"role": "user", "content": "same question/evidence"}],
           "split": "train", "question_id": "q1", "model_key": model, "draw": draw,
           "response_id": data.response_id(model, "q1", draw, raw)}
    row = data.annotation_template(row)
    row.update(reviewer="human audit", review_notes="Scope and snippets checked")
    for i, label in enumerate(row["candidates"]):
        label.update(**{"class": (classes or ["C3"] * len(answers))[i]},
                     official_accepted=(classes or ["C3"] * len(answers))[i] == "C3",
                     supported=(supported or [True] * len(answers))[i],
                     equivalent_to_original=(equivalent or [True] * len(answers))[i],
                     evidence="snippet 1; reviewed answer context")
    return row


def test_shorter_supported_response_beats_false_synonym():
    clean = response(["TNF-alpha", "tumour necrosis factor alpha"])
    bad = response(["TNF-alpha", "tumour necrosis factor alpha", "IL-6"],
                   ["C3", "C3", "C1"], [True, True, False], [True, True, False], draw=1)
    pairs, _ = data.build_pairs([clean, bad])
    assert len(pairs["train"]) == 1
    assert pairs["train"][0]["chosen"] == clean["raw_response"]
    assert pairs["train"][0]["rejected"] == bad["raw_response"]


def test_longer_response_can_win_for_valid_coverage():
    short = response(["TNF-alpha"])
    long = response(["TNF-alpha", "tumour necrosis factor alpha"], draw=1)
    pairs, _ = data.build_pairs([short, long])
    assert pairs["train"][0]["chosen"] == long["raw_response"]


def test_c2_is_correct_and_not_a_semantic_negative():
    accepted = response(["TNF-alpha"])
    semantic = response(["TNF-alpha", "tumour necrosis factor alpha"], ["C3", "C2"], draw=1)
    assert data.quality(semantic)["incorrect"] == 0
    assert not data.preferred(data.quality(accepted), data.quality(semantic))


def test_conflicting_coverage_and_error_increase_is_excluded():
    short = response(["TNF-alpha"])
    conflict = response(["TNF-alpha", "full name", "IL-6"], ["C3", "C3", "C1"],
                        [True, True, False], [True, True, False], draw=1)
    assert not data.build_pairs([short, conflict])[0]["train"]


def test_length_and_raw_whitespace_alone_do_not_earn_preferences():
    a = response(["TNF-alpha"])
    b = copy.deepcopy(a)
    b["raw_response"] = json.dumps(json.loads(a["raw_response"]), indent=4)
    b["response_id"] = data.response_id("llama31", "q1", 1, b["raw_response"])
    assert not data.build_pairs([a, b])[0]["train"]


@pytest.mark.parametrize("field", ["class", "supported", "equivalent_to_original", "official_accepted"])
def test_unresolved_labels_never_create_pairs(field):
    row = response(["TNF-alpha"])
    row["candidates"][0][field] = None
    assert data.quality(row) is None


def test_official_mismatch_does_not_become_c1_and_must_match_c3():
    row = response(["TNF-alpha"])
    row["candidates"][0]["official_accepted"] = False
    with pytest.raises(ValueError, match="C3"):
        data.quality(row)


@pytest.mark.parametrize("raw", [
    '{"answers":[{"answer":"x","relation_type":"original"}',
    '{"answers":[{"answer":"x","relation_type":"synonym"}]}',
    '{"answers":[{"answer":"x","relation_type":"original"},{"answer":" X ","relation_type":"synonym"}]}',
    '{"answers":[{"answer":"x","relation_type":"original","explanation":"x"}]}',
    '{"answers":[]}',
])
def test_primary_pairs_do_not_repair_or_deduplicate_responses(raw):
    with pytest.raises(ValueError):
        data.strict_answers(raw)


def test_pair_cap_and_question_separation():
    rows = [response(["TNF-alpha"], draw=0)] + [
        response(["TNF-alpha", f"wrong{i}"], ["C3", "C1"], [True, False], [True, False], draw=i)
        for i in range(1, 5)]
    validation = copy.deepcopy(rows)
    for row in validation:
        row["question_id"] = "validation1"
        row["split"] = "validation"
    pairs, _ = data.build_pairs(rows + validation, cap=2)
    assert len(pairs["train"]) == len(pairs["validation"]) == 2
    assert {r["question_id"] for r in pairs["train"]} == {"q1"}
    assert {r["question_id"] for r in pairs["validation"]} == {"validation1"}


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert kwargs == {"enable_thinking": False}
        return [1, 2] if add_generation_prompt else [1, 2, 3, 4]


def test_native_template_masks_evidence_and_keeps_completion():
    row = {"prompt": [], "chosen": "x", "rejected": "y", "question_id": "q", "chosen_id": "a", "rejected_id": "b"}
    tokenized = render_preference(Tokenizer(), row, {"enable_thinking": False}, 4)
    assert tokenized["chosen_labels"] == [-100, -100, 3, 4]
    with pytest.raises(ValueError, match="refusing truncation"):
        render_preference(Tokenizer(), row, {"enable_thinking": False}, 3)


def test_completion_logps_ignore_prompt_padding_and_backpropagate():
    import torch
    logits = torch.zeros(1, 5, 4, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100]])
    value = completion_logps(logits, labels, torch)
    assert value.item() == pytest.approx(-2 * math.log(4))
    value.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0  # predicting prompt
    assert logits.grad[0, 1:3].abs().sum() > 0
    assert logits.grad[0, 3:].abs().sum() == 0  # padding/unused terminal logits


def test_frozen_reference_and_standard_dpo_gradient_direction():
    import torch
    chosen = torch.tensor([-2.0], requires_grad=True)
    rejected = torch.tensor([-4.0], requires_grad=True)
    ref_chosen, ref_rejected = chosen.detach().clone(), rejected.detach().clone()
    loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, .1, torch)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert chosen.grad.item() < 0 and rejected.grad.item() > 0
    assert not ref_chosen.requires_grad and not ref_rejected.requires_grad
    assert dpo_loss(chosen + 1, rejected, ref_chosen, ref_rejected, .1, torch).item() < loss.item()


def test_suppressed_or_partial_logits_fail_instead_of_training_wrong_loss():
    import torch
    labels = torch.tensor([[-100, 1, 2]])
    for logits in (None, torch.empty(0), torch.zeros(1, 1, 4)):
        with pytest.raises(ValueError, match="full sequence logits"):
            completion_logps(logits, labels, torch)


def test_collator_never_scores_padding():
    import torch
    rows = [{s + "_input_ids": ids for s in ("chosen", "rejected")} for ids in ([1, 2], [1, 2, 3])]
    for row in rows:
        for side in ("chosen", "rejected"):
            row[side + "_labels"] = [-100] + row[side + "_input_ids"][1:]
            row["ref_" + side] = -2.
    batch = collate_pairs(rows, 0, torch)
    assert batch["chosen_labels"].tolist() == [[-100, 2, -100], [-100, 2, 3]]
    assert batch["chosen_attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]


@pytest.fixture
def bank_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"sampled_responses": 4, "max_pairs_per_question": 2}))
    monkeypatch.setattr(data, "DEFAULT_CONFIG", config_path)
    config = data.read(config_path)
    panels = {s: [{"question_id": s + "-q", "messages": [{"role": "user", "content": s},
              {"role": "user", "content": "snippets"}, {"role": "assistant", "content": "GOLD TARGET"}]}]
              for s in ("train", "validation")}
    monkeypatch.setattr(data, "split_rows", lambda config, full: panels)
    banks = {}
    for model in data.MODELS:
        path = tmp_path / model
        path.mkdir()
        rows = []
        for split, panel in panels.items():
            for draw in range(5):
                raw = '{"answers": [{"answer": "x", "relation_type": "original"}]}'
                qid = panel[0]["question_id"]
                rows.append({"split": split, "question_id": qid, "draw": draw, "model_key": model,
                             "raw_response": raw, "prompt": panel[0]["messages"][:2],
                             "response_id": data.response_id(model, qid, draw, raw)})
        data.write_jsonl(path / "responses.jsonl", rows)
        data.write(path / "manifest.json", {"status": "complete", "smoke_test": False,
                   "model_key": model, "full_dataset": False, "dpo_config_sha256": data.digest(config_path),
                   "question_ids": {s: [r["question_id"] for r in p] for s, p in panels.items()},
                   "response_count": len(rows), "responses_sha256": data.digest(path / "responses.jsonl")})
        banks[model] = path
    return config, banks


def test_complete_bank_checks_all_draws_and_keeps_gold_out_of_prompt(bank_fixture):
    config, banks = bank_fixture
    _, rows = data.validate_bank(banks["llama31"], config)
    assert len(rows) == 10
    assert all("GOLD TARGET" not in json.dumps(r["prompt"]) for r in rows)


@pytest.mark.parametrize("mutation", ["dev_question", "missing_draw", "changed_prompt", "smoke"])
def test_bank_refuses_leakage_and_incomplete_outputs(bank_fixture, mutation):
    config, banks = bank_fixture
    path = banks["llama31"]
    rows = data.records(path / "responses.jsonl")
    manifest = data.read(path / "manifest.json")
    if mutation == "dev_question":
        rows[0]["question_id"] = "outer-dev-question"
    elif mutation == "missing_draw":
        rows.pop()
    elif mutation == "changed_prompt":
        rows[0]["prompt"][-1]["content"] = "snippets plus injected gold"
    else:
        manifest["smoke_test"] = True
    data.write_jsonl(path / "responses.jsonl", rows)
    manifest["responses_sha256"] = data.digest(path / "responses.jsonl")
    data.write(path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        data.validate_bank(path, config)
