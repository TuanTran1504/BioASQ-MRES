from __future__ import annotations

import json
import hashlib
import contextlib
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.notebook_workflows import available_methods, defaults, execute, plan
from src.notebook_workflows.operations import compare_banks, curriculum_split, occurrence_export
from src.notebook_workflows.presets import PRESETS, configuration
from src.notebook_workflows.runner import ROOT


@pytest.mark.parametrize("category,preset", [
    (category, preset) for category, presets in PRESETS.items() for preset in presets
])
def test_all_presets_can_be_previewed_without_artifacts(tmp_path, category, preset):
    method, params = configuration(category, preset)
    preview = plan(method, params, project_root=tmp_path)
    assert preview["method"] == method
    assert not (tmp_path / "Artifacts").exists()


def test_preview_does_not_launch_or_write(tmp_path, capsys):
    preview = plan("evidence_coverage", project_root=tmp_path)
    with patch("subprocess.Popen", side_effect=AssertionError("Must not execute")):
        execute(preview)
    assert not (tmp_path / "Artifacts").exists()
    assert "evidence_coverage" in capsys.readouterr().out


@pytest.mark.parametrize("method,params", [
    ("answer_sft", {"learning_raet": .01}),
    ("answer_sft", {"output_dir": "elsewhere"}),
    ("standard_dpo", {"split_by": "pair"}),
    ("sample_candidates", {"model_ref": "model-must-be-in-a-list"}),
    ("snippet_split", {"dev_ratio": 1.0}),
    ("sample_candidates", {"samples_per_question_total": 0}),
    ("staged_dpo", {"objective": "typo"}),
    ("staged_dpo", {"stage_settings": {"format_alignment": {"output_root": "/elsewhere"}}}),
    ("staged_dpo", {"stage_settings": {"format_alignment": {"semantic_judge_enabled": True}}}),
    ("staged_dpo", {"semantic_judge_enabled": True}),
    ("evidence_annotation", {"max_new_calls": None}),
])
def test_rejects_invalid_or_unsafe_configuration(method, params):
    if method == "evidence_annotation":
        # None must not silently request an unlimited API batch.
        with pytest.raises(ValueError):
            plan(method, params)
    else:
        with pytest.raises(ValueError):
            plan(method, params)


def test_api_gate_and_missing_inputs_precede_writes(tmp_path):
    preview = plan("judge_candidates", project_root=tmp_path)
    with pytest.raises(ValueError, match="ALLOW_API"):
        execute(preview, run=True)
    with pytest.raises(ValueError, match="Set"):
        execute(preview, run=True, allow_api=True)
    assert not (tmp_path / "Artifacts").exists()


def test_existing_output_is_never_reused(tmp_path):
    source = tmp_path / "gold.json"
    source.write_text('{"questions":[]}', encoding="utf-8")
    preview = plan("evidence_coverage", {"sources": [str(source)]}, project_root=tmp_path)
    Path(preview["run_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        execute(preview, run=True)


def test_mutated_plan_is_rejected_before_execution(tmp_path):
    preview = plan("evidence_coverage", project_root=tmp_path)
    preview["run_dir"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="Plan changed"):
        execute(preview, run=True)


def test_aliases_cannot_leak_between_sft_train_and_dev(tmp_path):
    train = tmp_path / "train.json"
    dev = tmp_path / "dev.json"
    train.write_text(json.dumps([{"id": "q__supported_alias_0", "source_question_id": "q"}]))
    dev.write_text(json.dumps([{"id": "q"}]))
    with pytest.raises(ValueError, match="same original"):
        plan("answer_sft", {"train_input": [str(train)], "eval_input": [str(dev)],
                            "prompt_file": str(ROOT / "prompts/factoid_single_answer_aligned.json")},
             project_root=tmp_path)


def test_prepared_split_rejects_alias_expanded_input(tmp_path):
    source = tmp_path / "aliases.json"
    source.write_text(json.dumps([{"id": f"q__supported_alias_{i}"} for i in range(2)]))
    with pytest.raises(ValueError, match="alias-expanded"):
        plan("prepared_split", {"input": str(source)}, project_root=tmp_path)


def test_configuration_copies_are_independent():
    first = defaults("snippet_split")
    first["test_input"].clear()
    assert len(defaults("snippet_split")["test_input"]) == 4


def test_curriculum_routes_every_supported_pair_direction(tmp_path):
    source = tmp_path / "pairs.jsonl"
    rows = [{"pair_id": f"p{i}", "question_id": f"q{i}", "chosen_class": c, "rejected_class": r}
            for i, (c, r) in enumerate([("C3", "C1"), ("C3", "C2"), ("C2", "C1")])]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = curriculum_split({"pair_file": str(source)}, tmp_path / "out", ROOT)
    assert {name: info["pair_count"] for name, info in summary["curriculum"].items()} == {
        "concept_learning": 1, "format_alignment": 1, "hierarchical_ranking": 1,
    }
    source.write_text(json.dumps({"chosen_class": "C1", "rejected_class": "C3"}) + "\n")
    with pytest.raises(ValueError, match="Unexpected class"):
        curriculum_split({"pair_file": str(source)}, tmp_path / "other", ROOT)


def test_occurrence_export_preserves_context_and_matching_arms(tmp_path):
    source = tmp_path / "prepared.json"
    source.write_text(json.dumps([{
        "id": "q", "type": "factoid", "instruction": "Answer", "input_1": "Which?",
        "input_2": "PubMed ID: 123\n[BS]The answer is alpha.[ES]",
        "output": "[BE]alpha[EE][BE]absent[EE]",
    }]))
    out = tmp_path / "run"
    out.mkdir()
    summary = occurrence_export({"source": str(source), "match_mode": "normalized", "all_aliases": True}, out, ROOT)
    evidence = [json.loads(line) for line in (out / "export/evidence_sft.jsonl").read_text().splitlines()]
    answer = [json.loads(line) for line in (out / "export/answer_only_sft.jsonl").read_text().splitlines()]
    assert len(evidence) == len(answer) == 1
    assert evidence[0]["messages"][1] == answer[0]["messages"][1]
    assert "alpha" in evidence[0]["messages"][-1]["content"]
    assert "Evidence: [1]" in evidence[0]["messages"][-1]["content"]


def test_bank_comparison_aligns_question_ids(tmp_path):
    banks = {}
    for label, answer in [("a", "alpha"), ("b", "beta")]:
        path = tmp_path / f"{label}.jsonl"
        path.write_text(json.dumps({"question_id": "q", "raw_output": f"[BE]{answer}[EE]"}) + "\n")
        banks[label] = str(path)
    result = compare_banks({"banks": banks}, tmp_path / "comparison", ROOT)
    assert result["different_question_count"] == 1
    assert result["banks"]["a"]["questions"] == 1


def test_changed_input_requires_fresh_preview(tmp_path):
    source = tmp_path / "gold.json"
    source.write_text('{"questions":[]}')
    preview = plan("evidence_coverage", {"sources": [str(source)]}, project_root=tmp_path)
    source.write_text('{"questions":[], "changed":true}')
    with pytest.raises(ValueError, match="Input changed"):
        execute(preview, run=True)
    assert not Path(preview["run_dir"]).exists()


def test_generated_notebooks_execute_only_previews():
    from scripts.build_workflow_notebooks import WORKFLOWS, build
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == len(WORKFLOWS) == 8
    # Saved notebooks contain user-edited RUN flags and optional display cells.
    # Exercise the shipped defaults without executing users' experiment code.
    with patch("subprocess.Popen", side_effect=AssertionError("Preview started a process")):
        for name, category, title, preset in WORKFLOWS:
            ns = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(build(category, title, preset)["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"{name}:{index}", "exec"), ns)
            assert ns["RUN"] is False and ns["ALLOW_API"] is False and not ns["RESULTS"]


def test_archive_retains_every_original_byte():
    manifest = json.loads((ROOT / "reproducibility/notebook_archive_manifest.json").read_text(encoding="utf-8"))
    archive = ROOT / manifest["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == manifest["notebook_count"] == 42
        for row in manifest["entries"]:
            assert hashlib.sha256(z.read(row["original_path"])).hexdigest() == row["sha256"]


def test_judge_sample_count_and_provenance_are_configurable(tmp_path):
    from cse_dpo.annotate_split_dpo_candidate_bank import prepare_records
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q", "input_1": "Which?", "output": "[BE]alpha[EE]"}]))
    bank = tmp_path / "bank.jsonl"
    bank.write_text("".join(json.dumps({
        "question_id": "q", "response_id": f"r{i}", "sample_id": i,
        "parser_status": "ok", "parsed_items": ["alpha"], "generator_checkpoint": "qwen3b",
        "evidence": ["PubMed ID: 123\n[BS]alpha is the answer.[ES]"], "prompt": "question",
    }) + "\n" for i in range(2)))
    records, summary = prepare_records(bank, questions, expected_samples=2)
    assert summary["samples_per_question"] == 2
    assert {row["source_model"] for row in records} == {"qwen3b"}
    with pytest.raises(ValueError, match="exactly 10"):
        prepare_records(bank, questions)


def test_dev_occurrences_use_explicit_split_and_remain_semantically_unreviewed(tmp_path):
    from src.notebook_workflows.operations import dev_evidence
    dev = tmp_path / "dev.json"
    train = tmp_path / "train.json"
    row = {"id": "dev1", "input_1": "Which?", "output": "[BE]alpha[EE]",
           "input_2": "PubMed ID: 123\n[BS]ALPHA is the answer.[ES]"}
    dev.write_text(json.dumps([row]), encoding="utf-8")
    train.write_text(json.dumps([{**row, "id": "train1"}]), encoding="utf-8")
    params = {**defaults("dev_evidence"), "source": str(dev), "train_source": str(train)}
    preview = plan("dev_evidence", params, project_root=tmp_path)
    assert not preview["api"] and not preview["missing"]
    result = dev_evidence(params, tmp_path / "out", ROOT)
    assert result["normalized_match_pairs"] == 1
    assert result["literal_match_pairs"] == 0
    assert result["semantic_alias_annotations_complete"] is False
    assert result["source_manifest"]["train_dev_disjoint"] is True
    train.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps training"):
        plan("dev_evidence", params, project_root=tmp_path)


def test_rationale_sft_accepts_custom_question_counts_and_keeps_replay_disjoint(tmp_path, monkeypatch):
    from cse_dpo import run_factoid_rationale_sft_dpo_comparison as driver

    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
            text = "".join(m["role"] + ":" + m["content"] + "|" for m in messages)
            if add_generation_prompt:
                text += "assistant:"
            return list(text.encode("utf-8"))

    train = tmp_path / "train.json"
    dev = tmp_path / "dev.json"
    bank = tmp_path / "rationales.jsonl"
    out = tmp_path / "prepared"
    train.write_text(json.dumps([
        {"id": f"q{i}", "input_1": "Which?", "output": "[BE]alpha[EE]",
         "input_2": "PubMed ID: 123\n[BS]alpha is the answer.[ES]"} for i in range(4)
    ]), encoding="utf-8")
    dev.write_text(json.dumps([{"id": "heldout"}]), encoding="utf-8")
    bank.write_text("".join(json.dumps({
        "status": "accepted", "question_id": f"q{i}", "evidence_ids": ["1.1"],
        "chosen_answer": "alpha", "reason": "The snippet explicitly names alpha.",
    }) + "\n" for i in range(4)), encoding="utf-8")
    for key, value in {
        "SOURCE_TRAIN": train, "DEV_INPUT": dev, "RATIONALE_BANK": bank,
        "SFT_DATA_DIR": out, "SFT_MIXED_FILE": out / "mixed_train.jsonl",
        "ANSWER_ONLY_REPLAY_FRACTION": .2, "MAX_LENGTH": 4096,
    }.items():
        monkeypatch.setattr(driver, key, value)
    tokenizer, prompt = Tokenizer(), driver.load_prompt_spec()
    with pytest.raises(ValueError, match="Expected 1111"):
        driver.build_sft_data(tokenizer, prompt)
    mixed = driver.build_sft_data(tokenizer, prompt, expected_question_count=None)
    assert len(mixed) == 5
    assert sum(row["mode"] == "answer_only_replay" for row in mixed) == 1
    assert {row["question_id"] for row in mixed} == {f"q{i}" for i in range(4)}
    assert driver.build_sft_data(tokenizer, prompt, expected_question_count=None) == mixed
    dev.write_text(json.dumps([{"id": "q0"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        driver.build_sft_data(tokenizer, prompt, expected_question_count=None)
