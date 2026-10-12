"""Check capped SFT preserves evidence, accepted primary labels and experiment controls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import prepare_matched_8b_sft as data
from run_expansion_sft_qwen3 import validate_rows


def example(count):
    prompt = (ROOT / "gadi_sft_8b_starter/prompts/equivalent_expansion_v1.txt").read_text(encoding="utf-8")
    return {"question_id": "q", "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"question": "Which?", "snippets": [{"id": "s1", "text": "evidence"}]})},
        {"role": "assistant", "content": json.dumps({"answers": [{"answer": f"alias{i}",
            "relation_type": "original" if i == 0 else "synonym"} for i in range(count)]})}]}


@pytest.mark.parametrize("count", [1, 4, 5, 6, 10])
def test_transform_preserves_inputs_primary_order_and_source(count):
    original = example(count)
    source = copy.deepcopy(original)
    transformed = data.expansion_rows([original], {"expansion_candidate_limit": 5})[0]
    assert original == source
    assert transformed["question_id"] == source["question_id"]
    assert transformed["messages"][1] == source["messages"][1]
    assert "up to FIVE" in transformed["messages"][0]["content"]
    a = json.loads(source["messages"][-1]["content"])["answers"]
    b = json.loads(transformed["messages"][-1]["content"])["answers"]
    assert b == a[:5] and b[0] == a[0]
    validate_rows([transformed], "train")
    assert data.expansion_rows([original], {})[0] == source


def test_max5_configuration_changes_only_budget_and_dataset_location():
    directory = ROOT / "gadi_sft_8b_starter/configs"
    original = data.read(directory / "matched_8b_sft.json")
    capped = data.read(directory / "matched_8b_sft_max5.json")
    for key in ("source_dataset", "source_manifest_sha256", "outer_dev", "single_answer_prompt", "common", "models"):
        assert capped[key] == original[key]
    assert capped["output_dataset"] != original["output_dataset"]
    assert capped["expansion_candidate_limit"] == 5
    selection = data.read(directory / "sft_max5_selection_8b.json")
    assert selection["selection"]["max_candidates"] == 5
    assert selection["selection"]["max_seq_length"] == 8192
    assert selection["output_root"] == "outputs/sft_max5_8b"
    for model in original["models"]:
        config = data.resolved_config(capped, model, "expansion")
        assert config["max_candidates"] == 5
        assert Path(config["train_input"]).parts[:2] == ("data", "matched_8b_sft_max5")
        assert "max_candidates" not in data.resolved_config(original, model, "expansion")


def test_actual_derived_dataset_keeps_all_questions_and_gold_primary():
    root = ROOT / "data/BioASQ_expansion_sft/seed3407_teacher_strict_v1"
    if not root.exists():
        pytest.skip("Pinned teacher dataset not present")
    audit = {r["question_id"]: r for r in data.records(root / "audit.jsonl")}
    for split, count, changed in (("train", 1296, 111), ("validation", 144, 12)):
        rows = data.records(root / f"{split}.jsonl")
        converted = data.expansion_rows(rows, {"expansion_candidate_limit": 5})
        assert len(converted) == count
        assert sum(len(json.loads(r["messages"][-1]["content"])["answers"]) > 5 for r in rows) == changed
        for before, after in zip(rows, converted):
            assert before["question_id"] == after["question_id"] and before["messages"][1] == after["messages"][1]
            answers = json.loads(after["messages"][-1]["content"])["answers"]
            assert 1 <= len(answers) <= 5
            assert answers[0]["answer"] in audit[after["question_id"]]["gold_aliases"]


@pytest.mark.parametrize("root_first", [False, True])
def test_max5_training_startup_resolves_bundle_helpers_in_fresh_process(root_first):
    # A fresh process catches sys.path changes that a shared pytest module cache
    # can hide. Stop just before the expensive CUDA stack is imported.
    code = r'''
import builtins, json, sys
from pathlib import Path
root = Path(sys.argv[1])
bundle = root / "gadi_sft_8b_starter"
paths = [str(root), str(bundle), str(bundle / "scripts")] if sys.argv[2] == "1" else [str(bundle / "scripts"), str(bundle), str(root)]
sys.path[:0] = paths
from prepare_matched_8b_sft import expansion_rows
prompt = (bundle / "prompts/equivalent_expansion_v1.txt").read_text()
row = {"question_id": "q", "messages": [
    {"role": "system", "content": prompt},
    {"role": "user", "content": '{"question":"Which?","snippets":[]}'},
    {"role": "assistant", "content": '{"answers":[{"answer":"A","relation_type":"original"}]}'}]}
before = sys.path[:]
expansion_rows([row], {"expansion_candidate_limit": 5})
assert sys.path == before, "Max-five data transformation changed the Python import path"
from run_expansion_sft_qwen3 import train
original_import = builtins.__import__
class ReachedCUDAImport(Exception): pass
def guarded_import(name, *args, **kwargs):
    if name == "unsloth":
        raise ReachedCUDAImport()
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
try:
    train(None, {"execution_mode": "default"}, [], [], {})
except ReachedCUDAImport:
    from src.utility import training_execution
    assert Path(training_execution.__file__).resolve() == bundle / "src/utility/training_execution.py"
    print("Bundled startup helper resolved; CUDA not loaded")
else:
    raise AssertionError("Expected to reach CUDA import")
'''
    env = {**os.environ, "UNSLOTH_COMPILE_DISABLE": "0"}
    result = subprocess.run([sys.executable, "-c", code, str(ROOT), str(int(root_first))],
                            capture_output=True, text=True, cwd=ROOT, env=env)
    assert result.returncode == 0, result.stderr
