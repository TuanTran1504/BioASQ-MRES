import ast
from contextlib import contextmanager
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "gadi_sft_8b_starter"


@contextmanager
def bundle_imports():
    before = sys.path[:]
    sys.path.insert(0, str(BUNDLE / "scripts"))
    try:
        yield
    finally:
        sys.path[:] = before


with bundle_imports():
    import prepare_matched_8b_sft as prepare
    from run_expansion_sft_qwen3 import audit_response_masks, validate_rows


@pytest.fixture
def paired_data(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    hashes = {}
    for split, count, offset in (("train", 1296, 0), ("validation", 144, 1296)):
        rows = [{"question_id": str(i), "messages": [
            {"role": "system", "content": "Expand supported expressions"},
            {"role": "user", "content": json.dumps({"question": "How many?", "snippets": [
                {"id": "s", "text": "There are fifteen genes."}]})},
            {"role": "assistant", "content": json.dumps({"answers": [
                {"answer": "15", "relation_type": "original"},
                {"answer": "fifteen", "relation_type": "numerically_equivalent"}]})}]}
                for i in range(offset, offset + count)]
        path = source / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes[path.name] = prepare.digest(path)
    dev = [{"question_id": str(i)} for i in range(1440, 1600)]
    (tmp_path / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dev))
    prepare.write(source / "dev_question_ids.json", [row["question_id"] for row in dev])
    hashes["dev_question_ids.json"] = prepare.digest(source / "dev_question_ids.json")
    prepare.write(source / "manifest.json", {"official_test_overlap": 0, "train_dev_overlap": 0,
                                              "output_sha256": hashes})
    (tmp_path / "single.txt").write_text("Return exactly one tagged answer.")
    matrix = {"source_dataset": "source", "output_dataset": "paired", "outer_dev": "dev.jsonl",
              "single_answer_prompt": "single.txt", "common": {"seed": 3407}, "models": {},
              "source_manifest_sha256": prepare.digest(source / "manifest.json")}
    return tmp_path, matrix


def test_matched_build_preserves_evidence_primary_and_split(paired_data):
    root, matrix = paired_data
    prepare.build(matrix)
    expansion = prepare.records(root / "paired/expansion/train.jsonl")
    original = prepare.records(root / "paired/original/train.jsonl")
    assert len(expansion) == len(original) == 1296
    assert [r["question_id"] for r in expansion] == [r["question_id"] for r in original]
    assert all(a["messages"][1] == b["messages"][1] for a, b in zip(expansion, original))
    assert original[0]["messages"][-1]["content"] == "Answer: [BE]15[EE]"
    assert "fifteen" not in original[0]["messages"][-1]["content"]
    assert "answers" not in original[0]["messages"][1]["content"]
    prepare.build(matrix)  # Identical rebuilds are safe.
    with bundle_imports():
        from run_expansion_sft_qwen3 import validate_inputs
    for arm in ("original", "expansion"):
        monkey_root = prepare.ROOT
        config = prepare.resolved_config({**matrix, "models": {"test": {}}}, "test", arm)
        config = {**config, **{key: str(monkey_root / value) for key, value in config.items()
                               if key in {"train_input", "eval_input", "dataset_manifest"}}}
        train, validation, summary = validate_inputs(config)
        assert len(train) == 1296 and len(validation) == 144
        assert summary["train_validation_overlap"] == 0


def test_source_dev_leakage_fails_before_build(paired_data):
    root, matrix = paired_data
    (root / "dev.jsonl").write_text("".join(json.dumps({"question_id": str(i)}) + "\n" for i in range(160)))
    with pytest.raises(ValueError, match="disjoint"):
        prepare.build(matrix)
    assert not (root / "paired").exists()


def test_source_manifest_cannot_be_silently_replaced(paired_data):
    root, matrix = paired_data
    manifest = prepare.read(root / "source/manifest.json")
    manifest["official_test_overlap"] = 1
    prepare.write(root / "source/manifest.json", manifest)
    with pytest.raises(ValueError, match="pinned"):
        prepare.build(matrix)


def test_training_loads_shared_snapshot_and_rejects_changed_matrix(paired_data, monkeypatch):
    root, matrix = paired_data
    matrix["output_dataset"] = str(root / "paired")
    matrix["models"] = {"qwen3": {"model_name": "Qwen/base"}}
    prepare.build(matrix)
    with bundle_imports():
        import run_matched_8b_sft as runner
        import run_extractive_expansion_8b as expansion_runner
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(expansion_runner, "configure_job_local_compiler_cache", lambda: None)
    config_path = root / "matrix.json"
    prepare.write(config_path, matrix)
    snapshot = root / "snapshots/revision"
    snapshot.mkdir(parents=True)
    prepare.write(snapshot / "config.json", {})
    pins_path = root / "pins.json"
    prepare.write(pins_path, {"matrix_sha256": prepare.digest(config_path), "models": {"qwen3": {
        "model_name": "Qwen/base", "snapshot": str(snapshot), "revision": "revision"}}})
    calls = []
    monkeypatch.setattr(runner, "train", lambda *args: calls.append(args))
    for arm in ("original", "expansion"):
        monkeypatch.setattr(sys, "argv", ["train", "--model", "qwen3", "--formulation", arm,
            "--config", str(config_path), "--pins", str(pins_path), "--run-name", arm])
        runner.main()
    assert len(calls) == 2
    assert calls[0][0].model_name == calls[1][0].model_name == str(snapshot)
    assert all(not call[0].allow_download and not call[0].resume_from_checkpoint for call in calls)
    assert [call[1]["formulation"] for call in calls] == ["original", "expansion"]
    matrix["common"]["seed"] = 3408
    prepare.write(config_path, matrix)
    with pytest.raises(ValueError, match="matrix changed"):
        runner.main()
    assert len(calls) == 2


def test_matched_validation_rejects_tampering_even_with_rehashed_manifest(paired_data):
    root, matrix = paired_data
    prepare.build(matrix)
    path = root / "paired/original/train.jsonl"
    rows = prepare.records(path)
    rows[0]["messages"][1]["content"] = "changed evidence"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = path.parent / "manifest.json"
    manifest = prepare.read(manifest_path)
    manifest["output_sha256"][path.name] = prepare.digest(path)
    prepare.write(manifest_path, manifest)
    with pytest.raises(ValueError, match="differ from source"):
        prepare.validate(matrix)


def test_single_target_validation_accepts_biomedical_brackets_but_rejects_multiple_answers():
    row = {"question_id": "q", "messages": [
        {"role": "system", "content": "single"}, {"role": "user", "content": "question"},
        {"role": "assistant", "content": "Answer: [BE][18F]fluoride[EE]"}]}
    validate_rows([row], "test", "original")
    row["messages"][-1]["content"] += "[BE]second[EE]"
    with pytest.raises(ValueError, match="expected one"):
        validate_rows([row], "test", "original")


def test_response_mask_audit_rejects_prompt_loss_and_empty_targets():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(chr(value) for value in ids)
    text = "[INST]full evidence[/INST]Answer: [BE]15[EE]"
    ids = [ord(c) for c in text]
    boundary = text.index("Answer:")
    valid = {"input_ids": ids, "labels": [-100] * boundary + ids[boundary:]}
    assert audit_response_masks([valid], Tokenizer(), "[/INST]")["supervised_tokens"] > 0
    with pytest.raises(ValueError, match="prompt tokens"):
        audit_response_masks([{**valid, "labels": ids}], Tokenizer(), "[/INST]")
    with pytest.raises(ValueError, match="no trainable"):
        audit_response_masks([{**valid, "labels": [-100] * len(ids)}], Tokenizer(), "[/INST]")


@pytest.mark.parametrize("formulation,raw,passes", [
    ("original", "Answer: [BE]15[EE]", True),
    ("expansion", '{"answers":[{"answer":"15","relation_type":"original"}]}', True),
    ("original", "unparseable", False),
    ("expansion", "Answer: [BE]15[EE]", False)])
def test_generation_smoke_is_gold_blind_and_rejects_zero_answers(tmp_path, monkeypatch, formulation, raw, passes):
    with bundle_imports():
        import check_matched_8b_sft_smoke as smoke
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    directory = tmp_path / "smoke"
    directory.mkdir()
    config = {"eval_input": "validation.jsonl", "model_loader": "fast_model",
              "chat_template_kwargs": {}, "formulation": formulation}
    prepare.write(directory / "status.json", {"status": "completed", "smoke_test": True, "configuration": config})
    rows = [{"question_id": str(i), "messages": [
        {"content": "answer with supplied evidence"},
        {"content": json.dumps({"question": "How many?", "snippets": [{"id": "s", "text": "fifteen"}]})},
        {"content": "SECRET_TARGET"}]} for i in range(4)]
    (tmp_path / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    rendered = []
    monkeypatch.setattr(smoke, "configure_job_local_compiler_cache", lambda: None)
    monkeypatch.setattr(smoke, "render_prompt", lambda *args: rendered.append(args) or "prompt")
    monkeypatch.setattr(smoke, "input_token_count", lambda tokens: 10)
    class Tokens:
        shape = (1, 10)
        def to(self, device): return self
    class Output:
        def __getitem__(self, item): return [11]
    class Model:
        def parameters(self): return iter([SimpleNamespace(device="fake")])
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return Output()
    tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=1, decode=lambda *a, **k: raw)
    monkeypatch.setattr(smoke, "load_model", lambda *a: (Model(), tokenizer))
    monkeypatch.setattr(smoke, "tokenize_text", lambda *a, **k: {"input_ids": Tokens()})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(sys, "argv", ["check", str(directory)])
    if passes:
        smoke.main()
    else:
        with pytest.raises(RuntimeError, match="zero parseable"):
            smoke.main()
    result = prepare.read(directory / "generation_smoke.json")
    assert result["usable_questions"] == (4 if passes else 0)
    assert result["status"] == ("passed" if passes else "failed")
    assert "SECRET_TARGET" not in str(rendered)


def test_model_matrix_matches_settings_and_disables_qwen_thinking():
    matrix = prepare.read(BUNDLE / "configs/matched_8b_sft.json")
    for key in ("llama31", "qwen3", "ministral3"):
        original = prepare.resolved_config(matrix, key, "original")
        expansion = prepare.resolved_config(matrix, key, "expansion")
        assert original["model_name"] == expansion["model_name"]
        assert all(original[field] == expansion[field] for field in matrix["common"])
    assert matrix["common"]["early_stopping_patience"] == 0
    assert matrix["models"]["qwen3"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert matrix["models"]["ministral3"]["response_part"] == "[/INST]"


def test_fast_model_loader_adapts_language_modules_and_keeps_native_template(monkeypatch):
    # Isolate the loader function from CUDA imports, and exercise its actual branch.
    tree = ast.parse((BUNDLE / "src/utility/training.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "load_model_and_tokenizer")
    model = SimpleNamespace(named_modules=lambda: iter([
        ("model.language_model.layers.0.q_proj", None),
        ("model.vision_tower.layers.0.q_proj", None),
        ("model.language_model.layers.0.up_proj", None)]))
    tokenizer = SimpleNamespace(pad_token_id=1)
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            return model, tokenizer
        @staticmethod
        def get_peft_model(actual_model, **kwargs):
            assert actual_model is model
            calls.append(kwargs)
            return model
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastModel=Loader))
    namespace = {"FastLanguageModel": None, "resolve_dtype": lambda value: value,
                 "clean_text": str, "get_chat_template": lambda *a, **k: pytest.fail("native template replaced")}
    compile_tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(compile_tree), "isolated_loader", "exec"), namespace)
    args = SimpleNamespace(model_name="pinned", model_loader="fast_model", max_seq_length=8192,
                           dtype=None, no_4bit=False, local_files_only=True, lora_r=32,
                           lora_alpha=32, lora_dropout=0.05, seed=3407, prompt_format="chat",
                           preserve_native_chat_template=True)
    namespace["load_model_and_tokenizer"](args)
    assert calls[0]["local_files_only"] is True
    assert calls[1]["finetune_vision_layers"] is False
    assert calls[1]["target_modules"] == ["model.language_model.layers.0.q_proj", "model.language_model.layers.0.up_proj"]
