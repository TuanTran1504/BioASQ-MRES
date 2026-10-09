import ast
import importlib.util
from contextlib import contextmanager
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "gadi_sft_8b_starter"

_text_spec = importlib.util.spec_from_file_location(
    "bundle_text_tokenizer", BUNDLE / "src/utility/text_tokenizer.py")
_text_module = importlib.util.module_from_spec(_text_spec)
_text_spec.loader.exec_module(_text_module)
text_only_tokenizer = _text_module.text_only_tokenizer
_generation_spec = importlib.util.spec_from_file_location(
    "bundle_ministral_generation", BUNDLE / "src/utility/ministral_generation.py")
_generation_module = importlib.util.module_from_spec(_generation_spec)
_generation_spec.loader.exec_module(_generation_module)


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


@pytest.mark.parametrize("adapter_scope", ["language", "vision", "empty"])
@pytest.mark.parametrize("processor_wrapped", [False, True])
def test_fast_model_loader_adapts_language_modules_and_keeps_native_template(
        monkeypatch, adapter_scope, processor_wrapped):
    # Isolate the loader function from CUDA imports, and exercise its actual branch.
    tree = ast.parse((BUNDLE / "src/utility/training.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "load_model_and_tokenizer")
    model = SimpleNamespace(named_modules=lambda: iter([
        ("model.language_model.layers.0.self_attn.q_proj", None),
        ("model.vision_tower.layers.0.self_attn.q_proj", None),
        ("model.language_model.layers.0.mlp.up_proj", None)]))
    scope_name = "language_model" if adapter_scope == "language" else "vision_tower"
    model.named_parameters = lambda: iter([] if adapter_scope == "empty" else [
        (f"base_model.model.model.{scope_name}.layers.0.self_attn.q_proj.lora_A.default.weight",
         SimpleNamespace(requires_grad=True))])
    tokenizer = SimpleNamespace(pad_token_id=1, chat_template="native")
    processing_object = SimpleNamespace(tokenizer=tokenizer, chat_template="processor") if processor_wrapped else tokenizer
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            return model, processing_object
        @staticmethod
        def get_peft_model(actual_model, **kwargs):
            assert actual_model is model
            # FastModel's scoped regex API accepts projection leaf names, not
            # qualified module paths. Reproduce that contract from the failure.
            assert all("." not in name for name in kwargs["target_modules"])
            assert kwargs["finetune_language_layers"] is True
            assert kwargs["finetune_vision_layers"] is False
            import re
            matcher = re.compile(r".*language.*(?:self_attn|mlp).*\." +
                                 "(?:" + "|".join(re.escape(name) for name in kwargs["target_modules"]) + ")")
            selected = [name for name, _ in actual_model.named_modules() if matcher.fullmatch(name)]
            assert selected == ["model.language_model.layers.0.self_attn.q_proj",
                                "model.language_model.layers.0.mlp.up_proj"]
            calls.append(kwargs)
            return model
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastModel=Loader))
    namespace = {"FastLanguageModel": None, "resolve_dtype": lambda value: value,
                 "text_only_tokenizer": text_only_tokenizer,
                 "clean_text": str, "get_chat_template": lambda *a, **k: pytest.fail("native template replaced")}
    compile_tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(compile_tree), "isolated_loader", "exec"), namespace)
    args = SimpleNamespace(model_name="pinned", model_loader="fast_model", max_seq_length=8192,
                           dtype=None, no_4bit=False, local_files_only=True, lora_r=32,
                           lora_alpha=32, lora_dropout=0.05, seed=3407, prompt_format="chat",
                           preserve_native_chat_template=True)
    if adapter_scope == "language":
        _, actual_tokenizer = namespace["load_model_and_tokenizer"](args)
        assert actual_tokenizer is tokenizer
        assert actual_tokenizer.chat_template == "native"
    else:
        message = "vision/projector" if adapter_scope == "vision" else "no trainable"
        with pytest.raises(ValueError, match=message):
            namespace["load_model_and_tokenizer"](args)
    assert calls[0]["local_files_only"] is True
    assert calls[1]["finetune_vision_layers"] is False
    assert calls[1]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_text_tokenizer_preserves_native_template_and_recovers_processor_only_template():
    tokenizer = SimpleNamespace(chat_template="tokenizer native")
    processor = SimpleNamespace(tokenizer=tokenizer, chat_template="processor native")
    assert text_only_tokenizer(processor) is tokenizer
    assert tokenizer.chat_template == "tokenizer native"
    tokenizer.chat_template = None
    assert text_only_tokenizer(processor).chat_template == "processor native"
    tokenizer.chat_template = processor.chat_template = None
    with pytest.raises(ValueError, match="no native text chat template"):
        text_only_tokenizer(processor)
    assert text_only_tokenizer(tokenizer) is tokenizer


def test_text_only_preflight_bypasses_processor_content_block_error():
    from run_expansion_sft_qwen3 import token_preflight

    class TextTokenizer:
        chat_template = "[INST]native[/INST]"

        def apply_chat_template(self, messages, **kwargs):
            assert all(isinstance(message["content"], str) for message in messages)
            text = "[INST]" + messages[0]["content"] + "[/INST]" + messages[-1]["content"]
            return [ord(c) for c in text] if kwargs["tokenize"] else text

    class Processor:
        tokenizer = TextTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            # Reproduce ProcessorMixin's access that failed on Gadi.
            return [content["type"] for message in messages for content in message["content"]]

    processor = Processor()
    rows = [{"question_id": "q", "messages": [
        {"role": "user", "content": "full evidence"},
        {"role": "assistant", "content": "Answer: [BE]15[EE]"}]}]
    with pytest.raises(TypeError):
        token_preflight(rows, processor, max_seq_length=8192, chat_template_kwargs={}, label="train")
    summary, lengths = token_preflight(rows, text_only_tokenizer(processor),
        max_seq_length=8192, chat_template_kwargs={}, label="train")
    assert summary["overflow_count"] == 0
    assert lengths["q"] == len("[INST]full evidence[/INST]Answer: [BE]15[EE]")


def test_fast_model_inference_unwraps_processor_before_smoke_generation(monkeypatch):
    # Exercise the inference loader without a GPU or the training imports.
    with bundle_imports():
        import run_extractive_expansion_8b as inference
    tokenizer = SimpleNamespace(chat_template="native", pad_token_id=None,
                                eos_token_id=2, eos_token="</s>")
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(eval=lambda: model)
    inference_calls = []
    loader = SimpleNamespace(from_pretrained=lambda **kwargs: (model, processor),
                             for_inference=lambda actual: inference_calls.append(actual))
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastModel=loader))
    monkeypatch.setitem(sys.modules, "src.utility.eval_models", SimpleNamespace(
        load_model_and_tokenizer_for_eval=None, prime_unsloth_runtime=lambda: None))
    monkeypatch.setitem(sys.modules, "src.utility.eval_types", SimpleNamespace(
        ModelSpec=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "src.utility.text_tokenizer", _text_module)
    monkeypatch.setitem(sys.modules, "src.utility.ministral_generation", _generation_module)
    actual_model, actual_tokenizer = inference.load_model("pinned", 8192, True, "fast_model")
    assert actual_model is model and actual_tokenizer is tokenizer
    assert tokenizer.chat_template == "native"
    assert tokenizer.pad_token == "</s>"
    assert inference_calls == [model]


@pytest.fixture
def completed_pairs(paired_data, monkeypatch):
    root, matrix = paired_data
    matrix["models"] = {"qwen3": {"model_name": "Qwen/base", "model_loader": "fast_language_model",
                                  "chat_template_kwargs": {"enable_thinking": False}}}
    prepare.build(matrix)
    matrix_path = root / "matrix.json"
    prepare.write(matrix_path, matrix)
    snapshot = root / "snapshots/pinned-revision"
    snapshot.mkdir(parents=True)
    prepare.write(snapshot / "config.json", {})
    runs = {}
    for formulation in ("original", "expansion"):
        name = "training-" + formulation
        directory = root / name
        adapter = directory / "adapter"
        adapter.mkdir(parents=True)
        config = prepare.resolved_config(matrix, "qwen3", formulation)
        config.update(matrix_sha256=prepare.digest(matrix_path), base_snapshot=str(snapshot),
                      base_revision=snapshot.name)
        prepare.write(directory / "status.json", {"status": "completed", "smoke_test": False,
                                                  "selected_train_examples": 1296, "selected_validation_examples": 144,
                                                  "dataset_validation": {
                                                      "train_sha256": prepare.digest(root / config["train_input"]),
                                                      "validation_sha256": prepare.digest(root / config["eval_input"])},
                                                  "configuration": config})
        prepare.write(adapter / "training_complete.json", {"status": "completed"})
        prepare.write(adapter / "adapter_config.json", {"base_model_name_or_path": str(snapshot)})
        prepare.write(adapter / "tokenizer_config.json", {"chat_template": "native"})
        (adapter / "adapter_model.safetensors").write_bytes(b"test weights; never loaded")
        runs[formulation] = name
    evaluation = {"training_matrix": "matrix.json", "input": "dev.jsonl",
                  "input_sha256": prepare.digest(root / "dev.jsonl"),
                  "max_seq_length": 6144, "max_new_tokens": 512, "seed": 3407,
                  "models": {"qwen3": runs}}
    with bundle_imports():
        import run_matched_8b_evaluation as runner
    monkeypatch.setattr(runner, "ROOT", root)
    return root, evaluation, runner


def test_evaluation_validates_completed_pair_and_rejects_smoke_or_wrong_backbone(completed_pairs):
    root, config, runner = completed_pairs
    provenance = runner.validate_training_pair(config, "qwen3")
    assert provenance["original"]["base_revision"] == provenance["expansion"]["base_revision"]
    assert provenance["original"]["system_prompt"] == "Return exactly one tagged answer."
    path = root / "training-expansion/status.json"
    state = prepare.read(path)
    state["smoke_test"] = True
    prepare.write(path, state)
    with pytest.raises(ValueError, match="completed full"):
        runner.validate_training_pair(config, "qwen3")
    state["smoke_test"] = False
    prepare.write(path, state)
    prepare.write(root / "training-expansion/adapter/adapter_config.json", {"base_model_name_or_path": "other"})
    with pytest.raises(ValueError, match="backbone differs"):
        runner.validate_training_pair(config, "qwen3")


def test_evaluation_rejects_changed_dev_export_before_loading_weights(completed_pairs):
    root, config, runner = completed_pairs
    path = root / "dev.jsonl"
    path.write_text(path.read_text() + "\n")  # Same IDs, altered pinned file.
    with pytest.raises(ValueError, match="outer_dev_sha256|pinned 160"):
        runner.validate_training_pair(config, "qwen3")


@pytest.mark.parametrize("empty_arm", [None, "original_sampling10"])
def test_evaluation_runs_three_arms_with_native_prompts_and_gates_bad_smoke(completed_pairs, monkeypatch, empty_arm):
    root, config, runner = completed_pairs
    config_path = root / "evaluation.json"
    prepare.write(config_path, config)
    calls = []

    def subprocess_run(command, check):
        assert check
        arm_path = Path(command[command.index("--config") + 1])
        arm = prepare.read(arm_path)
        condition = arm_path.name.removesuffix("_config.json")
        calls.append((condition, arm, command))
        target = Path(command[command.index("--output-parent" if condition == "expansion_greedy" else "--output-dir") + 1])
        if condition == "expansion_greedy":
            target /= "generated"
        target.mkdir(parents=True)
        prepare.write(target / "status.json", {"status": "complete", "completed_questions": 4})
        rows = [{"question_id": str(i), "raw_response": "invalid"} for i in range(4)]
        (target / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        candidates = [] if condition == empty_arm else [{"question_id": "0", "answer": "15"}]
        (target / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidates))

    monkeypatch.setattr(runner.subprocess, "run", subprocess_run)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config_path), "--model", "qwen3",
                                    "--mode", "run", "--smoke-test", "--run-name", "test-evaluation"])
    if empty_arm:
        with pytest.raises(ValueError, match="zero parseable"):
            runner.main()
    else:
        runner.main()
    manifest = prepare.read(root / "outputs/matched_8b_evaluation/test-evaluation/manifest.json")
    assert manifest["status"] == ("failed" if empty_arm else "complete")
    assert [row[0] for row in calls] == list(runner.CONDITIONS)
    assert all(row[1]["mark_snippets"] is False for row in calls)
    assert all(row[1]["chat_template_kwargs"] == {"enable_thinking": False} for row in calls)
    assert [row[1]["num_generations"] for row in calls] == [1, 10, 1]
    assert [row[1]["temperature"] for row in calls] == [0, 0.8, 0]
    assert calls[0][1]["prompt"] == calls[1][1]["prompt"]
    assert Path(calls[0][1]["prompt"]).read_text() == "Return exactly one tagged answer."
    assert calls[0][2][calls[0][2].index("--model-name") + 1] != calls[2][2][calls[2][2].index("--model-name") + 1]
