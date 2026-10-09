import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "gadi_sft_8b_starter/src/utility/ministral_generation.py"
spec = importlib.util.spec_from_file_location("ministral_generation_tested", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("wrapped", [False, True])
def test_ministral_dynamic_generation_retains_peft_hooks_and_fp16_context(monkeypatch, wrapped):
    contexts, calls, hooks = [], [], []
    def autocast(**kwargs):
        contexts.append(kwargs)
        return nullcontext()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="fp16", inference_mode=nullcontext, autocast=autocast))
    def original_generate(**kwargs):
        assert kwargs["cache_implementation"] == "dynamic"
        assert kwargs["disable_compile"] is True
        calls.append(kwargs)
        return "generated answer"
    backbone = SimpleNamespace(config=SimpleNamespace(model_type="mistral3"),
                               _old_generate=original_generate,
                               generate=lambda **kw: pytest.fail("Unsloth forced-static wrapper used"))
    if wrapped:
        class PeftModel:
            def get_base_model(self): return backbone
            def generate(self, **kwargs):
                hooks.append("PEFT adapter hooks")
                return backbone.generate(**kwargs)
        model = PeftModel()
    else:
        model = backbone
    assert module.configure_ministral_generation(model) is model
    patched = backbone.generate
    module.configure_ministral_generation(model)
    assert backbone.generate is patched  # Reload configuration is idempotent.
    assert model.generate(max_new_tokens=512, do_sample=False, use_cache=True,
                          cache_implementation="static") == "generated answer"
    assert contexts == [{"device_type": "cuda", "dtype": "fp16"}]
    assert calls[0]["max_new_tokens"] == 512 and calls[0]["do_sample"] is False
    assert hooks == (["PEFT adapter hooks"] if wrapped else [])


def test_other_backbones_keep_their_existing_generation_backend():
    for model_type in ("qwen3", "llama", "gemma3"):
        original = lambda: None
        model = SimpleNamespace(config=SimpleNamespace(model_type=model_type), generate=original)
        assert module.configure_ministral_generation(model).generate is original
