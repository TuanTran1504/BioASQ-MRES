import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "gadi_sft_8b_starter/src/utility/training_execution.py"
spec = importlib.util.spec_from_file_location("training_execution_tested", path)
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


def test_eager_recovery_disables_unsloth_and_compiled_lora_execution(monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)
    execution.prepare_execution("eager")
    calls = []
    execution.activate_execution("eager", SimpleNamespace(compiler=SimpleNamespace(set_stance=calls.append)))
    assert execution.os.environ["UNSLOTH_COMPILE_DISABLE"] == "1"
    assert calls == ["force_eager"]


def test_default_mode_does_not_modify_compiler_policy(monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)
    execution.prepare_execution("default")
    execution.activate_execution("default", SimpleNamespace())
    assert "UNSLOTH_COMPILE_DISABLE" not in execution.os.environ
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    with pytest.raises(ValueError, match="compilation disabled"):
        execution.prepare_execution("default")


def test_eager_mode_refuses_missing_compiler_control():
    with pytest.raises(RuntimeError, match="set_stance"):
        execution.activate_execution("eager", SimpleNamespace())
