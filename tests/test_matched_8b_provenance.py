import copy

import pytest

from gadi_sft_8b_starter.scripts.matched_8b_provenance import validate_provenance


@pytest.fixture
def provenance():
    return {name: {"adapter": name + "/adapter", "base_revision": "pinned",
                   "matrix_sha256": "matrix", "status_sha256": "status",
                   "system_prompt": name + " prompt", "model_loader": "fast_language_model",
                   "adapter_files_sha256": {"adapter_model.safetensors": "weights", "tokenizer_config.json": "tokenizer"}}
            for name in ("original", "expansion")}


def test_legacy_default_is_compatible_without_mutating_source(provenance):
    current = copy.deepcopy(provenance)
    for source in current.values():
        source["execution_mode"] = "default"
    validate_provenance(provenance, current)
    validate_provenance(current, provenance)
    assert all("execution_mode" not in source for source in provenance.values())


@pytest.mark.parametrize("field", ["execution_mode", "adapter", "base_revision", "matrix_sha256",
                                   "status_sha256", "system_prompt", "model_loader", "adapter_model.safetensors",
                                   "tokenizer_config.json", "missing_hash"])
def test_real_changes_are_rejected_and_named(provenance, field):
    current = copy.deepcopy(provenance)
    source = current["expansion"]
    if field in ("adapter_model.safetensors", "tokenizer_config.json"):
        source["adapter_files_sha256"][field] = "changed"
    elif field == "missing_hash":
        source.pop("status_sha256")
    else:
        source[field] = "eager" if field == "execution_mode" else "changed"
    with pytest.raises(ValueError, match="expansion." + ("status_sha256" if field == "missing_hash" else
                            "adapter_files_sha256" if field.endswith((".safetensors", ".json")) else field)):
        validate_provenance(provenance, current)
