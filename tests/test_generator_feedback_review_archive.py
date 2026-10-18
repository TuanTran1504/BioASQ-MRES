"""Review transfer archive verifies all source bytes without marking pairs ready."""
import json
from pathlib import Path
import tarfile

import pytest

from scripts import archive_generator_feedback_preferences as archive


def test_pack_verifies_bytes_and_never_overwrites_source_or_archive(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "summary.json").write_text('{"training_ready":false}')
    state = dict(status="review_required",models=["qwen3","ministral3"],training_ready=False,
                 output_sha256={"summary.json":archive.digest(source / "summary.json")})
    (source / "manifest.json").write_text(json.dumps(state))
    previous = archive.digest(source / "manifest.json")
    out = tmp_path / "review.tar.gz"
    archive.pack(source,out)
    with tarfile.open(out) as tar:
        assert set(tar.getnames()) == {"manifest.json","summary.json"}
        assert json.load(tar.extractfile("manifest.json"))["training_ready"] is False
    assert archive.digest(source / "manifest.json") == previous
    with pytest.raises(ValueError): archive.pack(source,out)
    with pytest.raises(ValueError): archive.pack(source,source / "inside.tar.gz")
    (source / "summary.json").write_text("changed")
    with pytest.raises(ValueError,match="changed"): archive.pack(source,tmp_path / "other.tar.gz")
    assert not (tmp_path / "other.tar.gz").exists()
