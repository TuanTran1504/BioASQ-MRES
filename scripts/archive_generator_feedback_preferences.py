#!/usr/bin/env python3
"""Pack verified provisional generator preferences for local semantic review."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def pack(directory, archive):
    directory, archive = directory.resolve(), archive.resolve()
    if archive.exists() or archive.is_relative_to(directory):
        raise ValueError("Choose a new archive outside the preference directory")
    state_path = directory / "manifest.json"
    original_manifest = digest(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    models = state.get("models", [])
    if (state.get("status") != "review_required" or not models or len(models) != len(set(models))
            or not set(models) <= {"llama31", "qwen3", "ministral3"} or state.get("training_ready") is not False):
        raise ValueError("Expected completed provisional model-specific preferences")
    files = {"manifest.json": original_manifest, **state["output_sha256"]}
    def verify():
        for name, expected in files.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or digest(path) != expected:
                raise ValueError("Preference export changed or escaped its directory: "+name)
    verify()
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "x:gz") as handle:
            for name in sorted(files): handle.add(directory / name, arcname=name, recursive=False)
        verify()
        with tarfile.open(archive, "r:gz") as handle:
            if {m.name for m in handle.getmembers()} != set(files):
                raise ValueError("Archive file membership differs")
            for name, expected in files.items():
                stream = handle.extractfile(name)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                    raise ValueError("Archive content differs: "+name)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    print(f"Verified review archive for {', '.join(models)}: {archive}", flush=True)
    print(f"SHA256: {digest(archive)}; provisional only, not approved DPO training data.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    pack(a.directory, a.archive)
