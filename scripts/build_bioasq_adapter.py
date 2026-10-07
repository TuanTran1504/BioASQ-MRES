#!/usr/bin/env python3
"""Build the portable Java 8 scorer adapter with a JDK 9+ compiler."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "src/utility/java"
    source = directory / "BioASQPerQuestionEvaluator.java"
    evaluator = ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    output = directory / "bioasq-per-question-adapter.jar"
    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError("Building the adapter requires a JDK 9+ javac compiler")
    with tempfile.TemporaryDirectory(prefix="bioasq-adapter-build-") as temporary:
        subprocess.run([javac, "--release", "8", "-cp", str(evaluator), "-d", temporary, str(source)], check=True)
        payload = (Path(temporary) / "evaluation/BioASQPerQuestionEvaluator.class").read_bytes()
    if payload[:8] != b"\xca\xfe\xba\xbe\x00\x00\x00\x34":
        raise RuntimeError("Compiler did not produce Java 8 bytecode")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("evaluation/BioASQPerQuestionEvaluator.class", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, payload)
    metadata = {
        "source_sha256": hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        "source_normalization": "universal newlines converted to LF",
        "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "class_major_version": 52,
        "build_command": "javac --release 8 -cp BioASQEvaluation.jar -d BUILD_DIR BioASQPerQuestionEvaluator.java",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built {output.relative_to(ROOT)} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
