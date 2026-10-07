from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pytest

from src.utility.bioasq_official import evaluate_with_bioasq_java
from src.utility.eval_types import EvalExample


def test_java_per_question_scores_match_aggregate_with_spaced_paths(tmp_path, monkeypatch):
    """Exercise Java class loading and scoring, including Windows classpaths."""
    if not shutil.which("java") or not shutil.which("javac"):
        pytest.skip("Official evaluation integration requires a Java JDK")
    jar = (Path(__file__).resolve().parents[1]
           / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    if not jar.is_file():
        pytest.skip("Official BioASQ JAR is not available")
    # Preserve the JAR's relative library dependencies; put the adapter classes
    # in a spaced directory to exercise classpath handling and javac together.
    from src.utility import bioasq_official
    monkeypatch.setattr(bioasq_official, "_bundled_official_adapter", lambda *args: None)
    class_cache = tmp_path / "scorer with spaces"
    monkeypatch.setattr(bioasq_official.tempfile, "gettempdir", lambda: str(class_cache))
    source = Path(bioasq_official.__file__).parent / "java/BioASQPerQuestionEvaluator.java"
    old_digest = hashlib.sha256(source.read_bytes() + jar.read_bytes()).hexdigest()[:16]
    old_class = class_cache / f"bioasq-official-adapter-{old_digest}/evaluation/BioASQPerQuestionEvaluator.class"
    old_class.parent.mkdir(parents=True)
    stale_bytes = b"\xca\xfe\xba\xbe\x00\x00\x00\x3dold Java 17 cache"
    old_class.write_bytes(stale_bytes)
    examples = {
        (qid, "factoid"): EvalExample(qid, "factoid", "Which?", "", (),
                                     "[BE]alpha[EE]", "dev.json")
        for qid in ("correct", "wrong")
    }
    predictions = [
        {"question_id": qid, "question_type": "factoid", "source_path": "dev.json", "body": "Which?",
         "prediction": f"[BE]{answer}[EE]"}
        for qid, answer in (("correct", "alpha"), ("wrong", "beta"))
    ]
    result = evaluate_with_bioasq_java(
        prediction_rows=predictions, examples_by_key=examples,
        model_label="test", model_dir=tmp_path / "evaluation with spaces",
        args=argparse.Namespace(bioasq_java_jar=str(jar),
                                bioasq_java_heap="256M", bioasq_java_version=5),
    )
    scores = {row["question_id"]: row["mrr"] for row in result["per_question"]}
    assert scores == {"correct": 1.0, "wrong": 0.0}
    assert result["aggregate"]["overall_average_primary_score"] == pytest.approx(.5)
    new_classes = [path for path in class_cache.rglob("BioASQPerQuestionEvaluator.class") if path != old_class]
    assert len(new_classes) == 1
    # Major version 52 is Java 8, including when javac itself is Java 17.
    assert new_classes[0].read_bytes()[:8] == b"\xca\xfe\xba\xbe\x00\x00\x00\x34"
    assert old_class.read_bytes() == stale_bytes


def test_bundled_adapter_scores_without_a_compiler(tmp_path, monkeypatch):
    if not shutil.which("java"):
        pytest.skip("Official evaluation integration requires a Java runtime")
    from src.utility import bioasq_official
    jar = Path(__file__).resolve().parents[1] / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    if not jar.exists():
        pytest.skip("Official BioASQ JAR is not available")
    original = bioasq_official._conda_executable

    def runtime_only(tool):
        assert tool != "javac", "Scoring must not invoke a compiler"
        return original(tool)

    monkeypatch.setattr(bioasq_official, "_conda_executable", runtime_only)
    monkeypatch.setattr(bioasq_official.tempfile, "gettempdir", lambda: str(tmp_path / "no compile cache"))
    example = EvalExample("correct", "factoid", "Which?", "", (), "[BE]alpha[EE]", "dev.json")
    result = evaluate_with_bioasq_java(
        prediction_rows=[{"question_id": "correct", "question_type": "factoid", "source_path": "dev.json",
                          "body": "Which?", "prediction": "[BE]alpha[EE]"}],
        examples_by_key={("correct", "factoid"): example}, model_label="runtime-only", model_dir=tmp_path,
        args=argparse.Namespace(bioasq_java_jar=str(jar), bioasq_java_heap="256M", bioasq_java_version=5),
    )
    assert result["per_question"][0]["mrr"] == 1.0
    assert not (tmp_path / "no compile cache").exists()


def test_bundled_adapter_uses_normalized_source_hash_and_checks_artifact(tmp_path):
    import json
    from src.utility import bioasq_official
    source = tmp_path / "source.java"
    source.write_bytes(b"class Example {\r\n}\r\n")
    evaluator = tmp_path / "evaluator.jar"
    evaluator.write_bytes(b"evaluator")
    artifact = tmp_path / "bioasq-per-question-adapter.jar"
    artifact.write_bytes(b"adapter")
    metadata = {"source_sha256": hashlib.sha256(b"class Example {\n}\n").hexdigest(),
                "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    artifact.with_suffix(".json").write_text(json.dumps(metadata))
    assert bioasq_official._bundled_official_adapter(source, evaluator) == artifact
    artifact.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bioasq_official._bundled_official_adapter(source, evaluator)
    evaluator.write_bytes(b"different evaluator")
    assert bioasq_official._bundled_official_adapter(source, evaluator) is None
