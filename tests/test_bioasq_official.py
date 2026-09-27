from __future__ import annotations

import argparse
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
    class_cache = tmp_path / "scorer with spaces"
    monkeypatch.setattr(bioasq_official.tempfile, "gettempdir", lambda: str(class_cache))
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
