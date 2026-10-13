from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.model_registry import get_project_root, resolve_repo_path, slugify, utc_now_iso

from .bioasq_format import (
    build_bioasq_prediction_entry,
    normalize_yesno_prediction,
    parse_prediction_items,
)
from .data import clean_text
from .eval_types import EvalExample


OFFICIAL_EXACT_TYPES = ("yesno", "factoid", "list")
_PHASE_B_FLOAT_LINE = re.compile(
    r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?){9}\s*$"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_gold_question(example: EvalExample) -> Dict[str, Any]:
    question_id = clean_text(example.question_id)
    question_type = clean_text(example.question_type).lower()
    body = clean_text(example.body)

    if example.raw_question is not None:
        raw = example.raw_question
        payload: Dict[str, Any] = {
            "id": clean_text(raw.get("id", "")) or question_id,
            "type": clean_text(raw.get("type", "")) or question_type,
            "body": clean_text(raw.get("body", "")) or body,
        }
        if "exact_answer" in raw:
            payload["exact_answer"] = raw.get("exact_answer")
        if "ideal_answer" in raw:
            payload["ideal_answer"] = raw.get("ideal_answer")
        return payload

    payload = {"id": question_id, "type": question_type, "body": body}
    if question_type == "yesno":
        payload["exact_answer"] = normalize_yesno_prediction(example.gold_output)
    elif question_type in {"factoid", "list"}:
        values = parse_prediction_items(example.gold_output, question_type)
        payload["exact_answer"] = [[value] for value in values]
    elif question_type == "summary":
        payload["ideal_answer"] = [clean_text(example.gold_output)] if clean_text(example.gold_output) else []
    return payload


def _phase_b_metrics_from_values(values: Sequence[float], question_type_counts: Mapping[str, int]) -> Dict[str, Any]:
    yesno_count = int(question_type_counts.get("yesno", 0) or 0)
    factoid_count = int(question_type_counts.get("factoid", 0) or 0)
    list_count = int(question_type_counts.get("list", 0) or 0)
    exact_question_count = yesno_count + factoid_count + list_count

    by_type: Dict[str, Any] = {}
    macro_primary_scores = []
    weighted_primary_sum = 0.0

    if yesno_count:
        macro_primary_scores.append(values[7])
        weighted_primary_sum += values[0] * yesno_count
        by_type["yesno"] = {
            "question_count": yesno_count,
            "primary_metric": "macro_f1",
            "average_primary_score": values[7],
            "metrics": {
                "accuracy": values[0],
                "macro_f1": values[7],
                "f1_yes": values[8],
                "f1_no": values[9],
            },
        }

    if factoid_count:
        macro_primary_scores.append(values[3])
        weighted_primary_sum += values[3] * factoid_count
        by_type["factoid"] = {
            "question_count": factoid_count,
            "primary_metric": "mrr",
            "average_primary_score": values[3],
            "metrics": {
                "strict_accuracy": values[1],
                "lenient_accuracy": values[2],
                "mrr": values[3],
            },
        }

    if list_count:
        macro_primary_scores.append(values[6])
        weighted_primary_sum += values[6] * list_count
        by_type["list"] = {
            "question_count": list_count,
            "primary_metric": "mean_f1",
            "average_primary_score": values[6],
            "metrics": {
                "mean_precision": values[4],
                "mean_recall": values[5],
                "mean_f1": values[6],
            },
        }

    return {
        "question_count": exact_question_count,
        "overall_average_primary_score": (
            weighted_primary_sum / exact_question_count if exact_question_count else None
        ),
        "overall_macro_average_primary_score": (
            sum(macro_primary_scores) / len(macro_primary_scores) if macro_primary_scores else None
        ),
        "by_type": by_type,
        "raw_phase_b_metrics": {
            "yesno_accuracy": values[0],
            "factoid_strict_accuracy": values[1],
            "factoid_lenient_accuracy": values[2],
            "factoid_mrr": values[3],
            "list_precision": values[4],
            "list_recall": values[5],
            "list_f1": values[6],
            "yesno_macro_f1": values[7],
            "yesno_f1_yes": values[8],
            "yesno_f1_no": values[9],
        },
    }


def _parse_phase_b_output(stdout: str) -> Sequence[float]:
    for line in reversed(stdout.splitlines()):
        cleaned = clean_text(line)
        if not cleaned:
            continue
        if not _PHASE_B_FLOAT_LINE.match(cleaned):
            continue
        parts = cleaned.split()
        if len(parts) != 10:
            continue
        return [float(part) for part in parts]

    floats = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", stdout)
    if len(floats) == 10:
        return [float(value) for value in floats]
    raise RuntimeError(f"Could not parse BioASQ Java output:\n{stdout}")


def _run_phase_b_command(
    *,
    gold_path: Path,
    prediction_path: Path,
    jar_path: Path,
    java_heap: str,
    challenge_version: int,
) -> Dict[str, Any]:
    java_command = ["java"]
    if shutil.which("java") is None:
        conda_path = shutil.which("conda")
        if conda_path is not None:
            java_command = [conda_path, "run", "-n", "bioasq", "java"]

    command = [
        *java_command,
        f"-Xmx{java_heap}",
        "-cp",
        str(jar_path),
        "evaluation.EvaluatorTask1b",
        "-phaseB",
        "-e",
        str(challenge_version),
        str(gold_path),
        str(prediction_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Java was not found. Install Java or make it available via the `bioasq` conda environment."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "BioASQ official Java evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    values = _parse_phase_b_output(completed.stdout)
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "values": values,
    }


def _conda_executable(tool: str) -> list[str]:
    direct = shutil.which(tool)
    if direct is not None:
        return [direct]
    conda = shutil.which("conda")
    if conda is not None:
        return [conda, "run", "-n", "bioasq", tool]
    candidate = Path.home() / "miniconda3" / "envs" / "bioasq" / "bin" / tool
    if candidate.exists():
        return [str(candidate)]
    requirement = "a Java runtime" if tool == "java" else "a Java JDK compiler"
    raise RuntimeError(f"{tool} was not found; the official BioASQ scorer requires {requirement}.")


def _bundled_official_adapter(source: Path, jar_path: Path) -> Path | None:
    """Use the checked-in Java 8 adapter when its source and evaluator match."""
    archive = source.parent / "bioasq-per-question-adapter.jar"
    metadata_path = archive.with_suffix(".json")
    if not archive.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_digest = hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    if metadata.get("source_sha256") != source_digest or metadata.get("evaluator_sha256") != hashlib.sha256(jar_path.read_bytes()).hexdigest():
        return None
    if metadata.get("artifact_sha256") != hashlib.sha256(archive.read_bytes()).hexdigest():
        raise RuntimeError("Bundled BioASQ adapter checksum mismatch; rebuild or restore the adapter JAR")
    return archive


def _official_adapter_classes(jar_path: Path) -> Path:
    source = Path(__file__).resolve().parent / "java" / "BioASQPerQuestionEvaluator.java"
    if not source.exists():
        raise FileNotFoundError(f"Official per-question adapter source was not found: {source}")
    bundled = _bundled_official_adapter(source, jar_path)
    if bundled is not None:
        return bundled
    # Gadi may provide javac 17 alongside a default Java 8 runtime. Pin bytecode
    # compatibility and separate this cache from earlier compiler-default builds.
    compile_options = ["-source", "8", "-target", "8"]
    digest = hashlib.sha256(
        source.read_bytes() + jar_path.read_bytes() + " ".join(compile_options).encode("ascii")
    ).hexdigest()[:16]
    classes = Path(tempfile.gettempdir()) / f"bioasq-official-adapter-{digest}"
    target = classes / "evaluation" / "BioASQPerQuestionEvaluator.class"
    if target.exists():
        return classes
    classes.mkdir(parents=True, exist_ok=True)
    command = [*_conda_executable("javac"), *compile_options,
               "-cp", str(jar_path), "-d", str(classes), str(source)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not compile the thin per-question adapter for the official BioASQ scorer.\n"
            f"Command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not target.exists():
        raise RuntimeError(f"Java compilation succeeded but did not create {target}")
    return classes


def _run_official_per_question(
    *, gold_path: Path, prediction_path: Path, jar_path: Path, challenge_version: int
) -> list[Dict[str, Any]]:
    classes = _official_adapter_classes(jar_path)
    classpath = os.pathsep.join((str(classes), str(jar_path)))
    command = [
        *_conda_executable("java"), "-cp", classpath,
        "evaluation.BioASQPerQuestionEvaluator",
        str(gold_path), str(prediction_path), str(challenge_version),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Official BioASQ per-question evaluation failed.\n"
            f"Command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    rows: list[Dict[str, Any]] = []
    type_names = {1: "factoid", 2: "yesno", 3: "summary", 4: "list"}
    for line in completed.stdout.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 9:
            continue
        question_type_code = int(parts[1])
        accuracy, strict, lenient, mrr, precision, recall, f1 = map(float, parts[2:])
        question_type = type_names.get(question_type_code, str(question_type_code))
        row: Dict[str, Any] = {
            "question_id": parts[0], "question_type": question_type,
            "backend": "bioasq_java",
        }
        if question_type == "yesno":
            row.update(primary_metric="accuracy", primary_score=accuracy, accuracy=accuracy)
        elif question_type == "factoid":
            row.update(primary_metric="mrr", primary_score=mrr, mrr=mrr,
                       strict_accuracy=strict, lenient_accuracy=lenient)
        elif question_type == "list":
            row.update(primary_metric="f1", primary_score=f1,
                       precision=precision, recall=recall, f1=f1)
        rows.append(row)
    return rows


def evaluate_with_bioasq_java(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    examples_by_key: Mapping[tuple[str, str], EvalExample],
    model_label: str,
    model_dir: Path,
    args: argparse.Namespace,
    include_per_question: bool = True,
) -> Dict[str, Any]:
    project_root = get_project_root()
    jar_path = resolve_repo_path(args.bioasq_java_jar, project_root=project_root) or Path(args.bioasq_java_jar)
    if not jar_path.exists():
        raise FileNotFoundError(f"BioASQ Java evaluator JAR was not found: {jar_path}")

    exact_rows = [
        row
        for row in prediction_rows
        if clean_text(row.get("question_type", "")).lower() in OFFICIAL_EXACT_TYPES
    ]
    if not exact_rows:
        raise ValueError(
            "BioASQ Java scoring only supports yesno, factoid, and list questions. "
            "No such questions were present in this evaluation run."
        )

    official_dir = model_dir / "official_bioasq"
    official_dir.mkdir(parents=True, exist_ok=True)

    def build_files(rows: Sequence[Mapping[str, Any]], stem: str) -> tuple[Path, Path, Dict[str, int]]:
        gold_questions = []
        prediction_questions = []
        type_counts = {"yesno": 0, "factoid": 0, "list": 0}
        for row in rows:
            question_id = clean_text(row.get("question_id", ""))
            question_type = clean_text(row.get("question_type", "")).lower()
            example = examples_by_key.get((question_id, question_type))
            if example is None:
                raise KeyError(
                    f"Could not locate the original evaluation example for question {question_id!r} ({question_type})."
                )
            gold_questions.append(_build_gold_question(example))
            prediction_questions.append(build_bioasq_prediction_entry(dict(row)))
            if question_type in type_counts:
                type_counts[question_type] += 1

        gold_path = official_dir / f"{stem}.gold.json"
        prediction_path = official_dir / f"{stem}.predictions.json"
        _write_json(gold_path, {"questions": gold_questions})
        _write_json(
            prediction_path,
            {
                "system": model_label,
                "questions": prediction_questions,
            },
        )
        return gold_path, prediction_path, type_counts

    overall_gold_path, overall_prediction_path, overall_type_counts = build_files(exact_rows, "overall")
    overall_run = _run_phase_b_command(
        gold_path=overall_gold_path,
        prediction_path=overall_prediction_path,
        jar_path=jar_path,
        java_heap=args.bioasq_java_heap,
        challenge_version=args.bioasq_java_version,
    )
    overall_aggregate = _phase_b_metrics_from_values(overall_run["values"], overall_type_counts)
    per_question = (
        _run_official_per_question(
            gold_path=overall_gold_path,
            prediction_path=overall_prediction_path,
            jar_path=jar_path,
            challenge_version=args.bioasq_java_version,
        )
        if include_per_question else []
    )
    if include_per_question and len(per_question) != len(exact_rows):
        raise RuntimeError(
            f"Official scorer returned {len(per_question)} per-question rows for {len(exact_rows)} inputs."
        )

    rows_by_source: Dict[str, list[Mapping[str, Any]]] = {}
    for row in exact_rows:
        rows_by_source.setdefault(str(row.get("source_path") or ""), []).append(row)

    batch_payloads = []
    for source_path, rows in sorted(rows_by_source.items(), key=lambda item: item[0]):
        batch_path = Path(source_path) if source_path else None
        batch_id = (
            clean_text(batch_path.stem)
            if batch_path is not None and clean_text(batch_path.stem)
            else clean_text(batch_path.name)
            if batch_path is not None and clean_text(batch_path.name)
            else source_path
        )
        batch_stem = f"batch-{slugify(batch_id, fallback='batch')}"
        gold_path, prediction_path, type_counts = build_files(rows, batch_stem)
        batch_run = _run_phase_b_command(
            gold_path=gold_path,
            prediction_path=prediction_path,
            jar_path=jar_path,
            java_heap=args.bioasq_java_heap,
            challenge_version=args.bioasq_java_version,
        )
        batch_payloads.append(
            {
                "batch_id": batch_id,
                "source_path": source_path,
                "question_count": len(rows),
                "question_types": sorted(
                    {
                        clean_text(row.get("question_type", "")).lower()
                        for row in rows
                        if clean_text(row.get("question_type", ""))
                    }
                ),
                "aggregate": _phase_b_metrics_from_values(batch_run["values"], type_counts),
                "paths": {
                    "gold": str(gold_path),
                    "predictions": str(prediction_path),
                },
            }
        )

    payload = {
        "created_at": utc_now_iso(),
        "backend": "bioasq_java",
        "phase": "phaseB",
        "challenge_version": args.bioasq_java_version,
        "java_heap": args.bioasq_java_heap,
        "jar_path": str(jar_path),
        "question_count": len(exact_rows),
        "ignored_question_count": len(prediction_rows) - len(exact_rows),
        "question_types": sorted(
            {
                clean_text(row.get("question_type", "")).lower()
                for row in exact_rows
                if clean_text(row.get("question_type", ""))
            }
        ),
        "aggregate": overall_aggregate,
        "per_question": per_question,
        "batches": batch_payloads,
        "paths": {
            "dir": str(official_dir),
            "gold": str(overall_gold_path),
            "predictions": str(overall_prediction_path),
        },
        "command": overall_run["command"],
        "stdout": overall_run["stdout"],
        "stderr": overall_run["stderr"],
    }
    _write_json(official_dir / "official_scores.json", payload)
    return payload
