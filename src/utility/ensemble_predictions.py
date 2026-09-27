from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.model_registry import get_project_root, resolve_repo_path, utc_now_iso
from src.prompt_registry import resolve_prompt_bundle

from .bioasq_format import build_bioasq_prediction_entry
from .bioasq_official import evaluate_with_bioasq_java
from .config import QUESTION_INSTRUCTIONS
from .data import clean_text
from .eval_dataset import load_eval_examples, resolve_eval_input_paths
from .eval_models import normalize_generated_item, parse_tagged_items
from .eval_types import EvalExample


PredictionRows = Dict[str, Dict[str, Any]]


def parse_threshold_values(value: str) -> List[float]:
    values: List[float] = []
    for part in value.split(","):
        part = clean_text(part)
        if not part:
            continue
        values.append(float(part))
    return values


def parse_labeled_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    label, path_value = value.split("=", 1)
    label = clean_text(label)
    if not label:
        raise ValueError(f"Missing label in --prediction value: {value}")
    return label, Path(path_value)


def parse_weight(value: str) -> Tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=WEIGHT for --weight, got: {value}")
    label, weight_value = value.split("=", 1)
    label = clean_text(label)
    if not label:
        raise ValueError(f"Missing label in --weight value: {value}")
    return label, float(weight_value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_prediction_rows(path: Path) -> PredictionRows:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Prediction file must contain a JSON list: {path}")
    return {
        clean_text(row.get("question_id", "")): row
        for row in rows
        if isinstance(row, dict) and clean_text(row.get("question_id", ""))
    }


def tagged_items(text: str, question_type: str) -> List[str]:
    if question_type == "factoid":
        return parse_tagged_items(text, "[BE]", "[EE]")
    if question_type == "list":
        return parse_tagged_items(text, "[BI]", "[EI]")
    return []


def samples_for_row(row: Mapping[str, Any]) -> List[str]:
    samples = row.get("generation_samples")
    if isinstance(samples, list) and samples:
        return [clean_text(sample) for sample in samples if clean_text(sample)]
    prediction = clean_text(row.get("prediction", ""))
    return [prediction] if prediction else []


def ordered_representatives(sample_groups: Sequence[Sequence[str]]) -> Tuple[Dict[str, str], List[str]]:
    representatives: Dict[str, str] = {}
    order: List[str] = []
    for items in sample_groups:
        for item in items:
            key = normalize_generated_item(item)
            if not key or key in representatives:
                continue
            representatives[key] = clean_text(item)
            order.append(key)
    return representatives, order


def format_items(keys: Sequence[str], representatives: Mapping[str, str], question_type: str) -> str:
    if question_type == "factoid":
        return " ".join(f"[BE] {representatives[key]} [EE]" for key in keys if key in representatives)
    if question_type == "list":
        return " ".join(f"[BI] {representatives[key]} [EI]" for key in keys if key in representatives)
    return ""


def majority_yesno(model_rows: Mapping[str, Mapping[str, Any]]) -> str:
    votes = []
    for row in model_rows.values():
        for sample in samples_for_row(row):
            value = clean_text(sample).lower()
            if value.startswith("yes"):
                votes.append("yes")
            elif value.startswith("no"):
                votes.append("no")
    if not votes:
        return ""
    counts = Counter(votes)
    return "yes" if counts["yes"] >= counts["no"] else "no"


def ensemble_prediction(
    example: EvalExample,
    rows_by_label: Mapping[str, PredictionRows],
    weights: Mapping[str, float],
    vote_threshold: float,
    fallback_label: str,
) -> str:
    model_rows = {
        label: rows[example.question_id]
        for label, rows in rows_by_label.items()
        if example.question_id in rows
    }
    if not model_rows:
        return ""

    if example.question_type == "yesno":
        return majority_yesno(model_rows)

    if example.question_type not in {"list", "factoid"}:
        unique_samples = []
        seen = set()
        for row in model_rows.values():
            for sample in samples_for_row(row):
                key = clean_text(sample)
                if key and key not in seen:
                    seen.add(key)
                    unique_samples.append(key)
        return "\n\n".join(unique_samples)

    sample_groups: List[List[str]] = []
    weighted_votes: Dict[str, float] = defaultdict(float)
    for label, row in model_rows.items():
        weight = weights.get(label, 1.0)
        for sample in samples_for_row(row):
            items = tagged_items(sample, example.question_type)
            sample_groups.append(items)
            sample_keys = {
                normalize_generated_item(item)
                for item in items
                if normalize_generated_item(item)
            }
            for key in sample_keys:
                weighted_votes[key] += weight

    representatives, order = ordered_representatives(sample_groups)
    selected_keys = [key for key in order if weighted_votes.get(key, 0.0) >= vote_threshold]

    if not selected_keys:
        fallback_row = model_rows.get(fallback_label) or next(iter(model_rows.values()))
        fallback_items = tagged_items(clean_text(fallback_row.get("prediction", "")), example.question_type)
        fallback_reps, fallback_order = ordered_representatives([fallback_items])
        representatives.update(fallback_reps)
        selected_keys = fallback_order

    return format_items(selected_keys, representatives, example.question_type)


def resolve_prediction_inputs(values: Iterable[str], project_root: Path) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for value in values:
        label, raw_path = parse_labeled_path(value)
        path = resolve_repo_path(str(raw_path), project_root=project_root) or raw_path
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found for {label}: {path}")
        resolved[label] = path
    if not resolved:
        raise ValueError("Pass at least one --prediction LABEL=PATH.")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline multi-model ensemble evaluator for saved BioASQ predictions.")
    parser.add_argument("--prediction", action="append", default=[], help="Saved predictions JSON as LABEL=PATH.")
    parser.add_argument("--weight", action="append", default=[], help="Optional model vote weight as LABEL=WEIGHT.")
    parser.add_argument("--vote-threshold", type=float, default=2.0)
    parser.add_argument(
        "--sweep-vote-thresholds",
        default="",
        help="Comma-separated thresholds to evaluate; writes threshold_sweep.json/csv and uses the best F1 threshold.",
    )
    parser.add_argument("--fallback-label", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-input", action="append", default=[])
    parser.add_argument("--question-types", nargs="+", default=["list"])
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt-registry-path", default="")
    parser.add_argument("--max-resources", type=int, default=3)
    parser.add_argument("--max-resource-chars", type=int, default=1200)
    parser.add_argument("--max-summary-answers", type=int, default=1)
    parser.add_argument("--max-factoid-answers", type=int, default=5)
    parser.add_argument("--max-list-items", type=int, default=100)
    parser.add_argument("--summary-reference-mode", choices=["first", "all-mean", "all-max"], default="first")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bioasq-java-jar", default="third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    parser.add_argument("--bioasq-java-version", type=int, default=5, choices=[2, 3, 5, 8, 9])
    parser.add_argument("--bioasq-java-heap", default="512m")
    return parser


def build_prediction_rows(
    examples: Sequence[EvalExample],
    rows_by_label: Mapping[str, PredictionRows],
    weights: Mapping[str, float],
    vote_threshold: float,
    fallback_label: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    prediction_rows = []
    for example in examples:
        prediction = ensemble_prediction(
            example=example,
            rows_by_label=rows_by_label,
            weights=weights,
            vote_threshold=vote_threshold,
            fallback_label=fallback_label,
        )
        prediction_rows.append(
            {
                "question_id": example.question_id,
                "question_type": example.question_type,
                "body": example.body,
                "source_path": example.source_path,
                "prompt_instruction": example.instruction,
                "prediction": prediction,
                "gold_output": example.gold_output,
            }
        )
    return prediction_rows


def metric_row(
    vote_threshold: float,
    prediction_rows: Sequence[Dict[str, Any]],
    aggregate: Mapping[str, Any],
) -> Dict[str, Any]:
    list_metrics = aggregate.get("by_type", {}).get("list", {}).get("metrics", {})
    return {
        "vote_threshold": vote_threshold,
        "mean_f1": list_metrics.get("mean_f1"),
        "mean_precision": list_metrics.get("mean_precision"),
        "mean_recall": list_metrics.get("mean_recall"),
        "avg_prediction_count": (
            sum(len(tagged_items(str(row.get("prediction") or ""), str(row.get("question_type") or "")))
                for row in prediction_rows) / len(prediction_rows)
            if prediction_rows
            else 0.0
        ),
    }


def write_threshold_sweep_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["vote_threshold", "mean_f1", "mean_precision", "mean_recall", "avg_prediction_count"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = get_project_root()
    prediction_paths = resolve_prediction_inputs(args.prediction, project_root=project_root)
    rows_by_label = {
        label: load_prediction_rows(path)
        for label, path in prediction_paths.items()
    }
    weights = {label: 1.0 for label in prediction_paths}
    weights.update(parse_weight(value) for value in args.weight)
    fallback_label = clean_text(args.fallback_label) or next(iter(prediction_paths))

    prompt_path_value = args.prompt_file or args.prompt_registry_path
    prompt_registry_path = resolve_repo_path(prompt_path_value, project_root=project_root)
    prompt_bundle = resolve_prompt_bundle(
        registry_path=prompt_registry_path,
        prompt_ref=args.prompt,
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )
    eval_paths = resolve_eval_input_paths(args, project_root=project_root)
    examples = load_eval_examples(eval_paths, args, prompt_instructions=prompt_bundle["instructions"])
    supported_question_types = sorted({example.question_type for example in examples})
    if not set(supported_question_types).issubset({"yesno", "factoid", "list"}):
        raise ValueError("Ensemble scoring requires official BioASQ Phase-B exact-answer question types.")
    examples_by_key = {(example.question_id, example.question_type): example for example in examples}

    output_dir = resolve_repo_path(args.output_dir, project_root=project_root) or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_sweep = []
    sweep_values = parse_threshold_values(args.sweep_vote_thresholds)
    selected_threshold = args.vote_threshold
    if sweep_values:
        for threshold in sweep_values:
            sweep_prediction_rows = build_prediction_rows(
                examples=examples,
                rows_by_label=rows_by_label,
                weights=weights,
                vote_threshold=threshold,
                fallback_label=fallback_label,
                args=args,
            )
            sweep_payload = evaluate_with_bioasq_java(
                prediction_rows=sweep_prediction_rows, examples_by_key=examples_by_key,
                model_label=f"ensemble-threshold-{threshold}",
                model_dir=output_dir/f"official_threshold_{threshold}", args=args,
                include_per_question=True)
            sweep_aggregate = sweep_payload["aggregate"]
            threshold_sweep.append(metric_row(threshold, sweep_prediction_rows, sweep_aggregate))
        threshold_sweep.sort(
            key=lambda row: (
                float(row.get("mean_f1") or 0.0),
                float(row.get("mean_precision") or 0.0),
                float(row.get("mean_recall") or 0.0),
            ),
            reverse=True,
        )
        selected_threshold = float(threshold_sweep[0]["vote_threshold"])
        write_json(output_dir / "threshold_sweep.json", threshold_sweep)
        write_threshold_sweep_csv(output_dir / "threshold_sweep.csv", threshold_sweep)

    prediction_rows = build_prediction_rows(
        examples=examples,
        rows_by_label=rows_by_label,
        weights=weights,
        vote_threshold=selected_threshold,
        fallback_label=fallback_label,
        args=args,
    )
    official_payload = evaluate_with_bioasq_java(
        prediction_rows=prediction_rows, examples_by_key=examples_by_key,
        model_label="weighted-frequency-ensemble", model_dir=output_dir,
        args=args, include_per_question=True)
    aggregate = official_payload["aggregate"]
    official_by_id = {row["question_id"]: row for row in official_payload["per_question"]}
    for row in prediction_rows:
        row["score"] = official_by_id[row["question_id"]]

    score_payload = {
        "created_at": utc_now_iso(),
        "method": "weighted_frequency_ensemble",
        "prediction_inputs": {label: str(path) for label, path in prediction_paths.items()},
        "weights": weights,
        "vote_threshold": selected_threshold,
        "requested_vote_threshold": args.vote_threshold,
        "threshold_sweep": threshold_sweep,
        "fallback_label": fallback_label,
        "dataset": {
            "eval_input": [str(path) for path in eval_paths],
            "question_types": supported_question_types,
            "question_count": len(examples),
        },
        "aggregate": aggregate,
        "scoring_backend": "bioasq_java",
    }
    write_json(output_dir / "predictions.json", prediction_rows)
    write_json(output_dir / "scores.json", score_payload)
    write_json(
        output_dir / "bioasq_predictions.json",
        {
            "system": "weighted-frequency-ensemble",
            "questions": [build_bioasq_prediction_entry(row) for row in prediction_rows],
        },
    )
    print(json.dumps(score_payload["aggregate"], indent=2))
    print(f"Saved ensemble artifacts to {output_dir}")


if __name__ == "__main__":
    main()
