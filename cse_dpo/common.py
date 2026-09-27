from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, List, Mapping, Sequence

from src.utility.data import clean_text, truncate_text


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_records(path: Path) -> List[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Expected object rows in {path}:{line_number}")
                rows.append(payload)
        return rows

    payload = read_json(path)
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"Expected a JSON list of objects: {path}")
        return list(payload)

    if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        rows = payload["records"]
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"Expected 'records' to contain objects: {path}")
        return list(rows)

    raise ValueError(f"Unsupported record container: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def summarize_numeric(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(items),
        "mean": mean(items),
        "median": median(items),
        "min": min(items),
        "max": max(items),
    }


def flatten_counter(counter_like: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in sorted(counter_like.items(), key=lambda item: str(item[0]))}


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def truncate_multiline(values: Sequence[str], max_chars_per_item: int = 500) -> list[str]:
    items: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        items.append(truncate_text(cleaned, max_chars=max_chars_per_item))
    return items


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
