"""OpenAI candidate generation for the shared BioASQ evaluator.

The implementation deliberately uses the same ``EvalExample`` messages and
prediction schema as local-model evaluation. Responses are cached by the full
request payload, so interrupted paid runs can reuse completed calls.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import clean_text
from .eval_dataset import build_messages
from .eval_models import aggregate_generated_answers
from .eval_types import EvalExample


def read_api_key(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    value = next((line for line in lines if line and not line.startswith("#")), "")
    if value.startswith("OPENAI_API_KEY="):
        value = value.split("=", 1)[1].strip()
    value = value.strip().strip('"').strip("'")
    if not value:
        raise ValueError(f"No API key found in {path}")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAICandidateGenerator:
    def __init__(
        self,
        *,
        model: str,
        args: Any,
        cache_dir: Path,
        budget_state: dict[str, int],
        read_cache_dir: Path | None = None,
    ):
        self.model = str(model)
        self.args = args
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.read_cache_dir = Path(read_cache_dir) if read_cache_dir else None
        self.budget_state = budget_state
        self.api_key: str | None = None

    def _request_payload(self, example: EvalExample, sample_index: int) -> dict[str, Any]:
        do_sample = bool(getattr(self.args, "do_sample", False))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(example),
            "max_tokens": int(self.args.max_new_tokens),
            "temperature": float(self.args.temperature) if do_sample else 0.0,
            "top_p": float(self.args.top_p) if do_sample else 1.0,
            "sample_index": int(sample_index),
        }
        return payload

    def _call(self, payload: Mapping[str, Any]) -> str:
        import requests

        maximum = int(getattr(self.args, "max_new_api_calls", 0) or 0)
        if self.budget_state["new_api_calls"] >= maximum:
            raise RuntimeError(
                f"OpenAI candidate call budget exhausted ({maximum}). Increase "
                "max_new_api_calls or reuse the saved cache."
            )
        if self.api_key is None:
            self.api_key = read_api_key(Path(self.args.api_key_file))

        request_payload = {key: value for key, value in payload.items() if key != "sample_index"}
        retries = max(0, int(getattr(self.args, "api_max_retries", 2) or 0))
        for attempt in range(retries + 1):
            if self.budget_state["new_api_calls"] >= maximum:
                raise RuntimeError(
                    f"OpenAI candidate call budget exhausted ({maximum}) during retries. "
                    "Increase max_new_api_calls or reuse the saved cache."
                )
            if float(getattr(self.args, "api_request_delay_seconds", 0.0) or 0.0) > 0:
                time.sleep(float(self.args.api_request_delay_seconds))
            self.budget_state["new_api_calls"] += 1
            try:
                response = requests.post(
                    self.args.openai_endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=int(self.args.api_timeout_seconds),
                )
                response.raise_for_status()
                return clean_text(response.json()["choices"][0]["message"]["content"])
            except Exception:
                if attempt >= retries:
                    raise
                self.budget_state["retry_count"] += 1
                time.sleep(min(60.0, (2**attempt) + random.random()))
        raise AssertionError("unreachable")

    def generate(self, example: EvalExample) -> tuple[str, list[str], list[dict[str, Any]]]:
        sample_count = max(1, int(getattr(self.args, "num_generations", 1) or 1))
        samples: list[str] = []
        telemetry: list[dict[str, Any]] = []
        for sample_index in range(sample_count):
            payload = self._request_payload(example, sample_index)
            cache_path = self.cache_dir / f"{_digest(payload)}.json"
            reusable = self.read_cache_dir / cache_path.name if self.read_cache_dir else None
            existing = cache_path if cache_path.is_file() else reusable if reusable and reusable.is_file() else None
            if existing:
                cached = json.loads(existing.read_text(encoding="utf-8"))
                response_text = clean_text(cached.get("response"))
                if existing != cache_path:
                    cache_path.write_text(
                        json.dumps(cached, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                origin = "cache"
            else:
                response_text = self._call(payload)
                cache_path.write_text(
                    json.dumps(
                        {"request": payload, "response": response_text},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                origin = "api"
            samples.append(response_text)
            telemetry.append(
                {
                    "backend": "openai",
                    "model": self.model,
                    "sample_index": sample_index,
                    "origin": origin,
                    "prompt_truncated": False,
                    "prompt_token_count": None,
                    "effective_prompt_token_count": None,
                    "truncated_token_count": 0,
                    "max_seq_length": None,
                }
            )
        prediction = (
            samples[0]
            if sample_count == 1
            else aggregate_generated_answers(samples, question_type=example.question_type, args=self.args)
        )
        return prediction, samples, telemetry
