from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class EvalExample:
    question_id: str
    question_type: str
    body: str
    instruction: str
    resources: Tuple[str, ...]
    gold_output: str
    source_path: str
    raw_question: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ModelSpec:
    ref: str
    label: str
    source: str
    load_target: str
    run_id: Optional[str] = None
    alias: Optional[str] = None
    base_model: Optional[str] = None
    adapter_dir: Optional[str] = None
    chat_template: Optional[str] = None
    prompt_format: Optional[str] = None
