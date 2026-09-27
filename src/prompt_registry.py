from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.model_registry import slugify, utc_now_iso


PROMPT_REGISTRY_VERSION = 1
SUPPORTED_QUESTION_TYPES = ("summary", "factoid", "list", "yesno")


def empty_prompt_registry() -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "version": PROMPT_REGISTRY_VERSION,
        "updated_at": now,
        "aliases": {},
        "prompts": {},
    }


def _extract_instructions(payload: Mapping[str, Any]) -> Dict[str, str]:
    for key in ("instructions", "question_instructions", "templates"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            instructions = {
                question_type: str(candidate[question_type]).strip()
                for question_type in SUPPORTED_QUESTION_TYPES
                if str(candidate.get(question_type, "")).strip()
            }
            if instructions:
                return instructions

    direct = {
        question_type: str(payload[question_type]).strip()
        for question_type in SUPPORTED_QUESTION_TYPES
        if str(payload.get(question_type, "")).strip()
    }
    if direct:
        return direct

    raise ValueError("Prompt entry is missing per-question-type instructions.")


def _normalize_prompt_entry(prompt_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    instructions = _extract_instructions(payload)
    normalized = dict(payload)
    normalized["prompt_id"] = str(normalized.get("prompt_id") or prompt_id)
    normalized["name"] = str(normalized.get("name") or normalized["prompt_id"])
    normalized["instructions"] = instructions
    return normalized


def load_prompt_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return empty_prompt_registry()

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt registry must be a JSON object: {path}")

    if "prompts" not in data:
        inline_prompt = _normalize_prompt_entry("inline", data)
        return {
            "version": data.get("version", PROMPT_REGISTRY_VERSION),
            "updated_at": data.get("updated_at", utc_now_iso()),
            "aliases": {"default": {"prompt_id": "inline"}},
            "prompts": {"inline": inline_prompt},
        }

    registry = empty_prompt_registry()
    registry["version"] = data.get("version", PROMPT_REGISTRY_VERSION)
    registry["updated_at"] = data.get("updated_at", utc_now_iso())

    aliases = data.get("aliases", {})
    if isinstance(aliases, dict):
        registry["aliases"] = aliases

    prompts = data.get("prompts", {})
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt registry 'prompts' must be an object: {path}")

    normalized_prompts: Dict[str, Any] = {}
    for prompt_key, payload in prompts.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"Prompt entry '{prompt_key}' must be an object: {path}")
        prompt_id = str(payload.get("prompt_id") or prompt_key)
        normalized_prompts[prompt_id] = _normalize_prompt_entry(prompt_id, payload)
    registry["prompts"] = normalized_prompts
    return registry


def _resolve_alias_target(alias_payload: Any) -> Optional[str]:
    if isinstance(alias_payload, str) and alias_payload.strip():
        return alias_payload.strip()
    if isinstance(alias_payload, Mapping):
        candidate = alias_payload.get("prompt_id") or alias_payload.get("alias")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _registry_prompt_ids(registry: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    prompts = registry.get("prompts", {})
    if not isinstance(prompts, dict):
        return {}
    return {str(prompt_id): value for prompt_id, value in prompts.items() if isinstance(value, dict)}


def resolve_prompt_bundle(
    registry_path: Optional[Path],
    prompt_ref: Optional[str],
    fallback_instructions: Mapping[str, str],
) -> Dict[str, Any]:
    fallback_bundle = {
        "prompt_id": "builtin-default",
        "name": "builtin-default",
        "instructions": {
            question_type: str(fallback_instructions[question_type]).strip()
            for question_type in SUPPORTED_QUESTION_TYPES
            if str(fallback_instructions.get(question_type, "")).strip()
        },
        "source": "builtin",
        "registry_path": None,
    }

    if registry_path is None:
        if prompt_ref:
            raise FileNotFoundError("A prompt reference was provided, but no prompt registry path was set.")
        return fallback_bundle

    registry = load_prompt_registry(registry_path)
    prompts = _registry_prompt_ids(registry)
    aliases = registry.get("aliases", {})
    normalized_ref = slugify(prompt_ref, fallback="default") if prompt_ref else "default"

    prompt_id = None
    if normalized_ref in prompts:
        prompt_id = normalized_ref
    elif isinstance(aliases, dict) and normalized_ref in aliases:
        prompt_id = _resolve_alias_target(aliases[normalized_ref])
    elif prompt_ref and prompt_ref in prompts:
        prompt_id = prompt_ref

    if prompt_id is None:
        if prompt_ref:
            raise KeyError(f"Unknown prompt reference: {prompt_ref}")
        if "default" in aliases:
            prompt_id = _resolve_alias_target(aliases["default"])
        if prompt_id is None and prompts:
            prompt_id = next(iter(sorted(prompts)))

    if prompt_id is None:
        return fallback_bundle

    prompt = prompts.get(prompt_id)
    if prompt is None:
        raise KeyError(f"Prompt registry alias does not resolve to a known prompt: {prompt_id}")

    instructions = dict(fallback_bundle["instructions"])
    instructions.update(prompt.get("instructions", {}))
    bundle = dict(prompt)
    bundle["instructions"] = instructions
    bundle["source"] = "registry"
    bundle["registry_path"] = str(registry_path)
    return bundle
