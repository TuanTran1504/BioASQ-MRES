import importlib.util
import json
from pathlib import Path

from src.notebook_workflows.coverage_comparison import EXTRACTIVE_EXPANSION_PROMPT


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "gadi_sft_8b_starter"


def load_runner():
    path = BUNDLE / "scripts/run_extractive_expansion_8b.py"
    spec = importlib.util.spec_from_file_location("gadi_expansion_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gadi_prompt_matches_local_expansion_protocol():
    prompt = (BUNDLE / "prompts/extractive_expansion_v2.txt").read_text(encoding="utf-8")
    config = json.loads((BUNDLE / "configs/extractive_expansion_8b.json").read_text())
    assert prompt.strip() == EXTRACTIVE_EXPANSION_PROMPT.strip()
    assert config["prompt"] == "prompts/extractive_expansion_v2.txt"
    assert config["prompt_version"] == "extractive-expansion-v2"
    assert config["max_seq_length"] == 6144
    assert config["max_new_tokens"] == 512
    assert config["temperature"] == 0.0
    assert config["require_all_snippets"] is True


def test_gadi_parser_repairs_citation_and_audits_bad_span():
    runner = load_runner()
    response = json.dumps({
        "answers": [
            {"answer": "alpha", "snippet_id": "2", "candidate_type": "minimal_direct"},
            {"answer": "invented", "snippet_id": "1", "candidate_type": "alternative_evidence"},
        ]
    })
    accepted, rejected, issues, compliant = runner.parse_extractive_response(
        response,
        [
            {"snippet_id": "1", "text": "An alpha example."},
            {"snippet_id": "2", "text": "A beta example."},
        ],
    )
    assert accepted[0]["snippet_id"] == "1"
    assert accepted[0]["reported_snippet_id"] == "2"
    assert accepted[0]["citation_corrected"] is True
    assert rejected[0]["reason"] == "answer_not_literal_in_any_supplied_snippet"
    assert "candidate_1_citation_corrected" in issues
    assert compliant is True


def test_gadi_parser_salvages_literal_candidate_with_schema_issue():
    runner = load_runner()
    response = json.dumps({
        "answers": [
            {"answer": "alpha", "snippet_id": "1", "candidate_type": "made_up_type", "extra": 1}
        ],
        "comment": "not allowed",
    })
    accepted, rejected, issues, compliant = runner.parse_extractive_response(
        response, [{"snippet_id": "1", "text": "An alpha example."}]
    )
    assert [row["answer"] for row in accepted] == ["alpha"]
    assert rejected == []
    assert "unexpected_top_level_fields" in issues
    assert "candidate_1_invalid_candidate_type" in issues
    assert compliant is False


def test_gadi_parser_recovers_complete_candidates_from_truncated_json():
    runner = load_runner()
    response = (
        '{"answers":['
        '{"answer":"alpha","snippet_id":"1","candidate_type":"minimal_direct"},'
        '{"answer":"beta","snippet_id":"2","candidate_type":"minimal_direct"'
    )
    accepted, rejected, issues, compliant = runner.parse_extractive_response(
        response,
        [
            {"snippet_id": "1", "text": "An alpha example."},
            {"snippet_id": "2", "text": "A beta example."},
        ],
    )
    assert [row["answer"] for row in accepted] == ["alpha"]
    assert rejected == []
    assert "incomplete_top_level_json_recovered" in issues
    assert compliant is False


def test_gadi_parser_deduplicates_before_applying_unique_candidate_limit():
    runner = load_runner()
    answers = [
        {"answer": "alpha", "snippet_id": "1", "candidate_type": "minimal_direct"}
        for _ in range(11)
    ]
    answers.append({"answer": "beta", "snippet_id": "2", "candidate_type": "minimal_direct"})
    accepted, rejected, issues, compliant = runner.parse_extractive_response(
        json.dumps({"answers": answers}),
        [
            {"snippet_id": "1", "text": "An alpha example."},
            {"snippet_id": "2", "text": "A beta example."},
        ],
    )
    assert [row["answer"] for row in accepted] == ["alpha", "beta"]
    assert sum(row["reason"] == "duplicate_answer_surface" for row in rejected) == 10
    assert "answer_count_out_of_range" in issues
    assert compliant is False
