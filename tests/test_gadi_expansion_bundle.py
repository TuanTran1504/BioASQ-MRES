import importlib.util
import json
from pathlib import Path

from src.notebook_workflows.coverage_comparison import EXPANSION_PROMPT, EXTRACTIVE_EXPANSION_PROMPT


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


def test_gadi_equivalent_prompt_matches_original_gpt_protocol():
    prompt = (BUNDLE / "prompts/equivalent_expansion_v1.txt").read_text(encoding="utf-8")
    config = json.loads((BUNDLE / "configs/equivalent_expansion_8b.json").read_text())
    assert prompt.strip() == EXPANSION_PROMPT.strip()
    assert config["prompt"] == "prompts/equivalent_expansion_v1.txt"
    assert config["prompt_version"] == "equivalent-expansion-v1"
    assert config["response_mode"] == "equivalent"
    assert config["max_seq_length"] == 6144
    assert config["max_new_tokens"] == 512
    assert config["temperature"] == 0.0
    assert config["require_all_snippets"] is True


def test_multi_model_configs_keep_the_same_expansion_protocol():
    expected = {
        "equivalent_expansion_qwen3_8b.json": (
            "unsloth/Qwen3-8B-unsloth-bnb-4bit",
            "fast_language_model",
            {"enable_thinking": False},
        ),
        "equivalent_expansion_ministral3_8b.json": (
            "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit",
            "fast_model",
            {},
        ),
        "equivalent_expansion_gemma3_27b.json": (
            "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
            "fast_model",
            {},
        ),
    }
    for filename, (model_name, model_loader, template_kwargs) in expected.items():
        config = json.loads((BUNDLE / "configs" / filename).read_text())
        assert config["model_name"] == model_name
        assert config["model_loader"] == model_loader
        assert config["chat_template_kwargs"] == template_kwargs
        assert config["prompt"] == "prompts/equivalent_expansion_v1.txt"
        assert config["response_mode"] == "equivalent"
        assert config["max_seq_length"] == 6144
        assert config["max_new_tokens"] == 512
        assert config["temperature"] == 0.0


def test_render_prompt_passes_model_specific_chat_template_options():
    runner = load_runner()

    class RecordingTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    tokenizer = RecordingTokenizer()
    rendered = runner.render_prompt(
        tokenizer,
        "system",
        {
            "question": "Question?",
            "snippets": [{"snippet_id": "1", "text": "Evidence."}],
        },
        chat_template_kwargs={"enable_thinking": False},
    )
    assert rendered == "rendered"
    assert tokenizer.kwargs["tokenize"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["enable_thinking"] is False


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


def test_gadi_equivalent_parser_accepts_original_and_relations():
    runner = load_runner()
    response = json.dumps({
        "answers": [
            {"answer": "alpha", "relation_type": "original"},
            {"answer": "A", "relation_type": "abbreviation_expansion"},
        ]
    })
    accepted, rejected, issues, compliant = runner.parse_equivalent_response(response)
    assert [row["answer"] for row in accepted] == ["alpha", "A"]
    assert [row["relation_type"] for row in accepted] == ["original", "abbreviation_expansion"]
    assert rejected == []
    assert issues == []
    assert compliant is True


def test_gadi_equivalent_parser_recovers_and_deduplicates_truncated_json():
    runner = load_runner()
    response = (
        '{"answers":['
        '{"answer":"alpha","relation_type":"original"},'
        '{"answer":"ALPHA","relation_type":"synonym"},'
        '{"answer":"beta","relation_type":"synonym"'
    )
    accepted, rejected, issues, compliant = runner.parse_equivalent_response(response)
    assert [row["answer"] for row in accepted] == ["alpha"]
    assert rejected[0]["reason"] == "duplicate_answer_surface"
    assert "incomplete_top_level_json_recovered" in issues
    assert compliant is False
