import contextlib
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from src.notebook_workflows import coverage_comparison as cc

ROOT = Path(__file__).resolve().parents[1]
JAR = ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"


def example(qid="q"):
    return {"question_id": qid, "question": "Which?", "snippets": [{"snippet_id": "1", "text": "An alpha example."}],
            "gold_aliases": ["alpha", "SECRET_GOLD"]}


def response(arm):
    return {"answers": [{"answer": "alpha", "relation_type": "original"}]} if arm == "expansion" else {"answer": "beta"}


def test_real_dev_has_160_unique_disjoint_questions_and_no_sft_instruction():
    data = ROOT / "data/BioASQ_factoid_sft_prepared/split_first_train90_dev10_supported_train_seed3407"
    examples = cc.load_dev(data / "dev_prepared.json", ROOT / "data/training13b.json", train_path=data / "train_questions.json")
    assert len(examples) == len({e["question_id"] for e in examples}) == 160
    for e in examples:
        assert e["gold_aliases"]
        for arm in ("expansion", "sampling"):
            user = json.loads(cc.request_payload(e, arm)["messages"][1]["content"])
            assert set(user) == {"question", "snippets"}
            assert "instruction" not in user and "output" not in user


def test_prompts_are_gold_blind_and_use_identical_context():
    e = {**example(), "instruction": "COPY_SECRET", "output": "SECRET_OUTPUT"}
    a, b = cc.request_payload(e, "expansion"), cc.request_payload(e, "sampling")
    assert a["messages"][1] == b["messages"][1]
    assert a["temperature"] == 0 and b["temperature"] == 1.2
    for payload in (a, b):
        assert all(secret not in json.dumps(payload) for secret in ("SECRET_GOLD", "COPY_SECRET", "SECRET_OUTPUT"))


def test_extractive_expansion_schema_and_literal_containment():
    e = example()
    payload = cc.request_payload(
        e,
        "expansion",
        expansion_prompt=cc.EXTRACTIVE_EXPANSION_PROMPT,
        expansion_mode="extractive",
    )
    item_schema = payload["response_format"]["json_schema"]["schema"]["properties"]["answers"]["items"]
    assert set(item_schema["properties"]) == {"answer", "snippet_id", "candidate_type"}
    valid = {"answers": [{"answer": "alpha", "snippet_id": "1", "candidate_type": "minimal_direct"}]}
    assert cc.validate_response(valid, "expansion", expansion_mode="extractive", snippets=e["snippets"]) == valid["answers"]
    invalid = {"answers": [{"answer": "an alpha", "snippet_id": "1", "candidate_type": "minimal_direct"}]}
    with pytest.raises(ValueError, match="literal substring"):
        cc.validate_response(invalid, "expansion", expansion_mode="extractive", snippets=e["snippets"])


def test_extractive_containment_repairs_citation_and_rejects_only_bad_span():
    snippets = [
        {"snippet_id": "1", "text": "An alpha example."},
        {"snippet_id": "2", "text": "A beta example."},
    ]
    answers = [
        {"answer": "alpha", "snippet_id": "2", "candidate_type": "minimal_direct"},
        {"answer": "gamma", "snippet_id": "1", "candidate_type": "canonical_surface"},
    ]
    accepted, rejected = cc.validate_extractive_containment(answers, snippets)
    assert accepted == [{**answers[0], "snippet_id": "1", "raw_position": 1,
                         "reported_snippet_id": "2", "citation_corrected": True}]
    assert rejected == [{**answers[1], "raw_position": 2,
                         "reason": "answer_not_literal_in_any_supplied_snippet"}]


def test_summary_allows_rejected_extractive_positions():
    e = example()
    candidates = [{"question_id": "q", "arm": "sampling", "position": i, "answer": "beta"}
                  for i in range(1, 11)]
    matches = {("q", "beta"): False}
    summary, rows = cc.summarize([e], candidates, matches, [])
    assert rows[0]["expansion_count"] == 0
    assert rows[0]["expansion_coverage_at10"] is False
    assert summary["methods"][0]["mean_candidates"] == 0


@pytest.mark.parametrize("value", [
    {"answers": []},
    {"answers": [{"answer": "alpha", "relation_type": "original"}] * 11},
    {"answers": [{"answer": "alpha", "relation_type": "synonym"}]},
    {"answers": [{"answer": " ", "relation_type": "original"}]},
])
def test_malformed_expansion_is_not_silently_repaired(value):
    with pytest.raises(ValueError):
        cc.validate_response(value, "expansion")


def test_sampling_slot_cache_produces_ten_independent_calls_and_resumes(tmp_path, monkeypatch):
    key = tmp_path / "key.txt"
    key.write_text("dummy", encoding="utf-8")
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": f"id{len(sent)}", "model": cc.MODEL, "usage": {"total_tokens": 5},
                        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"answer": f"a{len(sent)}"})}}]}
        return Response()

    monkeypatch.setattr("requests.post", post)
    payload = cc.request_payload(example(), "sampling")
    client = cc.ComparisonClient(tmp_path / "cache", key_file=key, max_calls=10, allow_api=True, delay=0)
    values = []
    for i in range(10):
        entry, origin = client.call(payload, {"trial": "one", "sample_index": i}, "sampling")
        values.append(entry["response"]["answer"])
        assert origin == "api"
    assert len(sent) == client.new_calls == len(set(values)) == 10
    assert all(p == payload for p in sent)  # Sampling identity is not inserted into the prompt/API payload.
    resumed = cc.ComparisonClient(tmp_path / "resumed", key_file=tmp_path / "missing-key", max_calls=0,
                                   allow_api=False, previous_cache=tmp_path / "cache")
    for i in range(10):
        entry, origin = resumed.call(payload, {"trial": "one", "sample_index": i}, "sampling")
        assert origin == "cache" and entry["response"]["answer"] == values[i]
    with pytest.raises(cc.RequestLimit):
        resumed.call(payload, {"trial": "two", "sample_index": 0}, "sampling")
    assert len(sent) == 10 and resumed.new_calls == 0


def test_failed_network_call_counts_toward_budget(tmp_path, monkeypatch):
    import requests
    key = tmp_path / "key.txt"
    key.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr("requests.post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("test timeout")))
    client = cc.ComparisonClient(tmp_path / "cache", key_file=key, max_calls=1, allow_api=True, delay=0)
    with pytest.raises(requests.Timeout):
        client.call(cc.request_payload(example(), "sampling"), {"i": 0}, "sampling")
    assert client.new_calls == 1
    with pytest.raises(cc.RequestLimit):
        client.call(cc.request_payload(example(), "sampling"), {"i": 0}, "sampling")


def test_503_retries_same_draw_and_caches_only_success(tmp_path, monkeypatch):
    import requests
    key = tmp_path / "key.txt"
    key.write_text("dummy", encoding="utf-8")
    sent, delays = [], []

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        result = requests.Response()
        result.status_code = 503 if len(sent) == 1 else 200
        result.headers["Retry-After"] = "3"
        result._content = json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"answer": "alpha"})}}]}).encode()
        return result

    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr(cc.time, "sleep", delays.append)
    client = cc.ComparisonClient(tmp_path / "cache", key_file=key, max_calls=2, allow_api=True, delay=0)
    payload, slot = cc.request_payload(example(), "sampling"), {"sample_index": 4}
    entry, origin = client.call(payload, slot, "sampling")
    assert origin == "api" and entry["request_attempts"] == 2
    assert sent == [payload, payload] and delays == [3]
    assert client.new_calls == 2 and client.retry_count == 1
    assert client.call(payload, slot, "sampling")[1] == "cache"
    assert len(sent) == 2 and len(list((tmp_path / "cache").iterdir())) == 1


@pytest.mark.parametrize("status_code,max_calls,max_retries,expected", [
    (503, 10, 2, 3), (503, 2, 5, 2), (401, 10, 3, 1), (400, 10, 3, 1),
])
def test_transport_retries_are_bounded_and_never_retry_auth_errors(
    tmp_path, monkeypatch, status_code, max_calls, max_retries, expected
):
    import requests
    key = tmp_path / "key.txt"
    key.write_text("dummy", encoding="utf-8")
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        result = requests.Response()
        result.status_code = status_code
        return result
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr(cc.time, "sleep", lambda seconds: None)
    client = cc.ComparisonClient(tmp_path / "cache", key_file=key, max_calls=max_calls, allow_api=True,
                                 delay=0, max_transport_retries=max_retries)
    with pytest.raises(requests.HTTPError):
        client.call(cc.request_payload(example(), "sampling"), {"sample_index": 0}, "sampling")
    assert len(calls) == client.new_calls == expected


def test_invalid_answer_is_not_resampled(tmp_path, monkeypatch):
    import requests
    key = tmp_path / "key.txt"
    key.write_text("dummy", encoding="utf-8")
    def post(*args, **kwargs):
        result = requests.Response()
        result.status_code = 200
        result._content = json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": '{"answer":""}'}}]}).encode()
        return result
    monkeypatch.setattr("requests.post", post)
    client = cc.ComparisonClient(tmp_path / "cache", key_file=key, max_calls=10, allow_api=True, delay=0)
    with pytest.raises(ValueError, match="nonempty"):
        client.call(cc.request_payload(example(), "sampling"), {}, "sampling")
    assert client.new_calls == 1 and client.retry_count == 0


def test_runner_has_one_expansion_and_ten_sampling_calls_per_question(tmp_path, monkeypatch):
    calls = []
    def fake_call(self, payload, slot, arm):
        calls.append(slot)
        self.new_calls += 1
        return {"response": response(arm), "usage": {"total_tokens": 3}}, "api"
    monkeypatch.setattr(cc.ComparisonClient, "call", fake_call)
    examples = [example("q1"), example("q2")]
    out = cc.run_comparison(examples, tmp_path, api_key_file="unused", run=True, allow_api=True)
    assert len(calls) == 22
    assert calls[0]["arm"] == "expansion" and calls[11]["arm"] == "sampling"
    assert json.loads((out / "status.json").read_text())["status"] == "complete"
    candidates = cc.read_jsonl(out / "candidates.jsonl")
    matches = {(c["question_id"], c["answer"]): c["answer"] == "alpha" for c in candidates}
    summary, _ = cc.summarize(examples, candidates, matches, cc.read_jsonl(out / "requests.jsonl"))
    assert summary["paired_outcomes"] == {"expansion_only": 2}
    assert summary["methods"][1]["mean_candidates"] == 10
    assert summary["methods"][1]["mean_unique_candidates"] == 1
    assert summary["coverage_at10_difference_percentage_points"] == 100


def test_extractive_runner_audits_bad_span_without_aborting(tmp_path, monkeypatch):
    calls = []

    def fake_call(self, payload, slot, arm, **kwargs):
        calls.append(slot)
        self.new_calls += 1
        if arm == "expansion":
            value = {"answers": [
                {"answer": "alpha", "snippet_id": "wrong", "candidate_type": "minimal_direct"},
                {"answer": "invented", "snippet_id": "1", "candidate_type": "alternative_evidence"},
            ]}
        else:
            value = {"answer": "beta"}
        return {"response": value, "usage": {"total_tokens": 3}}, "api"

    monkeypatch.setattr(cc.ComparisonClient, "call", fake_call)
    out = cc.run_comparison(
        [example()], tmp_path, expansion_mode="extractive", api_key_file="unused", run=True, allow_api=True
    )
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["invalid_extractive_candidates"] == 1
    assert status["corrected_extractive_citations"] == 1
    expansion = [row for row in cc.read_jsonl(out / "candidates.jsonl") if row["arm"] == "expansion"]
    assert expansion[0]["answer"] == "alpha" and expansion[0]["snippet_id"] == "1"
    assert expansion[0]["position"] == 1 and expansion[0]["reported_snippet_id"] == "wrong"
    rejected = cc.read_jsonl(out / "invalid_candidates.jsonl")
    assert rejected[0]["answer"] == "invented" and rejected[0]["raw_position"] == 2


def test_incomplete_run_preserved_but_not_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(cc.ComparisonClient, "call", lambda *a: (_ for _ in ()).throw(cc.RequestLimit("limit")))
    out = cc.run_comparison([example()], tmp_path, api_key_file="unused", run=True, allow_api=True)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "incomplete" and status["failed_slot"]["arm"] == "expansion"
    with pytest.raises(ValueError, match="incomplete") as caught:
        cc.analyze_run(out, jar_path=JAR)
    assert "Cause: RequestLimit: limit" in str(caught.value)
    assert repr(str(out / "cache")) in str(caught.value)


@pytest.mark.skipif(not shutil.which("java") or not shutil.which("javac"), reason="Official scorer requires JDK")
def test_official_matcher_scores_candidate_ten_and_all_aliases(tmp_path):
    e = {**example(), "gold_aliases": ["alpha", "alternate"]}
    candidates = [{"question_id": "q", "arm": arm, "position": i,
                   "answer": "ALTERNATE" if arm == "expansion" and i == 10 else ">25"}
                  for arm in ("expansion", "sampling") for i in range(1, 11)]
    matches = cc.official_candidate_matches([e], candidates, tmp_path, jar_path=JAR)
    assert matches[("q", "ALTERNATE")] is True and matches[("q", ">25")] is False
    summary, rows = cc.summarize([e], candidates, matches, [])
    assert rows[0]["expansion_coverage_at10"] is True
    assert rows[0]["expansion_coverage_at5"] is False
    assert summary["paired_outcomes"] == {"expansion_only": 1}


def test_notebook_preview_runs_all_160_without_network_or_artifacts(monkeypatch):
    from scripts.build_coverage_comparison_notebook import build

    monkeypatch.chdir(ROOT)
    # The saved notebook contains the user's live flags, outputs and cache path.
    # Exercise the generator's preview defaults without executing a paid run.
    saved = build()
    ns = {}
    with patch.object(cc, "ComparisonClient", side_effect=AssertionError("Preview created client")), \
         patch("requests.post", side_effect=AssertionError("Preview called API")), \
         contextlib.redirect_stdout(io.StringIO()):
        for i, cell in enumerate(saved["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"notebook10:{i}", "exec"), ns)
    assert ns["RUN"] is False and ns["ALLOW_API"] is False
    assert ns["RUN_DIR"] is None and len(ns["EXAMPLES"]) == 160
    assert ns["PREVIEW"]["required_requests"] == 1760
