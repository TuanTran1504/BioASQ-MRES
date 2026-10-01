import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.notebook_workflows import answer_variants as av


def example(**overrides):
    return {"question_id": "q", "question": "Which receptor?", "initial_answer": "ABC receptor",
            "snippets": [{"snippet_id": "1", "text": "ABC receptor means alpha beta receptor."}],
            "gold_aliases": ["ABC receptor", "SECRET_GOLD_ALIAS"], "seed_mode": "gold", **overrides}


def generated(answer="alpha beta receptor"):
    return {"variants": [{"answer": answer, "evidence_ids": ["1"], "reason": "Explicit expansion."}],
            "abstention_reason": ""}


def test_request_whitelist_and_ranker_do_not_reveal_gold_or_seed():
    e = example(judge_rationale="SECRET_RATIONALE", initial_class="C3")
    for candidate in (None, generated()["variants"][0]):
        request = av.request_payload(e, "expand_abbreviation", model="test", candidate=candidate)
        assert "SECRET_GOLD_ALIAS" not in json.dumps(request)
        assert "SECRET_RATIONALE" not in json.dumps(request)
    context = json.loads(av.ranking_input(e))
    assert set(context) == {"question", "snippets"}
    assert "initial_answer" not in context


def test_selection_keeps_full_gold_and_one_question_per_example(tmp_path):
    raw = tmp_path / "raw.json"
    eligible = tmp_path / "eligible.json"
    raw.write_text(json.dumps({"questions": [
        {"id": "q", "type": "factoid", "body": "Which?", "exact_answer": [["a", "b"]],
         "snippets": [{"text": "a means b"}]},
        {"id": "heldout", "type": "factoid", "body": "Which?", "exact_answer": ["c"],
         "snippets": [{"text": "c"}]},
    ]}), encoding="utf-8")
    eligible.write_text('[{"id":"q"}]', encoding="utf-8")
    rows = av.load_examples(raw, eligible)
    assert len(rows) == 1 and rows[0]["gold_aliases"] == ["a", "b"]


def test_preview_never_writes_or_creates_client(tmp_path):
    with patch.object(av, "CachedClient", side_effect=AssertionError("preview created client")):
        preview = av.run_expansion([example()], tmp_path / "outputs", operations=list(av.TRANSFORMS),
                                   generator_model="test", verifier_model="test", api_key_file="absent",
                                   max_new_calls=21)
    assert preview["maximum_requests_without_retries"] == 21
    assert not (tmp_path / "outputs").exists()


def test_invalid_evidence_and_unexplained_abstention_rejected():
    value = generated()
    value["variants"][0]["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError, match="snippet IDs"):
        av.validate_generation(value, example(), 2)
    with pytest.raises(ValueError, match="Abstention"):
        av.validate_generation({"variants": [], "abstention_reason": ""}, example(), 2)
    av.validate_generation({"variants": [], "abstention_reason": "No supported alternative"}, example(), 2)


def test_cache_only_reuse_and_hard_budget(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    payload = av.request_payload(example(), "synonym", model="test")
    av.write_json(previous / (av.digest(payload) + ".json"), {"request": payload, "response": generated()})
    client = av.CachedClient(tmp_path / "cache", api_key_file=tmp_path / "absent", max_calls=0,
                             previous_cache=previous)
    with patch("requests.post", side_effect=AssertionError("Unexpected network")):
        value, origin = client.call(payload, lambda v: av.validate_generation(v, example(), 2))
        assert origin == "cache" and value == generated() and client.new_calls == 0
        with pytest.raises(av.BudgetExhausted, match="ALLOW_API"):
            client.call({**payload, "model": "different"}, lambda v: None)
        client.allow_api = True
        with pytest.raises(av.BudgetExhausted, match="MAX_NEW_API_CALLS"):
            client.call({**payload, "model": "different"}, lambda v: None)


@pytest.mark.parametrize("equivalent,expected_count", [(True, 2), (False, 1)])
def test_pipeline_verifies_and_exports_without_gold_features(tmp_path, monkeypatch, equivalent, expected_count):
    def fake_call(self, payload, validate):
        name = payload["response_format"]["json_schema"]["name"]
        value = generated() if name == "answer_variants" else {
            "equivalent": equivalent, "relation_valid": True, "reason": "Reviewed."}
        validate(value)
        self.new_calls += 1
        return value, "test"

    monkeypatch.setattr(av.CachedClient, "call", fake_call)
    out = av.run_expansion([example()], tmp_path, operations=["expand_abbreviation"],
                           generator_model="test", verifier_model="test", api_key_file="absent",
                           max_new_calls=2, run=True, allow_api=True)
    assert json.loads((out / "status.json").read_text())["status"] == "complete"
    candidates = av.read_jsonl(out / "candidates.jsonl")
    assert len(candidates) == expected_count
    ranked = av.rank_candidates([example()], candidates)
    summary, labeled = av.evaluate_and_export([example()], ranked, out)
    assert summary["metrics"]["oracle_match"] == 1
    exported = av.read_jsonl(out / "reranker_train.jsonl") + av.read_jsonl(out / "reranker_validation.jsonl")
    assert len(exported) == expected_count
    assert all(set(r) == {"question_id", "context", "candidate", "label"} for r in exported)
    assert all("initial_answer" not in r["context"] and "SECRET_GOLD_ALIAS" not in r["context"] for r in exported)
    if equivalent:
        assert any(r["diagnostic_class"] == "C2" for r in labeled)


def test_budget_interruption_preserves_original_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(av.CachedClient, "call", lambda *args: (_ for _ in ()).throw(av.BudgetExhausted("limit")))
    out = av.run_expansion([example()], tmp_path, operations=["synonym"], generator_model="test",
                           verifier_model="test", api_key_file="absent", max_new_calls=0,
                           run=True, allow_api=True)
    assert json.loads((out / "status.json").read_text())["status"] == "incomplete"
    assert av.read_jsonl(out / "candidates.jsonl")[0]["operation"] == "original"


def test_prediction_equivalence_does_not_imply_gold_equivalence(tmp_path):
    e = example(initial_answer="wrong entity", seed_mode="predictions")
    candidates = [{"question_id": "q", "answer": "wrong synonym", "operation": "synonym", "verification": "model_verified"}]
    ranked = av.rank_candidates([e], candidates)
    _, labeled = av.evaluate_and_export([e], ranked, tmp_path)
    assert labeled[0]["diagnostic_class"] == "unverified"
    e.pop("gold_aliases")
    summary, _ = av.evaluate_and_export([e], ranked, tmp_path)
    assert summary["scored_questions"] == 0 and summary["metrics"]["oracle_match"] is None
    assert not av.read_jsonl(tmp_path / "reranker_train.jsonl")


def test_ranker_receives_no_provenance_and_ties_do_not_prefer_seed():
    candidates = [
        {"question_id": "q", "answer": "z", "operation": "original"},
        {"question_id": "q", "answer": "a", "operation": "synonym"},
    ]
    def scorer(pairs):
        assert all(set(json.loads(p[0])) == {"question", "snippets"} for p in pairs)
        return [0, 0]
    assert av.rank_candidates([example()], candidates, scorer)[0]["answer"] == "a"


def test_notebook_preview_executes_without_writes_or_network(monkeypatch):
    from scripts.build_answer_variants_notebook import ROOT, build

    monkeypatch.chdir(ROOT)
    saved = json.loads((ROOT / "notebooks/09_controlled_answer_variants.ipynb").read_text(encoding="utf-8"))
    assert saved == build()
    with patch.object(av, "CachedClient", side_effect=AssertionError("Preview started execution")), \
         patch("requests.post", side_effect=AssertionError("Preview called API")), \
         contextlib.redirect_stdout(io.StringIO()):
        namespace = {}
        for index, cell in enumerate(saved["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"notebook09:{index}", "exec"), namespace)
    assert namespace["RUN"] is False and namespace["ALLOW_API"] is False
    assert namespace["RUN_DIR"] is None and len(namespace["EXAMPLES"]) == 10
