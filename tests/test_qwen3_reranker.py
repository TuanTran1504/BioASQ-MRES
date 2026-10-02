import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen3_reranker",
    ROOT / "gadi_sft_8b_starter/scripts/run_qwen3_reranker.py",
)
QR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QR)


def example(question_id="q1"):
    return {
        "question_id": question_id,
        "question": "Which protein is inhibited by drug X?",
        "snippets": [
            {"id": "1", "text": "First supplied snippet."},
            {"id": "2", "text": "Drug X strongly inhibits kinase ABC in cells."},
            {"id": "3", "text": "Final supplied snippet."},
        ],
    }


def candidate(qid, answer, label, sources=None, rank=1):
    sources = sources or ["gpt_equivalent"]
    return {
        "question_id": qid,
        "answer": answer,
        "label": label,
        "sources": sources,
        "source_ranks": {source: rank for source in sources},
    }


def test_document_contains_every_snippet_in_original_order():
    document = QR.format_document(example())
    assert "First supplied snippet." in document
    assert "Drug X strongly inhibits kinase ABC" in document
    assert "Final supplied snippet." in document
    assert document.index("First supplied") < document.index("Drug X") < document.index("Final supplied")


def test_reranker_body_contains_question_candidate_and_all_evidence():
    body = QR.format_reranker_body(example(), "kinase ABC")
    assert "Biomedical factoid question:" in body
    assert "Candidate answer: kinase ABC" in body
    assert "[Snippet 1]" in body
    assert "[Snippet 2]" in body
    assert "[Snippet 3]" in body


def test_training_pairs_exclude_all_negative_questions_and_cap_negatives():
    by_question = {
        "q1": [
            candidate("q1", "kinase ABC", 1),
            candidate("q1", "kinase XYZ", 0),
            candidate("q1", "protein DEF", 0),
        ],
        "q2": [candidate("q2", "wrong", 0)],
    }
    examples = {"q1": example("q1"), "q2": example("q2")}
    pairs = QR.build_training_pairs(
        ["q1", "q2"],
        by_question,
        examples,
        max_negatives_per_positive=1,
    )
    assert len(pairs) == 1
    assert pairs[0]["question_id"] == "q1"
    assert pairs[0]["positive"]["label"] == 1
    assert pairs[0]["negative"]["label"] == 0


def test_question_folds_are_disjoint_and_exhaustive():
    question_ids = [f"q{index}" for index in range(20)]
    by_question = {
        qid: [candidate(qid, f"answer {qid}", int(index < 12))]
        for index, qid in enumerate(question_ids)
    }
    folds = QR.make_question_folds(question_ids, by_question, folds=5, seed=3407)
    observed = []
    for train_ids, test_ids in folds:
        assert set(train_ids).isdisjoint(test_ids)
        assert set(train_ids) | set(test_ids) == set(question_ids)
        observed.extend(test_ids)
    assert sorted(observed) == sorted(question_ids)
