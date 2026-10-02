import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neural_reranker",
    ROOT / "gadi_sft_8b_starter/scripts/train_neural_cross_encoder_reranker.py",
)
NR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NR)


def example():
    return {
        "question_id": "q1",
        "question": "Which protein is inhibited by drug X?",
        "snippets": [
            {"id": "1", "text": "Unrelated background information."},
            {"id": "2", "text": "Drug X strongly inhibits kinase ABC in cells."},
            {"id": "3", "text": "Drug X is used experimentally."},
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


def test_evidence_selection_prioritizes_candidate_support():
    selected = NR.select_evidence(example(), "kinase ABC", max_snippets=1)
    assert selected[0]["id"] == "2"


def test_model_input_keeps_question_candidate_separate_from_evidence():
    first, second = NR.make_model_input(example(), "kinase ABC", max_snippets=2)
    assert "Question:" in first
    assert "Candidate answer: kinase ABC" in first
    assert "Evidence:" in second
    assert "kinase ABC" in second


def test_training_slates_exclude_questions_without_positive_candidate():
    by_question = {
        "q1": [candidate("q1", "kinase ABC", 1), candidate("q1", "kinase XYZ", 0)],
        "q2": [candidate("q2", "wrong", 0)],
    }
    examples = {
        "q1": example(),
        "q2": {**example(), "question_id": "q2"},
    }
    slates = NR.build_training_slates(
        ["q1", "q2"], by_question, examples, max_negatives=3
    )
    assert [slate["question_id"] for slate in slates] == ["q1"]
    assert slates[0]["positive_count"] == 1
    assert slates[0]["negative_count"] == 1


def test_question_folds_are_disjoint_and_exhaustive():
    question_ids = [f"q{index}" for index in range(20)]
    by_question = {
        qid: [candidate(qid, f"answer {qid}", int(index < 12))]
        for index, qid in enumerate(question_ids)
    }
    folds = NR.make_question_folds(question_ids, by_question, folds=5, seed=3407)
    observed = []
    for train_ids, test_ids in folds:
        assert set(train_ids).isdisjoint(test_ids)
        assert set(train_ids) | set(test_ids) == set(question_ids)
        observed.extend(test_ids)
    assert sorted(observed) == sorted(question_ids)
