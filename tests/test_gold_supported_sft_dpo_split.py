from __future__ import annotations

from cse_dpo.split_gold_supported_sft_dpo import build_split, stratified_question_split


def _question(index: int, aliases: list[str], context_size: int = 100) -> dict:
    output = "".join(f"[BE]{alias}[EE]" for alias in aliases)
    return {
        "id": f"q{index:03d}",
        "type": "factoid",
        "input_1": f"Question {index}?",
        "input_2": "x" * context_size,
        "output": output,
    }


def _alias_rows(question: dict) -> list[dict]:
    answers = []
    remainder = question["output"]
    while "[BE]" in remainder:
        answer = remainder.split("[BE]", 1)[1].split("[EE]", 1)[0]
        answers.append(answer)
        remainder = remainder.split("[EE]", 1)[1]
    return [
        {
            **question,
            "id": f"{question['id']}__supported_alias_{index}",
            "source_question_id": question["id"],
            "output": f"[BE]{answer}[EE]",
        }
        for index, answer in enumerate(answers)
    ]


def test_question_split_is_exact_disjoint_and_deterministic() -> None:
    questions = [
        _question(index, [f"answer {index}"] if index % 4 else [f"answer {index}", f"alias {index}"], 100 + index)
        for index in range(20)
    ]
    first = stratified_question_split(questions, sft_fraction=0.8, seed=3407)
    second = stratified_question_split(questions, sft_fraction=0.8, seed=3407)
    assert first[:2] == second[:2]
    sft_ids, dpo_ids, _ = first
    assert len(sft_ids) == 16
    assert len(dpo_ids) == 4
    assert not (set(sft_ids) & set(dpo_ids))
    assert set(sft_ids) | set(dpo_ids) == {question["id"] for question in questions}


def test_alias_expansion_never_crosses_question_split() -> None:
    questions = [
        _question(index, [f"answer {index}"] if index % 3 else [f"answer {index}", f"alias {index}"])
        for index in range(20)
    ]
    alias_rows = [row for question in questions for row in _alias_rows(question)]
    result = build_split(questions, alias_rows, sft_fraction=0.8, seed=3407)
    sft_ids = set(result["sft_question_ids"])
    dpo_ids = set(result["dpo_question_ids"])
    assert {row["source_question_id"] for row in result["sft_alias_rows"]} <= sft_ids
    assert {row["source_question_id"] for row in result["dpo_alias_rows"]} <= dpo_ids
    assert not ({row["source_question_id"] for row in result["sft_alias_rows"]} & dpo_ids)
    assert len(result["sft_alias_rows"]) + len(result["dpo_alias_rows"]) == len(alias_rows)
