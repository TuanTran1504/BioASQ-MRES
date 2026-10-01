from __future__ import annotations

from types import SimpleNamespace

import pytest

from cse_dpo.build_synthetic_factoid_qa_pilot import (
    CachedJsonCaller,
    VERIFICATION_RUBRIC_VERSION,
    finalize,
    question_requests_multiple_answers,
    render_synthetic_dpo_prompt,
    validate_verification,
    verification_match_type,
    write_json,
    write_jsonl,
)


def evidence(text: str) -> list[dict[str, str]]:
    return [
        {
            "resource_id": "1",
            "kind": "target",
            "pool_id": "pool_00001",
            "pubmed_id": "123",
            "text": text,
        }
    ]


def response_item(**changes):
    value = {
        "synthetic_question_id": "synthetic_source_0001::q1",
        "status": "accepted",
        "candidate_answers": [{"answer": "NR2F1", "resource_id": "1"}],
        "extracted_answer": "NR2F1",
        "resource_id": "1",
        "unique_answer": True,
        "explicit_relation": True,
        "single_factoid_answer": True,
        "minimal_span_certain": True,
        "basis": "Resource 1 explicitly identifies NR2F1 as the mutated gene.",
    }
    value.update(changes)
    return {"items": [value]}


def validate(value, question: str, text: str):
    return validate_verification(
        value,
        {"synthetic_source_id": "synthetic_source_0001"},
        [{"question": question}],
        evidence(text),
    )


def test_v2_accepts_a_long_atomic_exact_span_without_a_word_limit():
    answer = (
        "autosomal dominant intellectual developmental disorder with speech delay "
        "and dysmorphic facial features caused by a pathogenic chromatin regulator variant"
    )
    value = response_item(
        candidate_answers=[{"answer": answer, "resource_id": "1"}],
        extracted_answer=answer,
        basis="Resource 1 explicitly gives this complete qualified disorder name.",
    )

    validate(value, "What disorder is caused by the variant?", f"The variant causes {answer}.")


def test_v2_downgrades_plural_enumeration_without_retrying():
    value = response_item(
        candidate_answers=[{"answer": "bedaquiline and delamanid", "resource_id": "1"}],
        extracted_answer="bedaquiline and delamanid",
        basis="Resource 1 explicitly lists both drugs as approved treatments.",
    )

    validate(
        value,
        "Which drugs were approved for drug-resistant tuberculosis?",
        "The approved drugs were bedaquiline and delamanid.",
    )

    assert value["items"][0]["status"] == "review"


@pytest.mark.parametrize("noun", ["therapies", "regions", "techniques", "organisms"])
def test_v2_downgrades_additional_plural_factoid_categories(noun):
    value = response_item(
        candidate_answers=[{"answer": "alpha and beta", "resource_id": "1"}],
        extracted_answer="alpha and beta",
        basis="Resource 1 explicitly supplies two independent targets for the question.",
    )

    validate(value, f"Which {noun} were identified?", "The identified targets were alpha and beta.")

    assert value["items"][0]["status"] == "review"


def test_v2_recovers_non_exact_candidate_as_review():
    value = response_item(
        candidate_answers=[{"answer": "the gene was NR2F1", "resource_id": "1"}],
        extracted_answer="NR2F1",
        basis="Resource 1 explicitly identifies NR2F1 as the mutated gene.",
    )

    validate(value, "Which gene is mutated?", "The mutated gene was NR2F1.")

    item = value["items"][0]
    assert item["status"] == "review"
    assert item["candidate_answers"] == [{"answer": "NR2F1", "resource_id": "1"}]
    assert item["unique_answer"] is True


def test_v2_recovers_exact_extracted_answer_missing_from_candidates_as_review():
    value = response_item(
        candidate_answers=[{"answer": "NR2F2", "resource_id": "1"}],
        extracted_answer="NR2F1",
        basis="Resource 1 contains two different gene surfaces in the evidence.",
    )

    validate(value, "Which gene is mutated?", "NR2F1 is mutated, whereas NR2F2 is unchanged.")

    item = value["items"][0]
    assert item["status"] == "review"
    assert item["candidate_answers"] == [
        {"answer": "NR2F2", "resource_id": "1"},
        {"answer": "NR2F1", "resource_id": "1"},
    ]
    assert item["unique_answer"] is False


def test_v2_clears_non_exact_extracted_answer_and_forces_review():
    value = response_item(
        extracted_answer="NR2F1 gene",
        basis="Resource 1 explicitly identifies NR2F1 as the mutated gene.",
    )

    validate(value, "Which gene is mutated?", "The mutated gene was NR2F1.")

    item = value["items"][0]
    assert item["status"] == "review"
    assert item["extracted_answer"] == ""
    assert item["resource_id"] == ""


def test_v2_derives_uniqueness_and_downgrades_multiple_candidates():
    value = response_item(
        candidate_answers=[
            {"answer": "2010", "resource_id": "1"},
            {"answer": "September 2010", "resource_id": "1"},
        ],
        extracted_answer="September 2010",
        unique_answer=True,
        basis="Resource 1 contains both a year and a more precise approval date.",
    )

    validate(
        value,
        "When was the drug approved?",
        "The drug was approved in 2010, specifically in September 2010.",
    )

    assert value["items"][0]["unique_answer"] is False
    assert value["items"][0]["status"] == "review"


def test_v2_allows_one_atomic_category_for_plural_grammar():
    answer = "KRAB-ZNF gene clusters"
    value = response_item(
        candidate_answers=[{"answer": answer, "resource_id": "1"}],
        extracted_answer=answer,
        basis="Resource 1 identifies one genomic feature category at these domains.",
    )

    validate(
        value,
        "What genomic features coincide with the CBX1 domains?",
        f"The domains coincide with {answer}.",
    )

    assert value["items"][0]["status"] == "accepted"


def test_v2_accepts_review_with_multiple_exact_candidates():
    candidates = [
        {"answer": "AZD5847", "resource_id": "1"},
        {"answer": "PA-824", "resource_id": "1"},
    ]
    value = response_item(
        status="review",
        candidate_answers=candidates,
        extracted_answer="",
        resource_id="",
        unique_answer=False,
        single_factoid_answer=False,
        basis="Resource 1 supplies two independent compounds, so the answer is not unique.",
    )

    validate(
        value,
        "Which compound was evaluated in the trial?",
        "The evaluated compounds were AZD5847 and PA-824.",
    )


def test_v2_rejects_placeholder_reasoning():
    value = response_item(basis="one concise sentence")

    with pytest.raises(ValueError, match="placeholder"):
        validate(value, "Which gene is mutated?", "The mutated gene is NR2F1.")


def test_answer_agreement_does_not_accept_longer_clauses():
    assert verification_match_type("NR2F1", "NR2F1") == "exact"
    assert verification_match_type("boreal toad", "the boreal toad") == "boundary_variant"
    assert (
        verification_match_type("NR2F1", "The mutated gene is NR2F1")
        == "mismatch"
    )


def test_plural_detection_only_checks_the_requested_noun_phrase():
    assert question_requests_multiple_answers("Which drugs treat the condition?")
    assert question_requests_multiple_answers("Which two binding partners regulate PP1?")
    assert not question_requests_multiple_answers(
        "Which receptor is involved in multiple diseases?"
    )


def test_verification_rubric_is_v2():
    assert VERIFICATION_RUBRIC_VERSION.endswith("-v2")


def test_json_artifacts_are_written_as_utf8(tmp_path):
    path = tmp_path / "unicode.json"
    write_json(path, {"answer": "TNF-α and NF-κB"})
    assert path.read_text(encoding="utf-8") == '{\n  "answer": "TNF-α and NF-κB"\n}\n'


def test_dpo_prompt_preserves_one_resource_header_and_answer_boundary():
    prompt = render_synthetic_dpo_prompt(
        {
            "question": "Which gene is mutated?",
            "evidence_pack": [
                {
                    "pubmed_id": "123",
                    "text": "PubMed ID: 123\n[BS]The mutated gene is NR2F1.[ES]",
                }
            ],
        }
    )

    assert prompt.count("PubMed ID: 123") == 1
    assert "# Question: Which gene is mutated?" in prompt
    assert prompt.endswith("# Answer:")


def test_finalize_refuses_to_export_an_incompletely_verified_run(tmp_path):
    write_jsonl(
        tmp_path / "generated_sources.jsonl",
        [
            {
                "synthetic_source_id": "synthetic_source_0001",
                "generation_eligible": True,
                "generated_items": [{"question": "Which gene?"}],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Verification is incomplete"):
        finalize(SimpleNamespace(output_root=tmp_path))

    assert not (tmp_path / "training_ready").exists()


def test_finalize_refuses_to_export_unprocessed_generation_sources(tmp_path):
    write_json(
        tmp_path / "generation_summary.json",
        {"status": "partial", "unprocessed_source_count": 2},
    )

    with pytest.raises(RuntimeError, match="Generation stopped"):
        finalize(SimpleNamespace(output_root=tmp_path))

    assert not (tmp_path / "training_ready").exists()


def test_cached_caller_recovers_a_previously_valid_response(tmp_path):
    args = SimpleNamespace(
        output_root=tmp_path,
        max_new_calls=1,
        request_delay_seconds=0,
        max_retries=0,
        rate_limit_max_retries=0,
        rate_limit_initial_sleep_seconds=0,
        rate_limit_max_sleep_seconds=0,
    )
    value = {"answer": "TNF-α"}
    first = CachedJsonCaller(args, "verification", "system", "rubric", "model")
    first.call_api = lambda prompt, key: value

    with pytest.raises(RuntimeError, match="old local rule"):
        first.run_one(
            "source",
            lambda feedback: "prompt",
            lambda response: (_ for _ in ()).throw(ValueError("old local rule")),
            "key",
        )

    second = CachedJsonCaller(args, "verification", "system", "rubric", "model")
    second.call_api = lambda prompt, key: pytest.fail("recovery made an API call")
    recovered, origin = second.run_one(
        "source",
        lambda feedback: "prompt",
        lambda response: response,
        "key",
    )

    assert recovered == value
    assert origin == "recovered"
    assert second.new_api_calls == 0
