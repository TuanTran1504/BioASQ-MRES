"""Selector-data integrity: verbatim surfaces, exclusion, gold blindness and targets."""
import copy
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from scripts import prepare_answer_selector_data as sd


def example(qid, aliases=None):
    return {"question_id": qid, "question": "Which target and numerical bound?",
            "snippets": [{"id": "1", "text": "The target HER2 was observed in more than 54,000 exomes."}],
            "gold_aliases": aliases or ["unused reference alias", "HER-2"]}


def raw(model, split, qid, draw, answers=None):
    answers = answers or ["HER2", "HER-2", ">54,000", "54,000"]
    return {"model_key": model, "split": split, "question_id": qid, "draw": draw,
            "response_id": f"{model}-{qid}-{draw}", "output_tokens": 60,
            "raw_response": json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                                    for i, a in enumerate(answers)]})}


def fixture():
    examples = {"train": [example("t")], "validation": [example("v")]}
    rows = [raw(m, s, e["question_id"], d) for s, ex in examples.items() for e in ex for m in sd.MODELS for d in range(5)]
    return examples, rows


def fake_matcher(examples, candidates, report, jar_path):
    assert report.is_dir()
    aliases = {e["question_id"]: {a.lower() for a in e["gold_aliases"]} for e in examples}
    return {(r["question_id"], r["answer"]): r["answer"].lower() in aliases[r["question_id"]] for r in candidates}


def label(examples, pools, tmp_path):
    sd.label_pools(pools, examples, tmp_path, Path("fake.jar"), fake_matcher)


def test_verbatim_near_duplicates_preserve_unicode_whitespace_case_and_bounds(tmp_path):
    ex, rows = fixture()
    for e in ex.values():
        e[0]["gold_aliases"] = ["SPDL1\u2009G>A"]
    for r in rows:
        r["raw_response"] = raw(r["model_key"], r["split"], r["question_id"], r["draw"],
                                ["SPDL1 G>A", "SPDL1\u2009G>A", "spdl1 G>A", ">54,000", "54,000"])["raw_response"]
    pools, diagnostics, audit = sd.pool_responses(rows, ex)
    assert [r["answer"] for r in pools["train"]["sampling4"]] == ["SPDL1 G>A", "SPDL1\u2009G>A", "spdl1 G>A", ">54,000", "54,000"]
    assert diagnostics["train/sampling4/verbatim_duplicate_forms_retained"] == 12
    label(ex, pools, tmp_path)
    accepted = [r["answer"] for r in pools["train"]["sampling4"] if r["official_accepted"]]
    assert accepted == ["SPDL1\u2009G>A"]
    assert len(audit) == 30


def test_no_greedy_contamination_failed_draw_denominator_caps_and_ceiling():
    ex, rows = fixture()
    for r in rows:
        if r["draw"] == 0:
            r["raw_response"] = raw(r["model_key"], r["split"], r["question_id"], 0, ["greedy only"])["raw_response"]
    bad = next(r for r in rows if r["split"] == "train" and r["model_key"] == "llama31" and r["draw"] == 2)
    bad["raw_response"] = "malformed"
    capped = next(r for r in rows if r["split"] == "train" and r["model_key"] == "llama31" and r["draw"] == 3)
    capped["raw_response"] = raw("llama31", "train", "t", 3, ["1", "2", "3", "4", "5", "sixth"])["raw_response"]
    capped["output_tokens"] = 512
    pools, diag, audit = sd.pool_responses(rows, ex)
    assert diag["train/sampling4/requested_responses"] == 12
    assert diag["train/sampling4/empty_responses"] == 1
    assert diag["train/sampling4/output_token_ceiling"] == 1
    assert {r["answer"] for r in pools["train"]["greedy"]} == {"greedy only"}
    answers = {r["answer"]: r for r in pools["train"]["sampling4"]}
    assert "greedy only" not in answers and "sixth" not in answers
    assert answers["HER2"]["draw_support"]["llama31"] == 2
    assert answers["HER2"]["sample_frequencies"]["llama31"] == 0.5
    assert answers["HER2"]["source_ranks"]["llama31"] == 1
    assert answers["HER-2"]["source_ranks"]["llama31"] == 2
    assert answers["HER2"]["sources"] == list(sd.MODELS)
    assert any("unique_candidate_limit_exceeded" in r["warnings"] for r in audit)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "heldout", "split", "assistant", "empty_aliases"])
def test_membership_and_request_integrity_fail_closed(mutation):
    ex, rows = fixture()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "heldout":
        with pytest.raises(ValueError, match="overlap"):
            sd.membership(ex, ["v"])
        return
    elif mutation == "split":
        rows[0]["split"] = "dev"
    elif mutation == "assistant":
        ex["train"][0]["assistant"] = "reference"
    else:
        ex["train"][0]["gold_aliases"] = []
    with pytest.raises(ValueError):
        sd.membership(ex, [])
        sd.validate_raw(rows, ex)


def test_gold_blind_layout_keeps_every_candidate_and_does_not_order_by_labels():
    e = example("q")
    rows = [{"question_id": "q", "answer": f"candidate {i}", "official_accepted": i == 7} for i in range(20)]
    before = copy.deepcopy(rows)
    messages, candidates, target = sd.layout(e, rows, "sampling4", 3407)
    flipped = [{**r, "official_accepted": not r["official_accepted"]} for r in rows]
    messages2, candidates2, _ = sd.layout(e, flipped, "sampling4", 3407)
    assert messages == messages2
    assert rows == before
    payload = json.loads(messages[1]["content"])
    assert len(payload["candidates"]) == 20 and payload["snippets"] == e["snippets"]
    assert set(payload) == {"question", "snippets", "candidates"}
    assert all(set(r) == {"id", "answer"} for r in payload["candidates"])
    assert "unused reference alias" not in messages[1]["content"]
    assert "official_accepted" not in messages[1]["content"]
    assert next(r for r in candidates if r["id"] == target[0])["official_accepted"]
    assert len(set(target)) == len(target) == 5
    assert sd.layout(e, rows, "sampling4", 3407) == (messages, candidates, target)
    layouts = [json.loads(sd.layout(example(f"q{i}"), rows, "sampling4", 3407)[0][1]["content"])["candidates"]
               for i in range(8)]
    assert len({tuple(r["answer"] for r in p) for p in layouts}) > 1


def test_all_aliases_empty_and_uncovered_pools_and_potential_pairs_not_actual_dpo(tmp_path):
    ex, rows = fixture()
    # Validation contains one covered pool; the second must remain in evaluation inputs.
    for r in rows:
        if r["split"] == "validation" and r["draw"] > 0:
            r["raw_response"] = "invalid"
    pools, _, _ = sd.pool_responses(rows, ex)
    label(ex, pools, tmp_path)
    summary = sd.construct_dataset(pools, ex, tmp_path)
    val_inputs = sd.records(tmp_path / "validation_inputs.jsonl")
    val_sft = sd.records(tmp_path / "validation_sft.jsonl")
    assert len(val_inputs) == 2 and len(val_sft) == 1
    assert summary["pools"]["validation/max5_sampling4"]["empty_pools"] == 1
    assert summary["pools"]["train/max5_sampling4"]["possible_official_positive_negative_comparisons"] == 3
    assert summary["dpo_pairs_created"] == 0
    supervision = sd.records(tmp_path / "train_supervision.jsonl")
    assert all(r["gold_aliases"] == ["unused reference alias", "HER-2"] for r in supervision)
    assert all(r["semantic_review_status"] == "unresolved" for r in supervision)
    assert not list(tmp_path.glob("*dpo*"))
    for r in sd.records(tmp_path / "review_candidates.jsonl"):
        assert r["semantic_category"] == "unresolved" and r["reviewer"] == ""


def test_mechanical_flags_do_not_claim_equivalence():
    flags, _ = sd.close_flags("HER-2", "HER2")
    assert "punctuation_or_spacing" in flags
    flags, _ = sd.close_flags(">54,000", "54,000")
    assert "punctuation_or_spacing" in flags and "numeric_comparison_review" in flags
    # Same letters/digits cannot justify equivalence when a numerical bound changed.
    assert "equivalent_wording" not in flags


def test_pinned_input_tampering_and_path_escape_rejected(tmp_path):
    (tmp_path / "input.json").write_text("original")
    hashes = {"input.json": sd.digest(tmp_path / "input.json")}
    sd.verify_files(tmp_path, hashes)
    (tmp_path / "input.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        sd.verify_files(tmp_path, hashes)
    with pytest.raises(ValueError, match="changed"):
        sd.verify_files(tmp_path, {"../outside.json": "fake"})


def test_official_java_scores_verbatim_aliases_and_unicode(tmp_path):
    from src.utility.bioasq_official import _conda_executable
    try:
        _conda_executable("java")
    except RuntimeError:
        pytest.skip("Official Java runtime unavailable")
    ex = {"train": [example("t", ["unused", "SPDL1\u2009G>A"])],
          "validation": [example("v", ["unused", "HER-2"])]}
    rows = [raw(m, s, e["question_id"], d, ["SPDL1 G>A", "SPDL1\u2009G>A"] if s == "train" else ["HER2", "HER-2"])
            for s, panel in ex.items() for e in panel for m in sd.MODELS for d in range(5)]
    pools, _, _ = sd.pool_responses(rows, ex)
    jar = sd.ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    sd.label_pools(pools, ex, tmp_path, jar)
    assert [r["official_accepted"] for r in pools["train"]["sampling4"]] == [False, True]
    assert [r["official_accepted"] for r in pools["validation"]["sampling4"]] == [False, True]


def test_complete_export_manifest_archive_and_failure_state(monkeypatch, tmp_path):
    ex, rows = fixture()
    submission = tmp_path / "submission"
    submission.mkdir()
    for s in sd.SPLITS:
        sd.jsonl(submission / f"{s}_examples.jsonl", ex[s])
    sd.jsonl(submission / "native_responses.jsonl", rows)
    sd.write(submission / "heldout_ids.json", {"dev160": ["held-dev"], "test95": ["held-test"]})
    sd.write(submission / "source_provenance.json", {"fixture": True})
    sd.write(submission / "prepared.json", {"seed": 3407, "schema": "fixture"})
    monkeypatch.setattr(sd, "COUNTS", {"train": 1, "validation": 1})
    monkeypatch.setattr(sd, "validate_submission", lambda p: sd.read(p / "prepared.json"))
    real_label = sd.label_pools
    monkeypatch.setattr(sd, "label_pools", lambda p, e, o, j: real_label(p, e, o, j, fake_matcher))
    output, archive = tmp_path / "out", tmp_path / "export.tar.gz"
    sd.run(submission, output, archive)
    state = sd.read(output / "manifest.json")
    assert state["status"] == "complete" and not state["model_training"]
    sd.verify_files(output, state["output_sha256"])
    with tarfile.open(archive) as handle:
        names = handle.getnames()
    assert "train_sft.jsonl" in names and "report.md" in names
    assert not any(n.startswith("/") or ".." in Path(n).parts for n in names)
    def fail(*args):
        raise RuntimeError("scorer failed")
    monkeypatch.setattr(sd, "label_pools", fail)
    with pytest.raises(RuntimeError, match="scorer failed"):
        sd.run(submission, tmp_path / "failed")
    assert sd.read(tmp_path / "failed/manifest.json")["status"] == "failed"


def test_preparation_freezes_native_banks_aliases_code_and_scorer(monkeypatch, tmp_path):
    from scripts import train_sft_ensemble_reranker as rr
    from scripts import prepare_reranker_guided_dpo as guided
    ex, rows = fixture()
    repo = tmp_path / "repo"
    starter = repo / "gadi_sft_8b_starter"
    baseline = starter / "baseline"
    old = starter / "old-submission"
    paths, hashes = {}, {}
    for model in sd.MODELS:
        path = starter / model
        path.mkdir(parents=True)
        sd.jsonl(path / "responses.jsonl", [r for r in rows if r["model_key"] == model])
        hashes[model] = sd.digest(path / "responses.jsonl")
        sd.write(path / "manifest.json", {"responses_sha256": hashes[model]})
        paths[model] = str(path)
    sd.write(old / "prepared.json", {"max5_banks": paths})
    counts = {"train": 1, "validation": 1}
    sd.write(baseline / "manifest.json", {"status": "complete", "question_counts": counts,
             "submission": str(old), "prepared_sha256": sd.digest(old / "prepared.json"), "max5_bank_hashes": hashes})
    sd.write(starter / "configs/max5_dpo_banks_8b.json", {"sampled_responses": 4, "training_matrix": "matrix.json"})
    sd.write(starter / "matrix.json", {"outer_dev": "dev.jsonl"})
    sd.jsonl(starter / "dev.jsonl", [{"question_id": f"dev-{i}"} for i in range(160)])
    sd.jsonl(starter / "data/official_factoid_test95_v1/test_inputs.jsonl", [{"question_id": f"test-{i}"} for i in range(95)])
    job = starter / "jobs/audit_answer_selector_data.pbs"
    job.parent.mkdir(parents=True)
    shutil.copy2(sd.ROOT / "gadi_sft_8b_starter/jobs/audit_answer_selector_data.pbs", job)
    dist = Path("third_party/Evaluation-Measures/flat/BioASQEvaluation/dist")
    shutil.copytree(sd.ROOT / dist, repo / dist)
    panels = {s: [{"question_id": e["question_id"], "messages": [
        {"role": "system", "content": "native max-five"}, {"role": "user", "content": json.dumps({
            "question": e["question"], "snippets": e["snippets"]})}]} for e in panel] for s, panel in ex.items()}
    aliases = {e["question_id"]: e["gold_aliases"] for panel in ex.values() for e in panel}
    data = SimpleNamespace(pack_banks=lambda paths, cfg: ({m: {"manifest": {"responses_sha256": h}} for m, h in hashes.items()}, rows),
                           panels=lambda cfg: (panels, aliases))
    monkeypatch.setattr(guided, "configure_data_root", lambda p: data)
    monkeypatch.setattr(sd, "COUNTS", counts)
    def snapshot(repository, destination):
        target = destination / "scripts/prepare_answer_selector_data.py"
        target.parent.mkdir(parents=True)
        shutil.copy2(sd.ROOT / "scripts/prepare_answer_selector_data.py", target)
    monkeypatch.setattr(rr, "snapshot_code", snapshot)
    submission = tmp_path / "prepared"
    sd.prepare(repo, baseline, submission)
    state = sd.validate_submission(submission)
    assert state["question_counts"] == counts
    assert sd.records(submission / "train_examples.jsonl")[0]["gold_aliases"] == aliases["t"]
    assert len(sd.records(submission / "native_responses.jsonl")) == 30
    assert (submission / "scorer/lib").is_dir()
    assert not list(submission.glob("*test_gold*"))
    (submission / "native_responses.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        sd.validate_submission(submission)


def test_membership_reads_existing_input_only_test_archive_when_bundle_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="input-only"):
        sd.test_membership_path(tmp_path)
    path = tmp_path / "outputs/sft_ensemble_test/submission-real/test_inputs.jsonl"
    sd.jsonl(path, [{"question_id": str(i), "question": "Q", "snippets": []} for i in range(95)])
    assert sd.test_membership_path(tmp_path) == path
    sd.jsonl(path, [{"question_id": "same"}] * 95)
    with pytest.raises(FileNotFoundError, match="input-only"):
        sd.test_membership_path(tmp_path)


def test_cpu_only_launcher_uses_python3_and_frozen_job():
    shell = (sd.ROOT / "gadi_sft_8b_starter/scripts/submit_answer_selector_audit.sh").read_text()
    job = (sd.ROOT / "gadi_sft_8b_starter/jobs/audit_answer_selector_data.pbs").read_text()
    assert "python3 ../scripts/prepare_answer_selector_data.py" in shell
    assert '"$submission/job.pbs"' in shell
    assert "#PBS -q normal" in job and "ngpus" not in job
    assert '--mode audit --submission "$SELECTOR_SUBMISSION"' in job
    assert "max5-selector-dataset-${PBS_JOBID%%.*}.tar.gz" in job


@pytest.mark.parametrize("fail_stage", ["none", "prepare", "qsub"])
def test_launcher_submits_exactly_one_audit_and_records_job_or_fails_closed(tmp_path, fail_stage):
    import os
    import subprocess
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    stub = r'''
module() { :; }
source() { :; }
python3() {
  [[ "$FAIL_STAGE" != prepare ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift
      mkdir -p "$1"
      printf '# fake frozen CPU audit\n' > "$1/job.pbs"
      return 0
    fi
    shift
  done
  return 1
}
qsub() {
  [[ "$FAIL_STAGE" != qsub ]] || return 1
  printf '%s\n' "$*" >> "$TEST_LOG"
  printf '900.gadi-pbs\n'
}
'''
    directory = tmp_path / "scripts"
    directory.mkdir()
    launcher = directory / "submit.sh"
    source = (sd.ROOT / "gadi_sft_8b_starter/scripts/submit_answer_selector_audit.sh").read_text()
    launcher.write_text(source.replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "qsub.log"
    result = subprocess.run([executable, launcher.as_posix()], capture_output=True, text=True,
        env={**os.environ, "TEST_USER": "fixture", "TEST_LOG": log.as_posix(), "FAIL_STAGE": fail_stage}, timeout=30)
    records = list(tmp_path.glob("outputs/answer_selector_data/submission-*/jobs.tsv"))
    if fail_stage != "none":
        assert result.returncode != 0 and not records and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert len(records) == 1 and records[0].read_text().splitlines() == ["stage\tjob_id", "audit\t900.gadi-pbs"]
        assert len(log.read_text().splitlines()) == 1
        assert "SELECTOR_SUBMISSION=" in log.read_text() and "job.pbs" in log.read_text()
