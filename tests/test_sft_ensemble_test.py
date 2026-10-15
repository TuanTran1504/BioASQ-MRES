"""Check fixed voting, full-panel integrity, archived greedy reuse and submission."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from scripts import analyze_sft_ensemble_test as scorer
from scripts import rescore_sft_ensemble_frequency as frequency
import run_sft_ensemble_test as worker
from run_single_answer_sampling import parse_single_answer, unique_candidates, expansion_candidates, sample_seed
from run_extractive_expansion_8b import parse_equivalent_response
from prepare_matched_8b_sft import write, read, digest


def draw(answer, draw=1, arm="original", qid="q"):
    raw = f"Answer: [BE]{answer}[EE]" if arm == "original" else json.dumps({"answers": [
        {"answer": "wrong", "relation_type": "original"}, {"answer": answer, "relation_type": "synonym"}]})
    sample = {"draw": draw, "seed": sample_seed(qid, draw), "raw_response": raw,
              "output_tokens": 10, "generation_seconds": 0.5, "parse_error": None}
    if arm == "original":
        sample["answer"] = parse_single_answer(raw)
    else:
        sample["answers"] = parse_equivalent_response(raw, 5)[0]
    return sample


def archive(directory, submission, prepared, model, mode):
    directory.mkdir(parents=True)
    write(submission / "manifest.json", prepared)
    inputs = scorer.records(submission / "test_inputs.jsonl")
    state = {"status": "complete", "model_key": model, "smoke_test": False, "expected_questions": 95,
             "submission_dir": str(submission), "submission_manifest_sha256": digest(submission / "manifest.json"),
             "runs": {}, "arm_hashes": {}, "settings": {}}
    for arm in scorer.ARMS:
        run = directory / arm; run.mkdir()
        count = 1 if mode == "greedy" else 10
        prompt = directory / f"{arm}_prompt.txt"
        prompt.write_text(prepared["sources"][model][arm]["source"]["system_prompt"], encoding="utf-8")
        cfg = {"num_generations": count, "response_mode":
            ("equivalent_" if arm == "max5" else "single_answer_") + ("greedy" if count == 1 else "sampling"),
            "seed": 3407, "max_candidates": 5 if arm == "max5" else 1, "max_new_tokens": 512,
            "temperature": 0 if count == 1 else 0.8, "top_p": 0.95, "mark_snippets": False,
            "require_all_snippets": True, "prompt": str(prompt), "prompt_sha256": digest(prompt)}
        rows, candidates = [], []
        for i, e in enumerate(inputs):
            samples = [draw("accepted" if i < 50 else "other", d, arm, e["question_id"]) for d in range(1, count + 1)]
            if i == 94:
                for s in samples:
                    s.update(raw_response="malformed", parse_error="bad format")
                    if arm == "original": s["answer"] = None
                    else: s["answers"] = []
            cc = unique_candidates(samples) if arm == "original" else expansion_candidates(samples)
            candidates.extend({"question_id": e["question_id"], **c} for c in cc)
            rows.append({"question_id": e["question_id"], "question": e["question"], "samples": samples,
                "snippets_truncated": False, "included_snippets": 1, "total_snippets": 1,
                "request_count": count, "prompt_tokens": 10, "input_tokens": count * 10,
                "output_tokens": count * 10, "generation_seconds": count * 0.5,
                "parse_error": "All samples failed" if i == 94 else None})
        for name, rr in (("examples.jsonl", inputs), ("generations.jsonl", rows), ("candidates.jsonl", candidates)):
            scorer.write_jsonl(run / name, rr)
        write(run / "config.json", cfg); write(run / "status.json", {"status": "complete"})
        state["runs"][arm] = arm; state["settings"][arm] = cfg
        state["arm_hashes"][arm] = {f: digest(run / f) for f in
            ("config.json", "status.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl")}
    write(directory / "manifest.json", state)


def panel(tmp_path):
    submission = tmp_path / "submission"; submission.mkdir()
    inputs = [{"question_id": str(i), "question": "Which protein?", "snippets": [{"text": "Complete evidence"}]} for i in range(95)]
    gold = [{"question_id": e["question_id"], "question": e["question"], "gold_aliases": ["accepted"],
             "batch": "B1" if i < 50 else "B2"} for i, e in enumerate(inputs)]
    scorer.write_jsonl(submission / "test_inputs.jsonl", inputs)
    scorer.write_jsonl(submission / "test_gold.jsonl", gold)
    prepared = {"repository": str(tmp_path), "sources": {m: {a: {"source": {
        "system_prompt": f"Native {a} prompt", "adapter_files_sha256": {"weights": "frozen"},
        "base_snapshot": m, "base_revision": "revision", "model_loader": "loader", "chat_template_kwargs": {}}}
        for a in scorer.ARMS} for m in scorer.MODELS}, "analysis_code_hashes": {},
        "file_hashes": {f: digest(submission / f) for f in ("test_inputs.jsonl", "test_gold.jsonl")}}
    return submission, prepared


def test_repeated_draws_are_not_model_votes_and_unique_source_rank():
    banks = {m: {"original": [{"question_id": "q", "samples": []}]} for m in scorer.MODELS}
    banks["llama31"]["original"][0]["samples"] = [draw("accepted", d) for d in range(1, 11)]
    banks["qwen3"]["original"][0]["samples"] = [draw("wrong", 1), draw("second", 2), draw("wrong", 3)]
    banks["ministral3"]["original"][0]["samples"] = [draw("wrong", 1)]
    pool = scorer.merge_pool(banks, "original")
    rows = {r["answer"]: r for r in pool}
    assert rows["accepted"]["sources"] == ["llama31"]
    assert rows["accepted"]["sample_frequencies"]["llama31"] == pytest.approx(1)
    assert rows["wrong"]["sources"] == ["qwen3", "ministral3"]
    assert rows["second"]["source_ranks"]["qwen3"] == 2
    assert scorer.order_pool(pool, "model_vote")[0]["answer"] == "wrong"
    assert scorer.order_pool(pool, "round_robin")[0]["answer"] == "accepted"


def test_rank_vote_uses_reciprocal_unique_source_ranks_and_never_gold():
    rows = [{"answer": "many_votes", "sources": list(scorer.MODELS), "source_ranks": dict.fromkeys(scorer.MODELS, 20),
             "sample_frequencies": dict.fromkeys(scorer.MODELS, 0.1), "label": 1},
            {"answer": "early", "sources": ["llama31", "qwen3"], "source_ranks": {"llama31": 1, "qwen3": 1},
             "sample_frequencies": {"llama31": 0.8, "qwen3": 0.8}, "label": 0}]
    assert scorer.order_pool(rows, "model_vote")[0]["answer"] == "many_votes"
    assert scorer.order_pool(rows, "rank_vote")[0]["answer"] == "early"
    original = {method: [r["answer"] for r in scorer.order_pool(rows, method)] for method in scorer.METHODS}
    for r in rows: r["label"] = 1 - r["label"]
    assert {method: [r["answer"] for r in scorer.order_pool(rows, method)] for method in scorer.METHODS} == original


def test_sample_frequency_counts_draws_instead_of_unique_model_votes():
    banks = {m: {"original": [{"question_id": "q", "samples": [draw("rare", 1),
        *[dict(draw("unused", d), answer=None, parse_error="failed") for d in range(2, 11)]]}]}
        for m in scorer.MODELS}
    banks["llama31"]["original"][0]["samples"] = [draw("frequent", d) for d in range(1, 11)]
    pool = scorer.merge_pool(banks, "original")
    assert scorer.order_pool(pool, "model_vote")[0]["answer"] == "rare"
    assert scorer.order_pool(pool, "sample_frequency")[0]["answer"] == "frequent"
    assert sum(next(r for r in pool if r["answer"] == "rare")["sample_frequencies"].values()) == pytest.approx(0.2)


def test_frequency_counts_duplicate_candidate_once_per_draw_and_retains_failed_draws():
    banks = {m: {"max5": []} for m in scorer.MODELS}
    banks["llama31"]["max5"] = [{"question_id": "q", "samples": [
        {"answers": [{"answer": "A", "relation_type": "original"},
                     {"answer": " a ", "relation_type": "synonym"}]},
        {"answers": [], "parse_error": "failed"}]}]
    pool = scorer.merge_pool(banks, "max5")
    assert len(pool) == 1
    assert pool[0]["sample_frequencies"] == {"llama31": 0.5}


def test_frequency_greedy_equals_model_vote_and_float_ties_use_fixed_provenance():
    rows = [{"answer": "later", "sources": ["qwen3"], "source_ranks": {"qwen3": 1},
             "sample_frequencies": {"qwen3": 1.0}},
            {"answer": "earlier", "sources": ["llama31"], "source_ranks": {"llama31": 1},
             "sample_frequencies": {"llama31": sum([0.1] * 10)}}]
    assert [r["answer"] for r in scorer.order_pool(rows, "sample_frequency")] == ["earlier", "later"]
    rows[1]["sample_frequencies"]["llama31"] = 1.0
    assert scorer.order_pool(rows, "sample_frequency") == scorer.order_pool(rows, "model_vote")


@pytest.mark.parametrize("mode", ["greedy", "sampling10"])
def test_load_reconstructs_complete_panel_and_retains_parse_failures(tmp_path, mode):
    submission, prepared = panel(tmp_path)
    directory = tmp_path / "run"
    archive(directory, submission, prepared, "llama31", mode)
    _, _, examples, gold, generations, costs = scorer.load_archive(directory, mode)
    assert len(examples) == len(gold) == len(generations["original"]) == 95
    count = 1 if mode == "greedy" else 10
    assert costs["original"]["request_count"] == 95 * count
    assert costs["original"]["failed_draws"] == count
    assert costs["original"]["parse_success_count"] == 94
    with (directory / "original/candidates.jsonl").open("a") as handle: handle.write("\n")
    with pytest.raises(ValueError, match="changed"):
        scorer.load_archive(directory, mode)


@pytest.mark.parametrize("kind", ["seed", "truncation", "missing_draw", "sample_budget", "duplicate_question", "prompt", "cost"])
def test_archives_reject_invalid_draws_evidence_or_configuration(tmp_path, kind):
    submission, prepared = panel(tmp_path)
    directory = tmp_path / "run"
    archive(directory, submission, prepared, "llama31", "sampling10")
    run = directory / "original"
    rows = scorer.records(run / "generations.jsonl")
    if kind == "seed": rows[0]["samples"][0]["seed"] += 1
    elif kind == "truncation": rows[0]["snippets_truncated"] = True
    elif kind == "missing_draw": rows[0]["samples"].pop()
    elif kind == "duplicate_question": rows[1]["question_id"] = rows[0]["question_id"]
    elif kind == "prompt": (directory / "original_prompt.txt").write_text("Altered native prompt")
    elif kind == "cost": rows[0]["generation_seconds"] += 1
    else:
        cfg = read(run / "config.json"); cfg["num_generations"] = 9; write(run / "config.json", cfg)
    scorer.write_jsonl(run / "generations.jsonl", rows)
    state = read(directory / "manifest.json")
    for name in state["arm_hashes"]["original"]: state["arm_hashes"]["original"][name] = digest(run / name)
    if kind == "sample_budget": state["settings"]["original"] = read(run / "config.json")
    write(directory / "manifest.json", state)
    with pytest.raises(ValueError): scorer.load_archive(directory, "sampling10")


def test_end_to_end_scores_frequency_and_rescores_legacy_pools_without_java(tmp_path, monkeypatch):
    submission, prepared = panel(tmp_path)
    greedy = {m: tmp_path / f"{m}-greedy" for m in scorer.MODELS}
    jobs = dict(zip(scorer.MODELS, ("10.gadi-pbs", "11.gadi-pbs", "12.gadi-pbs")))
    prepared.update(greedy_archives={m: str(p) for m, p in greedy.items()})
    write(submission / "sampling_jobs.json", jobs)
    old_submission = tmp_path / 'old'; old_submission.mkdir()
    for filename in ('test_inputs.jsonl', 'test_gold.jsonl'):
        shutil.copy2(submission / filename, old_submission / filename)
    old_prepared = copy.deepcopy(prepared)
    for model in scorer.MODELS:
        archive(greedy[model], old_submission, old_prepared, model, 'greedy')
    prepared['greedy_manifest_hashes'] = {m: digest(p / 'manifest.json') for m, p in greedy.items()}
    for model in scorer.MODELS:
        archive(tmp_path / 'gadi_sft_8b_starter/outputs/sft_ensemble_test' /
                f'{model}-sampling10-{jobs[model]}', submission, prepared, model, 'sampling10')
    def matcher(examples, candidates, report, jar_path):
        assert all(e["gold_aliases"] == ["accepted"] for e in examples)
        return {(r["question_id"], r["answer"]): r["answer"] == "accepted" for r in candidates}
    monkeypatch.setattr(scorer, "official_candidate_matches", matcher)
    output = tmp_path / "scored"
    scorer.analyze(submission, output, Path("fake.jar"), resamples=20)
    result = read(output / "comparison_summary.json")
    assert len(result["conditions"]) == 16
    assert len(result["paired_contrasts"]) == 16
    assert result["conditions"]["original_greedy_round_robin"]["metrics"]["mrr_at5"] == pytest.approx(50/95)
    assert result["conditions"]["max5_sampling10_model_vote"]["metrics"]["mrr_at5"] == pytest.approx(25/95)
    assert result["conditions"]["max5_sampling10_model_vote"]["per_batch"]["B1"]["coverage_at5"] == 1
    for arm in scorer.ARMS:
        for mode in ("greedy", "sampling10"):
            assert {result["conditions"][f"{arm}_{mode}_{m}"]["metrics"]["coverage_full_pool"]
                    for m in scorer.METHODS} == {50/95}
    assert result["ensemble_costs"]["max5_sampling10"]["request_count"] == 2850
    assert result["ensemble_costs"]["original_greedy"]["request_count"] == 285
    assert read(output / "manifest.json")["status"] == "complete"
    # Simulate the already completed 12-condition Gadi comparison.
    result["conditions"] = {k: v for k, v in result["conditions"].items() if "sample_frequency" not in k}
    result["paired_contrasts"] = {k: v for k, v in result["paired_contrasts"].items() if "sample_frequency" not in k}
    write(output / "comparison_summary.json", result)
    old_scores = read(output / "comparison_per_question.json")
    write(output / "comparison_per_question.json", {k: v for k, v in old_scores.items() if "sample_frequency" not in k})
    before = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("Frequency rescoring must reuse official labels without Java")
    monkeypatch.setattr(scorer, "official_candidate_matches", forbidden)
    rescored = tmp_path / "frequency"
    frequency.rescore(output, rescored, resamples=20)
    updated = read(rescored / "comparison_summary.json")
    assert len(updated["conditions"]) == 16
    assert len(updated["paired_contrasts"]) == 28
    assert updated["ensemble_costs"] == result["ensemble_costs"]
    for name, condition in result["conditions"].items():
        assert updated["conditions"][name] == condition
    for arm in scorer.ARMS:
        assert updated["conditions"][f"{arm}_greedy_sample_frequency"] == updated["conditions"][f"{arm}_greedy_model_vote"]
    assert read(rescored / "manifest.json")["status"] == "complete"
    assert {p.name: digest(p) for p in output.iterdir() if p.is_file()} == before
    with pytest.raises(FileExistsError):
        frequency.rescore(output, rescored, resamples=20)
    # Changing frequencies, even with unchanged official labels, must fail.
    candidate_path = output / "original_sampling10_candidates.jsonl"
    corrupted = scorer.records(candidate_path)
    corrupted[0]["sample_frequencies"]["llama31"] = 0.1
    scorer.write_jsonl(candidate_path, corrupted)
    with pytest.raises(ValueError, match="frequencies changed"):
        frequency.rescore(output, tmp_path / "tampered", resamples=20)
    assert not (tmp_path / "tampered").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_submits_only_three_smokes_three_full_and_one_dependent_score(tmp_path, fail):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable: pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    source = ROOT / "gadi_sft_8b_starter/scripts/submit_sft_ensemble_test.sh"
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then shift;mkdir -p "$1";return 0;fi
    shift
  done
}
qsub() {
  printf '%s\\n' "$*" >> "$TEST_LOG"
  number=$(cat "$TEST_COUNT");printf '%s' "$((number+1))" > "$TEST_COUNT"
  printf '%s.gadi-pbs\\n' "$number"
}
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log, count = tmp_path / "jobs", tmp_path / "count"; count.write_text("1")
    result = subprocess.run([executable, path.as_posix()], capture_output=True, text=True,
        env={**os.environ, "TEST_USER": "test", "TEST_FAIL": str(int(fail)), "TEST_LOG": log.as_posix(), "TEST_COUNT": count.as_posix()}, timeout=30)
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        jobs = log.read_text().splitlines(); assert len(jobs) == 7
        for i in (1, 3, 5): assert f"depend=afterok:{i}.gadi-pbs" in jobs[i]
        assert "depend=afterok:2.gadi-pbs:4.gadi-pbs:6.gadi-pbs" in jobs[-1]
        assert "jobs/score_sft_ensemble_test.pbs" in jobs[-1]


@pytest.mark.parametrize("parseable", [True, False])
def test_smoke_uses_validation_ten_draws_and_blocks_unparseable_outputs(tmp_path, monkeypatch, parseable):
    submission, prepared = panel(tmp_path)
    smoke_inputs = scorer.records(submission / "test_inputs.jsonl")[:4]
    # Make smoke membership explicitly distinct from test membership.
    for e in smoke_inputs: e["question_id"] = "validation-" + e["question_id"]
    scorer.write_jsonl(submission / "smoke_inputs.jsonl", smoke_inputs)
    prepared["file_hashes"]["smoke_inputs.jsonl"] = digest(submission / "smoke_inputs.jsonl")
    prepared.update(ready=True, sampling={"num_generations": 10, "temperature": 0.8, "top_p": 0.95, "seed": 3407})
    for model in scorer.MODELS:
        for arm in scorer.ARMS:
            prepared["sources"][model][arm]["source"]["adapter_files_sha256"] = {}
            prepared["sources"][model][arm]["source"]["adapter"] = "frozen-adapter"
            prepared["sources"][model][arm]["smoke_plan"] = {"max_seq_length": 1024,
                "expected_prompt_tokens": {r["question_id"]: 10 for r in smoke_inputs}, "expected_prompt_ids_sha256": {}}
    write(submission / "manifest.json", prepared)
    monkeypatch.setattr(worker, "inference_worker", lambda *args: Path("frozen-worker.py"))
    monkeypatch.setenv("PBS_JOBID", "100.gadi-pbs")
    calls = []
    def generate(command, cwd, check):
        calls.append(command)
        cfg_path = Path(command[command.index("--config") + 1]); cfg = read(cfg_path)
        output = Path(command[command.index("--output-dir") + 1]); output.mkdir()
        assert cfg["input"] == str(submission / "smoke_inputs.jsonl")
        assert cfg["num_generations"] == 10 and cfg["temperature"] == 0.8 and cfg["top_p"] == 0.95
        assert cfg["require_all_snippets"] and not cfg["mark_snippets"]
        assert "--preflight-all" in command and "--longest-first" in command
        assert command[command.index("--limit") + 1] == "4"
        cfg["prompt_sha256"] = digest(Path(cfg["prompt"]))
        write(output / "config.json", cfg)
        arm = "max5" if cfg["max_candidates"] == 5 else "original"
        rows = [{"question_id": e["question_id"], "samples": [draw("accepted", d, arm, e["question_id"]) for d in range(1, 11)]} for e in smoke_inputs]
        if not parseable:
            for r in rows:
                for s in r["samples"]: s["parse_error"] = "invalid format"
        scorer.write_jsonl(output / "generations.jsonl", rows)
        scorer.write_jsonl(output / "examples.jsonl", smoke_inputs)
        scorer.write_jsonl(output / "candidates.jsonl", [])
        write(output / "status.json", {"status": "complete"})
    monkeypatch.setattr(worker.subprocess, "run", generate)
    output = tmp_path / "gadi_sft_8b_starter/outputs/sft_ensemble_test/llama31-sampling10-100.gadi-pbs-smoke"
    if parseable:
        worker.run(submission, "llama31", True)
        assert len(calls) == 2 and read(output / "manifest.json")["status"] == "complete"
    else:
        with pytest.raises(ValueError, match="no parseable smoke"):
            worker.run(submission, "llama31", True)
        assert len(calls) == 1 and read(output / "manifest.json")["status"] == "failed"
