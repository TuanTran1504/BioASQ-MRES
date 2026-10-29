"""Expanded compactness provenance and validation-only checkpoint selection."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import select_generator_feedback_dpo as selection
import generator_feedback_dpo_data as data
from run_single_answer_sampling import generation_options, sample_seed
from scripts import judge_generator_feedback as judge


def response(answers):
    return json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                    for i, a in enumerate(answers)]})


def score(epoch, mrr=.6, tokens=50, ensemble_lost=0, llama_lost=0):
    return dict(epoch=epoch, mrr_at5=mrr, mean_output_tokens=tokens,
                ensemble_full_pool_lost=ensemble_lost, llama_full_pool_lost=llama_lost,
                generation_quality=dict(empty_responses=0, ceiling_responses=0, surplus_responses=0))


@pytest.mark.parametrize("change", [dict(ensemble_full_pool_lost=1), dict(llama_full_pool_lost=1),
                                   dict(mrr_at5=.59), dict(mean_output_tokens=51)])
def test_epoch_zero_remains_fallback(change):
    trial = score(1, tokens=40); trial.update(change)
    assert selection.choose([score(0), trial])["epoch"] == 0


def test_selection_ties_prefer_compactness_then_earliest_epoch():
    rows = [score(0), score(1, tokens=40), score(2, tokens=40), score(3, mrr=.61, tokens=45)]
    assert selection.choose(rows)["epoch"] == 3
    assert selection.choose(rows[:3])["epoch"] == 1
    rows[1]["generation_quality"]["empty_responses"] = 1
    assert selection.choose(rows[:2])["epoch"] == 0


@pytest.mark.parametrize("focal", ["llama31", "qwen3", "ministral3"])
def test_ensemble_candidate_is_explicitly_separate_from_strict_selection(focal):
    prefix = "llama" if focal == "llama31" else focal
    base, trial = score(0), score(1,mrr=.61,tokens=40)
    if prefix != "llama":
        for r in (base,trial): r[f"{prefix}_full_pool_lost"] = r.pop("llama_full_pool_lost")
    trial[f"{prefix}_full_pool_lost"] = 1
    assert selection.choose([base,trial],focal)["epoch"] == 0
    strict_status = [r["eligible"] for r in (base,trial)]
    copied = copy.deepcopy([base,trial])
    assert selection.choose(copied,focal,require_individual_coverage=False)["epoch"] == 1
    assert [r["eligible"] for r in (base,trial)] == strict_status


def test_short_invalid_outputs_are_not_compactness_successes():
    rows = [{"raw_response": "", "output_tokens": 0},
            {"raw_response": response(["Core"]), "output_tokens": 512},
            {"raw_response": response(["Core", "a", "b", "c", "d", "e"]), "output_tokens": 100}]
    quality = selection.generation_quality(rows)
    assert (quality["empty_responses"], quality["ceiling_responses"], quality["surplus_responses"]) == (1, 1, 1)


def test_four_validation_draws_keep_the_native_bank_seed_indices():
    cfg = dict(response_mode="equivalent_sampling", num_generations=4, temperature=.8,
               top_p=.95, max_new_tokens=512)
    assert generation_options(cfg)["do_sample"]
    assert generation_options({**cfg, "num_generations": 10})["do_sample"]
    with pytest.raises(ValueError): generation_options({**cfg, "num_generations": 5})
    native = [sample_seed("q", i+1, 3407) for i in range(1, 5)]
    offset = [sample_seed("q", i+1, 3407) for i in range(1, 5)]
    assert native == offset and native[0] != sample_seed("q", 1, 3407)


def test_native_validation_has_all_144_questions_and_keeps_gold_separate():
    rows = [{"question_id": f"v{i}", "messages": [{}, {"content": json.dumps({"question": "Which?",
             "snippets": [{"id": "s1", "text": "Core."}]})}]} for i in range(144)]
    panels = {"validation": rows, "train": [{"question_id": "fitting"}]}
    fake = types.SimpleNamespace(panels=lambda cfg: (panels, {r["question_id"]: ["Core"] for r in rows}))
    examples = selection.native_examples(fake, {}, [])
    assert len(examples) == 144 and examples[0]["snippets"] == [{"snippet_id": "s1", "text": "Core."}]
    panels["train"] = [{"question_id": "v0"}]
    with pytest.raises(ValueError, match="disjoint"): selection.native_examples(fake, {}, [])


def test_truncated_judgment_recovery_reuses_completed_requests(tmp_path):
    jobs, _ = judge.prepare([{"question_id": "q", "split": "train", "excluded_reason": "",
        "prompt": [{}, {"content": json.dumps({"question": "Which?", "snippets": [{"id": "s1", "text": "Core."}]})}],
        "candidates": [{"answer": "Core", "relation_type": "original"}]}])
    job = jobs[0]
    judge.shared.save(tmp_path / "responses" / (job["request_sha256"] + "-1.json"), {"choices": [{"finish_reason": "length"}]})
    changed = judge.recover_truncated_jobs(jobs, tmp_path)[0]
    assert changed["request_sha256"] != job["request_sha256"] and changed["unit_ids"] == job["unit_ids"]
    assert changed["payload"]["max_tokens"] == 2 * job["payload"]["max_tokens"]
    judge.shared.save(tmp_path / "cache" / (job["request_sha256"] + ".json"), {})
    assert judge.recover_truncated_jobs(jobs, tmp_path) == jobs


@pytest.mark.parametrize("tamper", [None, "seed", "top_p"])
@pytest.mark.parametrize("focal", ["llama31", "qwen3", "ministral3"])
@pytest.mark.parametrize("protocol", ["compactness", "coverage", "coverage_reuse"])
def test_three_epoch_selection_replaces_only_focal_and_never_supplies_gold_to_worker(tmp_path, monkeypatch, tamper, focal, protocol):
    sub, train, out = tmp_path / "sub", tmp_path / "train", tmp_path / "out"
    sub.mkdir(); (train / "adapter").mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        (train / "adapter" / name).write_text("saved")
    source = dict(adapter=str(train / "adapter"), system_prompt="Return up to FIVE answers.",
                  model_loader="fast_model" if focal == "ministral3" else "fast_language", chat_template_kwargs={}, base_snapshot="base")
    cfg = dict(review_scope="all_compactness", retain_epoch_checkpoints=True, num_train_epochs=3, model_key=focal)
    if protocol.startswith("coverage"):
        cfg.update(review_scope="official_form",checkpoint_selection="Coverage primary")
    metadata = dict(path="unused", manifest_sha256="hash", responses_sha256="raw")
    feedback = dict(banks={m: metadata for m in ("llama31", "qwen3", "ministral3")}, rerankers=str(tmp_path / "rr"), reranker_hashes={})
    prepared = dict(repository=str(tmp_path), configuration=cfg, code_hashes={}, preferences="reviewed", native_banks=feedback["banks"])
    data.write(sub / "prepared.json", prepared)
    state = dict(status="complete", smoke_test=False, source=source, configuration=cfg,
                 prepared_sha256=data.digest(sub / "prepared.json"), phase="save_adapter",
                 trained_pair_counts={"train": 1, "validation": 1}, global_steps=81, adapter_files_sha256={})
    data.write(train / "manifest.json", state)
    examples = [{"question_id": f"v{i}", "question": "Which?", "snippets": [{"snippet_id": "s1", "text": "Core."}],
                 "gold_aliases": ["Core"]} for i in range(144)]
    native = [dict(question_id=e["question_id"], model_key=m, split="validation", draw=d,
                   raw_response=response(["Core", "alias"]), output_tokens=50, prompt_tokens=88,
                   prompt=[{"role":"system","content":source["system_prompt"]}, {"role":"user", "content":json.dumps({
                       "question":e["question"],"snippets":[{"id":"s1","text":"Core."}]})}])
              for e in examples for m in feedback["banks"] for d in range(5)]
    generation = tmp_path / "generation.json"; data.write(generation, {"max_seq_length": 8704})
    fake = types.SimpleNamespace(DEFAULT_CONFIG=generation, MODELS=tuple(feedback["banks"]), pack_banks=lambda *args: ({}, native))
    monkeypatch.setattr(selection, "validate_native", lambda *a: ({}, {"train": [1], "validation": [2]}, source, feedback))
    monkeypatch.setattr(selection, "configure_data_root", lambda *a: fake)
    monkeypatch.setattr(selection, "native_examples", lambda *a: examples)
    monkeypatch.setattr(selection, "verify_files", lambda *a: None)
    monkeypatch.setattr(selection, "load_inputs", lambda *a: (None, None, None, None, {}, feedback["banks"], None))
    real_read = selection.read
    monkeypatch.setattr(selection, "read", lambda p: {"selectors": {"max5": {"model_path": "model.joblib"}}} if str(p).endswith("rr/manifest.json") or str(p).endswith("rr\\manifest.json")
                        else {} if str(p).endswith("reranker_guided_dpo_pilot.json") else real_read(p))
    monkeypatch.setattr(selection.joblib, "load", lambda *a: {"selector": "max5", "schema": "sft-ensemble-logistic-v1"})
    plan = dict(max_seq_length=600, expected_prompt_tokens={e["question_id"]: 88 for e in examples},
                expected_prompt_ids_sha256={e["question_id"]: "tokens" for e in examples})
    monkeypatch.setattr(selection, "full_input_plan", lambda *a: copy.deepcopy(plan))
    checkpoints = []
    for epoch in range(1, 4):
        path = train / "trainer_output" / f"checkpoint-{epoch*27}"; path.mkdir(parents=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"): (path / f).write_text(str(epoch))
        data.write(path / "trainer_state.json", dict(global_step=epoch*27, epoch=epoch,
                   log_history=[dict(step=epoch*27, epoch=epoch, eval_loss=.69-epoch*.001)]))
    monkeypatch.setattr(selection, "inference_worker", lambda *a: Path("worker.py"))
    calls = []
    def worker(command, check):
        config = real_read(Path(command[command.index("--config")+1]))
        destination = Path(command[command.index("--output-dir")+1]); destination.mkdir()
        assert all("gold" not in k for e in data.records(Path(config["input"])) for k in e)
        assert config["max_seq_length"] == 8704 and config["num_generations"] in (1,4)
        config["model_name"] = command[command.index("--model-name")+1]
        if tamper == "top_p": config["top_p"] = .5
        data.write(destination / "config.json", config)
        data.write_rows(destination / "examples.jsonl", data.records(Path(config["input"])))
        generated = []
        for e in examples:
            samples = [dict(draw=d, seed=sample_seed(e["question_id"], d+config["sample_seed_offset"],3407),
                            raw_response=response(["Core"]), answers=selection.parse_equivalent_response(response(["Core"]),5)[0], output_tokens=20)
                       for d in range(1, config["num_generations"]+1)]
            if tamper == "seed": samples[0]["seed"] += 1
            generated.append(dict(question_id=e["question_id"], samples=samples, snippets_truncated=False,
                                  included_snippets=1, total_snippets=1))
        data.write_rows(destination / "generations.jsonl", generated)
        data.write(destination / "status.json", {"status": "complete"}); calls.append(config)
    monkeypatch.setattr(selection.subprocess, "run", worker)
    def matcher(examples, candidates, directory, **kwargs):
        assert directory.is_dir() and len(examples) == 144
        if protocol.startswith("coverage"): assert kwargs["jar_path"] == sub / "scorer.jar"
        return {(r["question_id"], r["answer"]): r["answer"] == "Core" for r in candidates}
    monkeypatch.setattr(selection, "official_candidate_matches", matcher)
    def rank(payload, pools, contexts):
        conditions, scores = {}, {}
        for name, rr in pools.items():
            # Unchanged companions retain their original four-draw frequencies.
            for r in rr:
                if r["answer"] == "alias":
                    for companion in set(feedback["banks"])-{focal}:
                        assert r["sample_frequencies"].get(companion) == 1
                    if not name.startswith("epoch0_"):
                        assert focal not in r["sources"]
            groups = {q: [r for r in rr if r["question_id"] == q] for q in contexts}
            metrics, perq = selection.evaluate_rankings(groups)
            conditions[name+"_max5_reranker"] = {"metrics": metrics}; scores[name+"_max5_reranker"] = perq
        return conditions, scores, []
    monkeypatch.setattr(selection, "rank_frozen", rank)
    options = {}
    if protocol.startswith("coverage"):
        from scripts.evaluate_generator_form_designs import exact_pools, choose_coverage
        options = dict(native_validator=lambda *a: ({}, {"train":[1],"validation":[2]},source,feedback),
            checkpoint_chooser=choose_coverage,pool_builder=exact_pools,preserve_surfaces=True,jar_path_override=sub / "scorer.jar")
    if tamper:
        with pytest.raises(ValueError, match="seed"):
            selection.select(sub, train, out, **options)
        assert real_read(out / "manifest.json")["status"] == "failed"
        assert not (out / "selected_adapter").exists()
        return
    if protocol == "coverage_reuse":
        def missing_library(*a,**kw): raise RuntimeError("java.lang.NoClassDefFoundError: com/google/gson/stream/JsonReader")
        monkeypatch.setattr(selection,"official_candidate_matches",missing_library)
        with pytest.raises(RuntimeError,match="JsonReader"): selection.select(sub,train,out,**options)
        assert len(calls)==6 and real_read(out/"manifest.json")["status"]=="failed"
        original_hashes={p.relative_to(out).as_posix():data.digest(p) for p in out.rglob("*") if p.is_file()}
        saved=out
        monkeypatch.setattr(selection,"official_candidate_matches",matcher)
        def no_gpu(*a,**kw): raise AssertionError("CPU recovery attempted generation")
        monkeypatch.setattr(selection.subprocess,"run",no_gpu)
        out=tmp_path/"recovered"
        selection.select(sub,train,out,saved_generation_root=saved,**options)
        assert all(data.digest(saved/name)==value for name,value in original_hashes.items())
        assert real_read(out/"manifest.json")["saved_generation_root"]==str(saved)
    else:
        selection.select(sub, train, out, **options)
    summary = real_read(out / "comparison_summary.json")
    assert len(calls) == 6 and summary["question_count"] == 144
    assert [r["epoch"] for r in summary["checkpoint_scores"]] == [0,1,2,3]
    chosen_epoch = 1 if protocol == "compactness" else 0
    assert summary["selected_epoch"] == chosen_epoch
    assert summary["ensemble_only_candidate_epoch"] == chosen_epoch
    solo_prefix = "llama" if focal == "llama31" else focal
    assert f"{solo_prefix}_epoch1_sampling4" in summary["conditions"]
    assert real_read(out / "manifest.json")["status"] == "complete"
    assert (out / "selected_adapter/adapter_model.safetensors").read_text() == ("1" if chosen_epoch == 1 else "saved")


def test_fresh_selector_cli_imports():
    result = subprocess.run([sys.executable, str(ROOT / "gadi_sft_8b_starter/scripts/select_generator_feedback_dpo.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("model", ["llama31", "qwen3", "ministral3"])
def test_expanded_launcher_preflights_before_three_dependent_jobs(tmp_path, fail, model):
    import shutil
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable: pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    pref = tmp_path / "preferences"; pref.mkdir()
    source = ROOT / "gadi_sft_8b_starter/scripts/submit_expanded_generator_dpo.sh"
    stub = '''
module() { :; }
source() { :; }
python3() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then shift; mkdir -p "$1"; return 0; fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '402.gadi-pbs\\n'; }
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace("set -euo pipefail", "set -euo pipefail\n"+stub), newline="\n")
    log = tmp_path / "jobs.log"
    result = subprocess.run([executable, path.as_posix(), pref.as_posix(), model], capture_output=True, text=True,
                            env={**os.environ,"TEST_USER":"test","TEST_LOG":log.as_posix(),"TEST_FAIL":str(int(fail))}, timeout=30)
    if fail: assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        lines = log.read_text().splitlines()
        assert len(lines) == 3 and all("afterok:402.gadi-pbs" in line for line in lines[1:])
        assert all(f"DPO_MODEL={model}" in line for line in lines[:2])
        assert f"{model}-compactness-train-402.gadi-pbs" in lines[2]
