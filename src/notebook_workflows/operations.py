"""Shared implementations for workflows that previously lived inside notebooks.

Heavy dependencies are imported only inside the selected operation, in a worker
process. The notebook kernel remains free of loaded models and GPU state.
"""
from __future__ import annotations

import json
import gc
import re
import runpy
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def call_cli(module, parameters):
    from .runner import _argv, cli_schema
    schema_module = "src.utility.evaluation" if module == "src.utility.evaluate_models" else module
    argv = _argv(module, parameters, cli_schema(schema_module))
    original_argv = sys.argv
    try:
        sys.argv = [module, *argv[3:]]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = original_argv
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def packets_from_prepared(source):
    from cse_dpo import evidence_sft_data as evidence
    from cse_dpo.normalize_set_answers import normalize_answer_surface
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Evidence export source must be a prepared JSON list.")
    grouped = {}
    for row in rows:
        qid = row.get("source_question_id") or row.get("original_id") or row["id"]
        resources = evidence.raw_resources(row)
        if qid in grouped:
            if grouped[qid]["resources"] != resources or grouped[qid]["question"] != row["input_1"]:
                raise ValueError(f"Conflicting question/resource records for {qid}")
            grouped[qid]["alias_texts"].extend(evidence.aliases_of(row["output"]))
        else:
            grouped[qid] = {"question": row["input_1"], "resources": resources,
                            "alias_texts": evidence.aliases_of(row["output"]),
                            "prior_question_flag": row.get("prior_question_flag"),
                            "prior_alias_flags": row.get("prior_alias_flags", {})}
    packets = []
    for qid, group in sorted(grouped.items()):
        aliases = []
        for i, alias in enumerate(dict.fromkeys(group["alias_texts"]), 1):
            key = normalize_answer_surface(alias)
            aliases.append({
                "alias_id": f"a{i}", "text": alias,
                "occurrence_resource_ids": [r["resource_id"] for r in group["resources"]
                                           if key and f" {key} " in f" {normalize_answer_surface(r['text'])} "],
                "literal_resource_ids": [r["resource_id"] for r in group["resources"] if alias in r["text"]],
                "prior_flags": group["prior_alias_flags"].get(alias, []),
            })
        if not aliases or not group["resources"]:
            raise ValueError(f"Missing gold aliases or evidence: {qid}")
        packet = {"question_id": qid, "split": "train", "question": group["question"],
                  "resources": group["resources"], "aliases": aliases,
                  "original_gold_output": "".join(f"[BE]{a['text']}[EE]" for a in aliases),
                  "prior_question_flag": group["prior_question_flag"], "historical_positive_hints": []}
        packet["input_sha256"] = evidence.digest(packet)
        packets.append(packet)
    manifest = {"version": "notebook-prepared-packets-v1", "question_count": len(packets),
                "source_hashes": {str(source): evidence.file_hash(source)},
                "packets_sha256": evidence.digest(packets),
                "historical_flags": "Only flags present in the supplied prepared file are carried forward."}
    return packets, manifest


def occurrence_export(p, out, root):
    from cse_dpo.occurrence_sft_data import export_occurrence_data
    packets, manifest = packets_from_prepared(p["source"])
    return export_occurrence_data(packets, manifest, out / "export", mode=p["match_mode"],
                                  all_aliases=p["all_aliases"])


def evidence_annotation(p, out, root):
    from cse_dpo import evidence_sft_data as evidence
    packets, manifest = packets_from_prepared(p["source"])
    if p["phase"] == "export":
        decisions = evidence.load_decisions(packets, cache_dir=Path(p["annotation_dir"]))
        return evidence.export_data(packets, decisions, out / "export", all_supported_aliases=True,
                                    order_seed=p["seed"], allow_partial=False)
    annotation_root = out / "annotation"
    evidence.initialize_run(annotation_root, packets, manifest)
    if p.get("annotation_dir"):
        previous = Path(p["annotation_dir"])
        shutil.copytree(previous, annotation_root / "judgments" / previous.name)
    judge = evidence.chat_json_judge("https://api.openai.com/v1/chat/completions", p["model"],
                                    api_key_file=root / p["api_key_file"])
    cache = evidence.annotate(packets, annotation_root, judge, {"model": p["model"]},
                              limit=p["max_new_calls"])
    return {"cache_dir": str(cache), "note": "Export requires complete validated judgments."}


def dev_evidence(p, out, root):
    from cse_dpo import dev_evidence_annotation as dev, evidence_sft_data as evidence
    packets, manifest = dev.build_packets(root, dev_path=p["source"], train_path=p["train_source"])
    if p["phase"] == "occurrence":
        return dev.export_annotations(packets, manifest, {}, out / "export", occurrence_only=True)
    if p["phase"] == "export":
        decisions = dev.load_decisions(packets, cache_dir=p["annotation_dir"])
        return dev.export_annotations(packets, manifest, decisions, out / "export")
    annotation_root = out / "annotation"
    dev.initialize_run(annotation_root, packets, manifest)
    if p.get("annotation_dir"):
        previous = Path(p["annotation_dir"])
        shutil.copytree(previous, annotation_root / "judgments" / previous.name)
    judge = evidence.chat_json_judge("https://api.openai.com/v1/chat/completions", p["model"],
                                    api_key_file=root / p["api_key_file"], system_prompt=dev.JUDGE_PROMPT)
    result = dev.annotate(packets, annotation_root, judge, {"model": p["model"]},
                         limit=p["max_new_calls"], validation_retries=0)
    return {"annotation_result": result}


def merge_candidates(p, out, root):
    merged, ids = [], set()
    for label, path in p["banks"].items():
        for index, row in enumerate(read_jsonl(path)):
            source_id = row.get("response_id", str(index))
            response_id = f"{label}::{source_id}"
            if response_id in ids:
                raise ValueError(f"Duplicate response_id in bank {label}: {source_id}")
            ids.add(response_id)
            merged.append({**row, "response_id": response_id, "source_response_id": source_id,
                           "generator_label": label})
    write_jsonl(out / "candidate_bank.jsonl", merged)
    return {"rows": len(merged), "questions": len({r["question_id"] for r in merged})}


def curriculum_split(p, out, root):
    from cse_dpo.annotate_remaining_candidate_bank_questions import split_curriculum_pairs
    rows = read_jsonl(p["pair_file"])
    valid = {("C3", "C1"), ("C3", "C2"), ("C2", "C1")}
    unknown = {tuple(r.get(key) for key in ("chosen_class", "rejected_class")) for r in rows} - valid
    if unknown:
        raise ValueError(f"Unexpected class directions; resolve these before splitting: {unknown}")
    return split_curriculum_pairs(Path(p["pair_file"]), out / "staged")


def compare_banks(p, out, root):
    from src.utility.bioasq_format import parse_prediction_items
    banks, summary = {}, {}
    for label, path in p["banks"].items():
        rows = read_jsonl(path)
        grouped = defaultdict(set)
        for row in rows:
            answers = parse_prediction_items(row.get("raw_output", row.get("prediction", "")), "factoid")
            grouped[row["question_id"]].update(re.sub(r"\s+", " ", a).strip().casefold() for a in answers)
        banks[label] = grouped
        counts = Counter(row["question_id"] for row in rows)
        summary[label] = {"rows": len(rows), "questions": len(grouped),
                          "sample_count_distribution": dict(Counter(counts.values())),
                          "unique_answer_count_distribution": dict(Counter(map(len, grouped.values())))}
    qids = sorted(set().union(*(set(bank) for bank in banks.values())))
    differences = [
        {"question_id": qid, "answers": {label: sorted(bank.get(qid, set())) for label, bank in banks.items()}}
        for qid in qids
        if len({tuple(sorted(bank.get(qid, set()))) for bank in banks.values()}) > 1
    ]
    write_json(out / "differences.json", differences)
    return {"banks": summary, "different_question_count": len(differences)}


def evidence_coverage(p, out, root):
    from scripts.prepare_factoid_snippet_sft import supported_aliases
    summaries, details = [], []
    for source in p["sources"]:
        questions = json.loads(Path(source).read_text(encoding="utf-8"))["questions"]
        qs = [q for q in questions if q.get("type") == "factoid"]
        selected = [{"question_id": q["id"], "aliases": supported_aliases(q, p["match_mode"])} for q in qs]
        summaries.append({"source": source, "factoid_questions": len(qs),
                          "matching_questions": sum(bool(row["aliases"]) for row in selected)})
        details.extend({**row, "source": source} for row in selected)
    write_json(out / "question_coverage.json", details)
    return summaries


def staged_dpo(p, out, root):
    from cse_dpo.train_factoid_three_stage_dpo import Config, run_three_stage
    parameters = {**p, "output_root": str(out / "training")}
    return run_three_stage(Config(**parameters))


def rationale_sft(p, out, root):
    from cse_dpo import run_factoid_rationale_sft_dpo_comparison as driver
    from transformers import AutoTokenizer
    driver.BASE_MODEL = Path(p["base_model"])
    driver.OLD_SFT_ADAPTER = Path(p["initial_adapter"])
    driver.SOURCE_TRAIN = Path(p["train_source"])
    driver.DEV_INPUT = Path(p["dev_source"])
    driver.RATIONALE_BANK = Path(p["rationale_bank"])
    driver.SFT_DATA_DIR = out / "prepared_data"
    driver.SFT_MIXED_FILE = driver.SFT_DATA_DIR / "mixed_train.jsonl"
    driver.MAX_LENGTH = p["max_length"]
    driver.SEED = p["seed"]
    driver.ANSWER_ONLY_REPLAY_FRACTION = p["replay_fraction"]
    tokenizer = AutoTokenizer.from_pretrained(str(driver.BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = driver.load_prompt_spec()
    records = driver.build_sft_data(tokenizer, prompt, expected_question_count=None)
    dev = driver.build_dev_records(tokenizer, prompt)
    arm = driver.Arm("rationale_sft", out / "training/adapter_final", out / "training",
                     True, p["from_base"], p["epochs"], p["learning_rate"])
    adapter = driver.train_sft_arm(arm, tokenizer, records, dev, prompt, preflight_only=False)
    return {"adapter": str(adapter), "examples": len(records)}


def error_aware_dpo(p, out, root):
    from cse_dpo import run_factoid_error_aware_multitask_dpo_qwen25_05b as driver
    driver.PROJECT_PYTHON = Path(sys.executable)
    driver.BASE_MODEL, driver.INITIAL_ADAPTER = Path(p["base_model"]), Path(p["initial_adapter"])
    driver.STRICT_STAGE1, driver.C1_RATIONALES = Path(p["stage1_pairs"]), Path(p["c1_rationales"])
    driver.GOLD_RATIONALES = Path(p["gold_rationales"])
    driver.DATASET_ROOT, driver.OUTPUT_ROOT = out / "data", out / "training"
    driver.AUXILIARY_WEIGHT, driver.MAX_LENGTH = p["auxiliary_weight"], p["max_length"]
    rows, summary = driver.build_dataset()
    driver.token_preflight(rows)
    config = driver.training_config()
    config.dev_source_input = p["dev_source"]
    write_json(out / "training_config.json", config.__dict__)
    if p["train"]:
        from cse_dpo.train_factoid_three_stage_dpo import run_three_stage
        run_three_stage(config)
    return {**summary, "training_requested": p["train"]}


def conditioned_sampling(p, out, root):
    from cse_dpo import compare_frequency_vs_conditioned_sampling as driver
    driver.PROJECT_PYTHON = Path(sys.executable)
    driver.BASE_MODEL, driver.ADAPTER = Path(p["base_model"]), Path(p["adapter"])
    driver.DEV_INPUT, driver.FROZEN_SAMPLE10_BANK = Path(p["source"]), Path(p["bank"])
    driver.OUTPUT_ROOT = out / "comparison"
    sys.argv = ["conditioned_sampling"] + (["--limit", str(p["limit"])] if p["limit"] else [])
    driver.main()
    return {"output": str(driver.OUTPUT_ROOT)}


def openai_direct(p, out, root):
    from argparse import Namespace
    from cse_dpo import compare_gpt_direct_vs_structured_reasoning_dev as api
    from cse_dpo.generated_bioasq_eval import load_gold_examples
    from src.utility.data import load_records
    args = api.build_parser().parse_args([])
    args.model, args.output_root = p["model"], out
    args.reuse_direct_cache_root = None
    args.max_validation_retries = args.rate_limit_max_retries = 0
    args.direct_max_tokens = p["max_new_tokens"]
    args.api_key_file = root / p["api_key_file"]
    records = load_records(p["sources"], Namespace(
        question_types=["factoid"], max_resources=0, max_resource_chars=0,
        max_factoid_answers=10000, max_list_items=100, max_summary_answers=5))
    gold = load_gold_examples([Path(path) for path in p["sources"]])
    generator = api.PairedGenerator(args, api.read_api_key(args.api_key_file))
    predictions = []
    for row in records[:p["question_limit"]]:
        result = generator.generate_arm(row, "direct")
        predictions.append({"question_id": row["id"], "direct_prediction": result["response"]})
        write_jsonl(out / "predictions.jsonl", predictions)
    scores, per_question = api.score_arm("direct", predictions, gold, out)
    return {"scores": scores, "new_api_calls": generator.new_api_calls}


def sampling_comparison(p, out, root):
    from .catalog import PROMPT
    common = {**PROMPT, "eval_input": [p["source"]], "model_ref": [p["model_ref"]],
              "question_types": ["factoid"], "seed": p["seed"],
              "max_seq_length": p["max_seq_length"], "max_new_tokens": p["max_new_tokens"]}
    strategies = p["strategies"]
    summaries = {}
    # The independent bank is shared by first-N and frequency evaluations.
    if set(strategies) & {"first5", "frequency"}:
        call_cli("cse_dpo.generate_candidate_bank", {
            **common, "samples_per_question_total": p["samples"], "temperature": p["temperature"],
            "top_p": p["top_p"], "output_dir": str(out / "bank"), "batch_size": 1,
        })
        paths = list((out / "bank").rglob("candidate_bank.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"Expected one generated bank, found {len(paths)}")
        rows = read_jsonl(paths[0])
        by_question = defaultdict(list)
        for row in sorted(rows, key=lambda r: (r["question_id"], r["sample_id"])):
            by_question[row["question_id"]].append(row["raw_output"])
        from cse_dpo.generated_bioasq_eval import load_gold_examples
        from src.utility.bioasq_format import parse_prediction_items
        from src.utility.bioasq_official import evaluate_with_bioasq_java
        from argparse import Namespace
        gold = load_gold_examples([Path(p["source"])])
        score_args = Namespace(bioasq_java_jar="third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",
                               bioasq_java_version=5, bioasq_java_heap="4G")
        for strategy in set(strategies) & {"first5", "frequency"}:
            predictions = []
            for qid, samples in by_question.items():
                candidates = []
                for sample in samples[:5] if strategy == "first5" else samples:
                    values = parse_prediction_items(sample, "factoid")
                    if len(values) == 1 and "[BE]" in sample and "[EE]" in sample:
                        candidates.append(values[0])
                surfaces = {}
                votes = Counter()
                for candidate in candidates:
                    key = re.sub(r"\s+", " ", candidate).strip().casefold()
                    surfaces.setdefault(key, candidate)
                    votes[key] += 1
                keys = sorted(votes, key=lambda key: -votes[key]) if strategy == "frequency" else list(surfaces)
                predictions.append({"question_id": qid, "question_type": "factoid",
                                    "prediction": " ".join(f"[BE]{surfaces[key]}[EE]" for key in keys[:5])})
            scored = evaluate_with_bioasq_java(
                prediction_rows=predictions, examples_by_key={(qid, "factoid"): ex for qid, ex in gold.items()},
                args=score_args, model_label=strategy, model_dir=out / strategy)
            write_json(out / strategy / "predictions.json", predictions)
            summaries[strategy] = scored
    for strategy in [s for s in strategies if s in {"greedy", "direct_top5"}]:
        call_cli("src.utility.evaluate_models", {
            **common, "num_generations": 1, "temperature": 0., "top_p": 1.,
            "prompt": "factoid-top-five-eval-v1" if strategy == "direct_top5" else common["prompt"],
            "output_dir": str(out / strategy), "score_backend": "bioasq_java",
        })
        summaries[strategy] = {"output": str(out / strategy)}
    return summaries


def stage1_audit(p, out, root):
    from cse_dpo import audit_stage1_training_generations as audit
    stage = audit.resolve_stage1(root, Path(p["run_root"]))
    if Path(stage["base_model"]).resolve() != Path(p["base_model"]).resolve():
        raise ValueError("Base model differs from the completed stage manifest.")
    eligible, excluded, summary = audit.prepare_questions(
        root, stage, Path(p["source"]), root / "data/training13b.json")
    tokenizer, questions, bundle = audit.load_tokenizer_and_prompts(root, stage, eligible)
    settings = {"max_prompt_tokens": p["max_prompt_tokens"], "max_new_tokens": p["max_new_tokens"],
                "seed": p["seed"]}
    audit.initialize_audit(out / "audit", stage, questions, settings, summary)
    model = audit.load_stage1_model(stage)
    audit.run_audit(model, tokenizer, questions, out / "audit",
                    max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                    limit_questions=p["limit"])
    return audit.score_completed_questions(root, out / "audit", questions)
