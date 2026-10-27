#!/usr/bin/env python3
"""Audit fitting-only coverage preferences; never train or approve DPO examples."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_answer_selector_data as sd

MODELS = sd.MODELS
SCHEMA = "generator-coverage-preference-audit-v1"
OBJECTIVES = ("sampled_response_coverage", "greedy_recovery")


def frozen_audit(directory):
    state = sd.read(directory / "manifest.json")
    if (state.get("status") != "complete" or state.get("schema") != "max5-answer-selector-audit-v1"
            or state.get("question_counts") != sd.COUNTS):
        raise ValueError("Need the complete native four-sample selector data audit")
    required = {"train_inputs.jsonl", "train_supervision.jsonl", "response_audit.jsonl",
                "heldout_ids.json", "validation_inputs.jsonl", "source_provenance.json"}
    if not required <= set(state["output_sha256"]):
        raise ValueError("Unpinned audit inputs")
    sd.verify_files(directory, state["output_sha256"])
    return state


def load_inputs(directory):
    state = frozen_audit(directory)
    inputs = sd.records(directory / "train_inputs.jsonl")
    supervision = sd.records(directory / "train_supervision.jsonl")
    responses = [r for r in sd.records(directory / "response_audit.jsonl") if r["split"] == "train"]
    heldout = sd.read(directory / "heldout_ids.json")
    # Validation membership only; no validation labels/candidates become preferences.
    validation = {r["question_id"] for r in sd.records(directory / "validation_inputs.jsonl")}
    if len(validation) != sd.COUNTS["validation"]:
        raise ValueError("Wrong validation membership")
    protected = {"validation": sorted(validation), **heldout}
    return inputs, supervision, responses, protected, state


def index_responses(inputs, supervision, audits, protected):
    panels, questions, labels = {}, {}, {}
    for row in inputs:
        identity = (row["question_id"], row["pool"])
        if row["split"] != "train" or identity in panels or row["pool"] not in ("max5_greedy", "max5_sampling4"):
            raise ValueError("Duplicate or non-fitting input")
        payload = json.loads(row["messages"][1]["content"])
        context = {k: payload[k] for k in ("question", "snippets")}
        if row["question_id"] in questions and questions[row["question_id"]] != context:
            raise ValueError("Different evidence between modes")
        questions[row["question_id"]] = context
        panels[identity] = payload["candidates"]
    qids = set(questions)
    if (len(qids) != sd.COUNTS["train"] or len(panels) != 2 * len(qids)
            or any(qids & set(v) for v in protected.values())):
        raise ValueError("Wrong fitting membership or held-out overlap")
    rows, slots = {}, {}
    for row in audits:
        slot = (row["question_id"], row["model_key"], row["draw"])
        rid = row["response_id"]
        if (row["split"] != "train" or slot[0] not in qids or slot[1] not in MODELS
                or type(slot[2]) is not int or not 0 <= slot[2] <= 4 or slot in slots or rid in rows
                or type(row["output_tokens"]) is not int or row["output_tokens"] < 0):
            raise ValueError("Invalid, duplicate or out-of-panel response")
        slots[slot] = rid
        rows[rid] = {**row, "candidates": []}
    if set(slots) != {(q, m, d) for q in qids for m in MODELS for d in range(5)}:
        raise ValueError("Incomplete requests; failed draws must remain recorded")
    seen_views, positions = set(), set()
    for view in supervision:
        identity = (view["question_id"], view["pool"])
        if view["split"] != "train" or identity not in panels or identity in seen_views:
            raise ValueError("Unexpected or duplicate supervision view")
        seen_views.add(identity)
        if panels[identity] != [{"id": c["id"], "answer": c["answer"]} for c in view["candidates"]]:
            raise ValueError("Supervision and input surfaces differ")
        qid = view["question_id"]
        aliases = view["gold_aliases"]
        if not aliases or (qid in labels and labels[qid] != aliases):
            raise ValueError("Missing or inconsistent accepted aliases")
        labels[qid] = aliases
        surfaces = set()
        for c in view["candidates"]:
            if c["answer"] in surfaces or c["question_id"] != qid or type(c["official_accepted"]) is not bool:
                raise ValueError("Ambiguous candidate labels")
            surfaces.add(c["answer"])
            for occurrence in c["occurrences"]:
                rid = occurrence["response_id"]
                r = rows[rid]
                slot = (qid, occurrence["model"], occurrence["draw"])
                position = occurrence["position"]
                if (slots.get(slot) != rid or position not in r["retained_positions"]
                        or (rid, position) in positions or not 1 <= position <= 5
                        or (view["pool"] == "max5_greedy") != (r["draw"] == 0)):
                    raise ValueError("Candidate occurrence or decoding mode differs")
                positions.add((rid, position))
                r["candidates"].append({"answer": c["answer"], "position": position,
                    "relation_type": occurrence["relation_type"], "official_accepted": c["official_accepted"]})
    if seen_views != set(panels):
        raise ValueError("Missing supervision views")
    for r in rows.values():
        r["candidates"].sort(key=lambda c: c["position"])
        if [c["position"] for c in r["candidates"]] != sorted(r["retained_positions"]):
            raise ValueError("Missing response surface labels")
        r["accepted"] = any(c["official_accepted"] for c in r["candidates"])
        r["initial_accepted"] = bool(r["candidates"] and r["candidates"][0]["official_accepted"])
        # Strict initial pilot: parser repairs, caps, exclusions and ceilings remain
        # in coverage denominators but cannot create preference training signals.
        r["eligible"] = bool(r["candidates"]) and not r["warnings"] and not r["rejected_candidates"] and r["output_tokens"] < 512
    return questions, labels, rows, slots


def attach_native(native, questions, rows):
    """Bind full natural outputs and exact original prompts to audited occurrences."""
    seen = set()
    for r in native:
        rid = r["response_id"]
        if rid in seen or rid not in rows:
            raise ValueError("Unexpected native response ID")
        seen.add(rid)
        expected = rows[rid]
        for field in ("split", "question_id", "model_key", "draw", "output_tokens"):
            if r[field] != expected[field]:
                raise ValueError("Native response metadata changed")
        prompt = r["prompt"]
        if (len(prompt) != 2 or [p["role"] for p in prompt] != ["system", "user"]
                or json.loads(prompt[1]["content"]) != questions[r["question_id"]]):
            raise ValueError("Native generator prompt/evidence differs")
        candidates, _, _ = sd.parse_surfaces(r["raw_response"])
        if [(c["answer"], c["raw_position"], c["relation_type"]) for c in candidates] != [
                (c["answer"], c["position"], c["relation_type"]) for c in expected["candidates"]]:
            raise ValueError("Native parsing differs from officially labelled audit")
        expected.update(raw_response=r["raw_response"], prompt=prompt)
    if seen != set(rows):
        raise ValueError("Incomplete native fitting response bank")


def build_audit(questions, aliases, rows, slots, cap=2, seed=3407):
    if cap < 1:
        raise ValueError("Positive pair cap required")
    stats = {m: Counter() for m in MODELS}
    pairs, outcomes = [], []
    ensemble = Counter()
    for qid in sorted(questions):
        group = {m: [rows[slots[qid, m, d]] for d in range(5)] for m in MODELS}
        sampled = [r for rr in group.values() for r in rr[1:]]
        successful = [r for r in sampled if r["accepted"]]
        ensemble["questions"] += 1
        ensemble["sampling4_covered"] += bool(successful)
        ensemble["greedy_covered"] += any(rr[0]["accepted"] for rr in group.values())
        for model, rr in group.items():
            s = stats[model]
            s["questions"] += 1
            s["greedy_covered"] += rr[0]["accepted"]
            s["sampling4_covered"] += any(r["accepted"] for r in rr[1:])
            s["all_five_uncovered"] += not any(r["accepted"] for r in rr)
            s["sampled_requests"] += 4
            s["ineligible_sampled_requests"] += sum(not r["eligible"] for r in rr[1:])
            other_sampled = [r for r in sampled if r["model_key"] != model]
            companions_miss = not any(r["accepted"] for r in other_sampled)
            positive = [r for r in rr[1:] if r["eligible"] and r["accepted"]]
            negative = [r for r in rr[1:] if r["eligible"] and not r["accepted"]]
            s["sampled_mixed_questions"] += bool(positive and negative)
            s["sampled_mixed_companions_uncovered_questions"] += bool(positive and negative and companions_miss)
            groups = {"sampled_response_coverage": list(itertools.product(positive, negative)),
                "greedy_recovery": [(a, rr[0]) for a in positive if rr[0]["eligible"] and not rr[0]["accepted"]]}
            outcomes.append({"question_id": qid, "split": "train", "model_key": model,
                "greedy_covered": rr[0]["accepted"], "sampling4_covered": any(r["accepted"] for r in rr[1:]),
                "sampled_successes": sum(r["accepted"] for r in rr[1:]), "companions_sampling4_uncovered": companions_miss,
                "eligible_positive_draws": [r["draw"] for r in positive], "eligible_negative_draws": [r["draw"] for r in negative]})
            for objective, candidates in groups.items():
                seen, choices = set(), []
                for chosen, rejected in candidates:
                    signature = tuple(tuple((c["answer"], c["relation_type"]) for c in r["candidates"]) for r in (chosen, rejected))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    if "prompt" in chosen and chosen["prompt"] != rejected["prompt"]:
                        raise ValueError("Pair prompt differs")
                    if objective == "sampled_response_coverage":
                        fixed = [r for r in sampled if r["response_id"] != chosen["response_id"]]
                    else:
                        fixed = [v[0] for m, v in group.items() if m != model]
                    fixed_covered = any(r["accepted"] for r in fixed)
                    lexical = sorted({flag for a in chosen["candidates"] if a["official_accepted"]
                        for b in rejected["candidates"] for flag in sd.close_flags(a["answer"], b["answer"])[0]})
                    identity = hashlib.sha256(json.dumps([qid, model, objective, chosen["response_id"], rejected["response_id"]]).encode()).hexdigest()
                    choices.append({"pair_id": identity, "question_id": qid, "split": "train", "model_key": model,
                        "objective": objective, "chosen_id": chosen["response_id"], "rejected_id": rejected["response_id"],
                        "chosen_draw": chosen["draw"], "rejected_draw": rejected["draw"],
                        "chosen": chosen.get("raw_response"), "rejected": rejected.get("raw_response"),
                        "prompt": chosen.get("prompt"), "chosen_candidates": chosen["candidates"],
                        "rejected_candidates": rejected["candidates"], "gold_aliases": aliases[qid],
                        "chosen_initial_accepted": chosen["initial_accepted"], "rejected_initial_accepted": rejected["initial_accepted"],
                        "chosen_candidate_count": len(chosen["candidates"]), "rejected_candidate_count": len(rejected["candidates"]),
                        "companions_sampling4_uncovered": companions_miss,
                        "fixed_response_ids": [r["response_id"] for r in fixed],
                        "fixed_context": "eleven_other_sampled_draws" if objective == "sampled_response_coverage" else "two_other_greedy_responses",
                        "counterfactual_ensemble_coverage_gain": int(not fixed_covered),
                        "counterfactual_interpretation": "Replace the chosen draw's slot with either saved response; no new generation or measured improvement.",
                        "review_flags": lexical, "semantic_review_status": "pending", "training_ready": False,
                        "synthetic_chosen": False, "question": questions[qid]["question"], "snippets": questions[qid]["snippets"]})
                s[objective + "_pairs_before_cap"] += len(choices)
                # No length preference: keep complementary cases first, then a
                # seeded identity order. Each objective has its own cap.
                choices.sort(key=lambda p: (-p["counterfactual_ensemble_coverage_gain"],
                    hashlib.sha256(f"{seed}:{p['pair_id']}".encode()).hexdigest()))
                selected = choices[:cap]
                pairs.extend(selected)
                s[objective + "_pairs"] += len(selected)
                s[objective + "_questions"] += bool(selected)
                s[objective + "_counterfactual_gain_questions"] += any(p["counterfactual_ensemble_coverage_gain"] for p in selected)
    return {"fitting_questions": len(questions), "sampled_responses_per_model": 4, "greedy_responses_per_model": 1,
        "models": {m: dict(s) for m, s in stats.items()}, "ensemble": dict(ensemble),
        "candidate_pairs": len(pairs), "distinct_pair_questions": len({p["question_id"] for p in pairs}),
        "responses_requiring_review": len({p[k] for p in pairs for k in ("chosen_id", "rejected_id")}),
        "native_text_available": all("raw_response" in r for r in rows.values()),
        "training_ready": False, "training": False, "new_generation": False, "paid_judging": False,
        "validation_pairs": 0, "dev160_pairs": 0, "test95_pairs": 0,
        "interpretation": "Fitting diagnostics and natural provisional pairs; not a DPO-trained model or observed coverage improvement.",
        "semantic_labels": "Official nonacceptance never implies medical incorrectness; evidence/alias/scope review unresolved."}, pairs, outcomes


def report(summary):
    lines = ["# Coverage-focused generator DPO: fitting-only audit", "",
        "Saved max-five SFT banks: one greedy and four sampled responses per model/question. No training or new generation.", "",
        "| Model | Greedy covered | Sampling4 covered | Mixed success/failure questions | Mixed + companions miss | Sampled pairs | Greedy-recovery pairs |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m, s in summary["models"].items():
        lines.append(f"| {m} | {s['greedy_covered']} | {s['sampling4_covered']} | {s['sampled_mixed_questions']} | "
            f"{s['sampled_mixed_companions_uncovered_questions']} | {s['sampled_response_coverage_pairs']} | {s['greedy_recovery_pairs']} |")
    e = summary["ensemble"]
    lines.extend(["", f"Fitting ensemble coverage: greedy {e['greedy_covered']}/{summary['fitting_questions']}; sampling4 {e['sampling4_covered']}/{summary['fitting_questions']}.",
        f"Provisional pairs: {summary['candidate_pairs']} across {summary['distinct_pair_questions']} distinct fitting questions.",
        f"Full native response text attached: {summary['native_text_available']}.", "",
        "| Model | Questions with fixed-eleven sampling coverage gain | Questions with greedy-ensemble recovery |",
        "|---|---:|---:|"])
    for model, s in summary["models"].items():
        lines.append(f"| {model} | {s['sampled_response_coverage_counterfactual_gain_questions']} | {s['greedy_recovery_counterfactual_gain_questions']} |")
    lines.extend(["",
        "## Interpretation", "",
        "- These pairs prefer response-level official acceptance, not fewer candidates or shorter text.",
        "- A response success/failure pair does not necessarily change ensemble coverage. Companion-miss and fixed-context counterfactual gains are reported separately.",
        "- For sampled pairs, the chosen draw's slot is replaced with either response while the other eleven sampled draws remain fixed. Greedy-recovery pairs keep the other two models' greedy responses fixed.",
        "- A counterfactual gain reuses saved responses; it is not evidence that training will cover new questions. All-bank failures require additional positive examples.",
        "- Failed, repaired, capped, excluded and ceiling draws remain in coverage denominators but cannot form this strict pilot's pairs.",
        "- All accepted aliases were used by the existing official Java matching. Official nonacceptance is not a semantic incorrectness label; wording, evidence, scope and aliases still require review.",
        "- Both objectives are exported separately in metadata. Pair caps apply per question, model and objective; response pairs and views are not independent questions.",
        "- Validation, dev160 and test95 contribute membership exclusions only; no preference pairs use them.",
        "- Review files are not training-approved DPO datasets. No GPU training, paid judging, reranker fitting or test evaluation occurred.",
        "", "## Next step", "", "Review natural successes/failures, especially near-equivalent wording and required expansions. Then freeze one coverage objective and a matched additional-SFT control before preparing a GPU pilot.", ""])
    return "\n".join(lines)


def prepare(repository, audit, submission):
    print("Validating existing official labels and fitting/held-out membership...", flush=True)
    inputs, supervision, responses, protected, original = load_inputs(audit)
    questions, _, rows, _ = index_responses(inputs, supervision, responses, protected)
    provenance = sd.read(audit / "source_provenance.json")
    native = []
    for model in MODELS:
        print(f"Binding full native fitting responses: {model}", flush=True)
        pin = provenance["native_banks"][model]
        bank = Path(pin["directory"])
        sd.verify_files(bank, {"manifest.json": pin["manifest_sha256"], "responses.jsonl": pin["responses_sha256"]})
        native.extend(r for r in sd.records(bank / "responses.jsonl") if r["split"] == "train")
    attach_native(native, questions, rows)
    submission.mkdir(parents=True, exist_ok=False)
    for name, value in (("train_inputs", inputs), ("train_supervision", supervision), ("response_audit", responses), ("native_responses", native)):
        sd.jsonl(submission / f"{name}.jsonl", value)
    sd.write(submission / "protected_ids.json", protected)
    sd.write(submission / "source.json", {"audit": str(audit), "manifest_sha256": sd.digest(audit / "manifest.json"),
        "source_output_sha256": original["output_sha256"], "provenance": provenance})
    from scripts.train_sft_ensemble_reranker import snapshot_code
    snapshot_code(repository, submission / "code")
    shutil.copy2(repository / "gadi_sft_8b_starter/jobs/audit_generator_coverage_dpo.pbs", submission / "job.pbs")
    pins = {p.relative_to(submission).as_posix(): sd.digest(p) for p in sorted(submission.rglob("*")) if p.is_file()}
    sd.write(submission / "prepared.json", {"schema": SCHEMA, "files_sha256": pins, "question_counts": sd.COUNTS,
        "cap_per_question_model_objective": 2, "seed": 3407, "training": False})
    frozen_audit(audit)
    print("Prepared fitting-only coverage audit for all three SFT generators; full native responses and official labels frozen.")


def run(output, audit=None, submission=None, archive=None):
    if output.exists() or (archive and (archive.exists() or archive.resolve().is_relative_to(output.resolve()))):
        raise ValueError("Need new output and an optional new archive outside it")
    if submission:
        source = sd.read(submission / "prepared.json")
        if source["schema"] != SCHEMA or source["question_counts"] != sd.COUNTS:
            raise ValueError("Unsupported prepared coverage audit")
        sd.verify_files(submission, source["files_sha256"])
        required = {"train_inputs.jsonl", "train_supervision.jsonl", "response_audit.jsonl", "native_responses.jsonl", "protected_ids.json", "job.pbs", "code/scripts/audit_generator_coverage_dpo.py"}
        if not required <= set(source["files_sha256"]):
            raise ValueError("Missing frozen coverage inputs")
        inputs, supervision, responses = (sd.records(submission / f"{n}.jsonl") for n in ("train_inputs", "train_supervision", "response_audit"))
        protected = sd.read(submission / "protected_ids.json")
        provenance = {"submission": str(submission), "prepared_sha256": sd.digest(submission / "prepared.json")}
    else:
        inputs, supervision, responses, protected, source = load_inputs(audit)
        provenance = {"audit": str(audit), "source_manifest_sha256": sd.digest(audit / "manifest.json")}
    questions, aliases, rows, slots = index_responses(inputs, supervision, responses, protected)
    if submission:
        attach_native(sd.records(submission / "native_responses.jsonl"), questions, rows)
    summary, pairs, outcomes = build_audit(questions, aliases, rows, slots)
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema": SCHEMA, "status": "running", **provenance, "training_ready": False, "training": False}
    sd.write(output / "manifest.json", state)
    try:
        sd.write(output / "summary.json", summary)
        sd.jsonl(output / "preference_candidates.jsonl", pairs)
        sd.jsonl(output / "question_outcomes.jsonl", outcomes)
        sample = []
        for model in MODELS:
            for objective in OBJECTIVES:
                pp = [p for p in pairs if p["model_key"] == model and p["objective"] == objective]
                sd.rng(3407, model, objective, "review").shuffle(pp)
                sample.extend(pp[:20])
        sd.jsonl(output / "review_sample.jsonl", sample)
        used = {p[k] for p in pairs for k in ("chosen_id", "rejected_id")}
        sd.jsonl(output / "review_responses.jsonl", [rows[rid] for rid in sorted(used)])
        (output / "report.md").write_text(report(summary), encoding="utf-8")
        if submission:
            sd.verify_files(submission, source["files_sha256"])
        else:
            frozen_audit(audit)
        state.update(status="complete", output_sha256={p.name: sd.digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"})
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sd.write(output / "manifest.json", state)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "x:gz") as handle:
            for path in sorted(output.iterdir()):
                if path.is_file():
                    handle.add(path, arcname=path.name, recursive=False)
        print("Archive:", archive)
    print(report(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "audit", "local"), required=True)
    p.add_argument("--repository", type=Path, default=ROOT)
    p.add_argument("--selector-audit", type=Path)
    p.add_argument("--submission", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        if not a.selector_audit or not a.submission:
            p.error("Preparation requires --selector-audit and --submission")
        prepare(a.repository.resolve(), a.selector_audit.resolve(), a.submission.resolve())
    else:
        if not a.output or (a.mode == "local" and not a.selector_audit) or (a.mode == "audit" and not a.submission):
            p.error("Need --output and the matching source argument")
        run(a.output.resolve(), audit=a.selector_audit.resolve() if a.mode == "local" else None,
            submission=a.submission.resolve() if a.mode == "audit" else None, archive=a.archive.resolve() if a.archive else None)


if __name__ == "__main__":
    main()
