#!/usr/bin/env python3
"""Prepare an inspected answer-form pilot; never call an API or train a model."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_answer_selector_data as data
from scripts import judge_generator_coverage_dpo as judge

SCHEMA = "generator-official-form-pilot-v1"
DEFAULT_REVIEW = ROOT / "configs/generator_form_pilot_inspection.json"


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sample_questions(pairs, per_model=24):
    """Take a deterministic question sample per model without using pair outcomes."""
    return sorted({q for model in data.MODELS for q in sorted(
        {p["question_id"] for p in pairs if p["model_key"] == model},
        key=lambda q: hashlib.sha256(("form-pilot-v1:" + q).encode()).hexdigest())[:per_model]})


def select(pairs, inspection):
    if inspection["objective"] != "official_answer_form_coverage":
        raise ValueError("This pilot does not certify semantic error corrections")
    full = inspection.get("scope") == "all_pairs"
    if full and (inspection.get("expected_pair_count") != len(pairs) or
                 inspection.get("expected_question_count") != len({p["question_id"] for p in pairs})):
        raise ValueError("Incomplete full-review panel")
    sampled = (sorted({p["question_id"] for p in pairs}) if full else
               sample_questions(pairs, inspection["questions_per_model"]))
    if set(inspection["decisions"]) != set(sampled):
        raise ValueError("Inspection must cover the entire prespecified sample or full panel")
    by_question = defaultdict(list)
    for p in pairs:
        by_question[p["question_id"]].append(p)
    for q, decision in inspection["decisions"].items():
        allowed = ("retain", "hold", "conditional") if full else ("retain", "hold")
        if decision["decision"] not in allowed or not decision["reason"]:
            raise ValueError("Missing inspection decision/reason")
        surfaces = {c["answer"] for p in by_question[q] for side in ("chosen", "rejected")
                    for c in p[side + "_candidates"]}
        if full:
            if (decision.get("surfaces_sha256") != identity(sorted(surfaces)) or
                    decision.get("pair_ids_sha256") != identity(sorted(p["pair_id"] for p in by_question[q]))):
                raise ValueError("Inspected surfaces or pair membership changed")
            declared = set(decision.get("blocked_surfaces", [])) | set(decision.get("allow_surfaces", []))
            groups = decision.get("equivalence_groups", {})
            grouped = [s for group in groups.values() for s in group]
            if (not declared <= surfaces or not set(grouped) <= surfaces or len(grouped) != len(set(grouped))
                    or (decision.get("allow_surfaces") and decision.get("blocked_surfaces"))
                    or (groups and declared)
                    or (decision["decision"] == "conditional" and not (declared or groups))):
                raise ValueError("Invalid or stale pair-level surface rule")
        if decision["decision"] == "retain" or full:
            snippets = by_question[q][0]["snippets"]
            evidence = [decision.get("evidence", {}), *decision.get("additional_evidence", [])]
            for item in evidence:
                if (not item.get("quote") or not any(s["id"] == item.get("snippet_id")
                        and item["quote"] in s["text"] for s in snippets)):
                    raise ValueError("Inspection evidence is not a literal source quote")
    selected, exclusions, groups = [], [], defaultdict(list)
    for p in pairs:
        q = p["question_id"]
        d = inspection["decisions"].get(q)
        reason = ("outside_inspected_sample" if d is None else
                  "assistant_scope_or_evidence_hold" if d["decision"] == "hold" else None)
        hits = []
        if d and d["decision"] == "conditional":
            contents = {c["answer"] for side in ("chosen", "rejected") for c in p[side + "_candidates"]}
            hits = sorted(contents & set(d.get("blocked_surfaces", [])))
            if d.get("allow_surfaces"):
                hits = sorted(contents - set(d["allow_surfaces"]))
            if hits:
                reason = "pair_surface_scope_or_form_hold"
            if d.get("equivalence_groups") and not any(contents <= set(group) for group in d["equivalence_groups"].values()):
                reason = "different_or_unreviewed_equivalence_group"
        if reason:
            exclusions.append({"pair_id": p["pair_id"], "question_id": q, "reason": reason,
                **({"inspection_reason": d["reason"], "flagged_surfaces": hits} if full else {})})
            continue
        if (p["review_category"] != "benchmark_form_preference" or p["split"] != "train"
                or p["synthetic_chosen"] or p["chosen"] == p["rejected"]
                or not any(c["official_accepted"] for c in p["chosen_candidates"])
                or any(c["official_accepted"] for c in p["rejected_candidates"])):
            raise ValueError("Invalid natural official-form preference")
        # Preserve surface order and punctuation; only identical native strings
        # are duplicates. Collapse repetitions across objective/draw metadata.
        groups[p["model_key"], q, identity([p["prompt"], p["chosen"], p["rejected"]])].append(p)
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda p: p["pair_id"])
        p = rows[0]
        selected.append({**p, "source_pair_ids": [r["pair_id"] for r in rows],
                         "source_objectives": sorted({r["objective"] for r in rows})})
        exclusions.extend({"pair_id": r["pair_id"], "question_id": r["question_id"],
                           "reason": "duplicate_native_preference"} for r in rows[1:])
    return selected, exclusions, sampled


def assign_splits(rows, seed=3407):
    """One shared question split across models; hold out 20%, minimum two/model."""
    questions = sorted({p["question_id"] for p in rows}, key=lambda q: identity([seed, q]))
    holdout = set(questions[:max(2, round(len(questions) * .2))])
    for model in data.MODELS:
        qs = [q for q in questions if any(p["model_key"] == model and p["question_id"] == q for p in rows)]
        if len(qs) < 4:
            raise ValueError("Too few inspected questions for " + model)
        for q in qs:
            if len(set(qs) & holdout) >= 2:
                break
            holdout.add(q)
    # Adding a shared holdout for another model can exhaust a small panel.
    for model in data.MODELS:
        if len({p["question_id"] for p in rows if p["model_key"] == model} - holdout) < 2:
            raise ValueError("Shared split leaves too few training questions for " + model)
    return {q: "inner_validation" if q in holdout else "train" for q in questions}


def load_verified(source, reviewed, selector):
    original, raw = judge.load_source(source, selector)
    state = data.read(reviewed / "manifest.json")
    required = {"reviewed_responses.jsonl", "pair_review.jsonl", "benchmark_form_preference.jsonl"}
    if (state.get("status") != "complete" or not required <= set(state.get("output_sha256", {}))
            or state["source_manifest_sha256"] != data.digest(source / "manifest.json")
            or state["selector_manifest_sha256"] != data.digest(selector / "manifest.json")):
        raise ValueError("Need a complete review bound to the same source and fitting audit")
    data.verify_files(reviewed, state["output_sha256"])
    annotations = data.records(reviewed / "reviewed_responses.jsonl")
    annotated = {r["response_id"]: r for r in annotations}
    if len(annotated) != len(raw) or len(annotations) != len(annotated):
        raise ValueError("Missing/duplicate reviewed responses")
    for r in raw:
        a = annotated[r["response_id"]]
        # Compare the original fields separately from the added judge decisions.
        if any(a.get(k) != v for k, v in r.items() if k not in ("candidates", "reviewer")):
            raise ValueError("Native response metadata changed in review")
        if len(a["candidates"]) != len(r["candidates"]) or any(
                any(c.get(k) != v for k, v in old.items())
                for c, old in zip(a["candidates"], r["candidates"])):
            raise ValueError("Reviewed candidate surfaces or labels changed")
    reconstructed = []
    for p in original:
        category, reason = judge.classify_pair(p, annotated)
        if category == "benchmark_form_preference":
            reconstructed.append({**p, "review_category": category, "review_reason": reason,
                "semantic_review_status": "LLM-only provisional review", "training_ready": False})
    if reconstructed != data.records(reviewed / "benchmark_form_preference.jsonl"):
        raise ValueError("Shortlist does not reconstruct from native responses and review")
    return reconstructed


def build(source, reviewed, selector, output, inspection_path=DEFAULT_REVIEW, archive=None):
    if output.exists():
        raise FileExistsError(output)
    pairs = load_verified(source, reviewed, selector)
    inspection = data.read(inspection_path)
    if (inspection["review_manifest_sha256"] != data.digest(reviewed / "manifest.json")
            or inspection["shortlist_sha256"] != data.digest(reviewed / "benchmark_form_preference.jsonl")):
        raise ValueError("Inspection belongs to a different judged shortlist")
    selected, exclusions, sampled = select(pairs, inspection)
    full = inspection.get("scope") == "all_pairs"
    split = assign_splits(selected, inspection["seed"])
    output.mkdir(parents=True)
    data.write(output / "inspection.json", inspection)
    data.write(output / "question_split.json", split)
    data.jsonl(output / "selected_pairs.jsonl", selected)
    data.jsonl(output / "exclusions.jsonl", exclusions)
    # Store complete evidence and every candidate for the inspected sample,
    # including held cases. Nothing is silently shortened for the model.
    data.jsonl(output / "inspection_cases.jsonl", [p for p in pairs if p["question_id"] in sampled])
    retained_ids = {pid for p in selected for pid in p["source_pair_ids"]}
    representative_ids = {p["pair_id"] for p in selected}
    excluded_by_id = {e["pair_id"]: e for e in exclusions}
    pair_decisions = []
    if full:
        for p in pairs:
            d = inspection["decisions"][p["question_id"]]
            pair_decisions.append({"pair_id": p["pair_id"], "question_id": p["question_id"],
                "model_key": p["model_key"], "chosen_id": p["chosen_id"], "rejected_id": p["rejected_id"],
                "decision": "retain_for_form_pilot" if p["pair_id"] in retained_ids else "hold",
                "duplicate_of_retained_preference": p["pair_id"] in retained_ids - representative_ids,
                "reason": d["reason"], "trigger": excluded_by_id.get(p["pair_id"], {}).get("reason"),
                "flagged_surfaces": excluded_by_id.get(p["pair_id"], {}).get("flagged_surfaces", []),
                "evidence": [d["evidence"], *d.get("additional_evidence", [])], "expert_certified": False})
        data.jsonl(output / "pair_decisions.jsonl", pair_decisions)
        # These are possible question-answer requirement repairs, not proofs of
        # medically false rejected text, and never enter the form-pilot files.
        requirement_ids = {"530a5117970c65fa6b000007", "5d35be1cb3a6380763000005"}
        requirement_candidates = [p for p in pairs if p["question_id"] in requirement_ids and p["pair_id"] not in retained_ids]
        data.jsonl(output / "answer_requirement_review_candidates.jsonl", requirement_candidates)
    counts = {}
    for model in data.MODELS:
        counts[model] = {}
        for name in ("train", "inner_validation"):
            rows = [p for p in selected if p["model_key"] == model and split[p["question_id"]] == name]
            dpo = [{k: p[k] for k in ("question_id", "prompt", "chosen", "rejected", "source_pair_ids", "source_objectives")}
                   for p in rows]
            # Intentional pair-matched repetitions give the SFT control the same
            # preferred-completion exposure and number of optimizer examples.
            sft = [{"question_id": p["question_id"], "source_pair_ids": p["source_pair_ids"],
                    "messages": [*p["prompt"], {"role": "assistant", "content": p["chosen"]}]} for p in rows]
            data.jsonl(output / model / f"dpo_{name}.jsonl", dpo)
            data.jsonl(output / model / f"sft_{name}.jsonl", sft)
            counts[model][name] = {"pairs": len(rows), "questions": len({p["question_id"] for p in rows}),
                "objectives": dict(Counter(o for p in rows for o in p["source_objectives"]))}
    summary = {"shortlist_pairs": len(pairs), "shortlist_questions": len({p["question_id"] for p in pairs}),
        "shortlist_exact_duplicate_preferences": len(pairs) - len({identity([p["model_key"], p["prompt"], p["chosen"], p["rejected"]]) for p in pairs}),
        "inspection_scope": "all_pairs" if full else "sample",
        "inspected_questions": len(sampled), "inspected_pairs": sum(p["question_id"] in sampled for p in pairs),
        **({} if full else {"sampled_questions": len(sampled), "sampled_pairs": sum(p["question_id"] in sampled for p in pairs)}),
        "retained_questions": len(split), "retained_unique_pairs": len(selected),
        "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)), "models": counts,
        "training_ready": False, "status": "prepared_for_tokenizer_and_native_source_preflight",
        "objective": "official_answer_form_coverage", "expert_review": False,
        "semantic_error_correction_claim": False, "training_launched": False,
        "preferred_character_length": dict(Counter("longer" if len(p["chosen"]) > len(p["rejected"]) else
            "shorter" if len(p["chosen"]) < len(p["rejected"]) else "equal" for p in selected)),
        "preferred_candidate_count": dict(Counter("more" if p["chosen_candidate_count"] > p["rejected_candidate_count"] else
            "fewer" if p["chosen_candidate_count"] < p["rejected_candidate_count"] else "equal" for p in selected)),
        "representative_saved_swap_gain_pairs": sum(p["counterfactual_ensemble_coverage_gain"] for p in selected),
        "representative_saved_swap_gain_questions": len({p["question_id"] for p in selected if p["counterfactual_ensemble_coverage_gain"]}),
        "note": ("All shortlisted pairs received decisions; holds are exclusions from this objective, not medical-error verdicts."
                 if full else "Small inspected pilot; unreviewed shortlist records are excluded, not presumed incorrect.")}
    if full:
        summary.update(retained_pair_records=len(retained_ids), held_pair_records=len(pairs) - len(retained_ids),
            retained_duplicate_records=len(retained_ids) - len(selected), uninspected_pair_records=0,
            answer_requirement_review_candidates=len(requirement_candidates),
            answer_requirement_candidate_questions=len({p["question_id"] for p in requirement_candidates}))
    data.write(output / "summary.json", summary)
    plan = {"arms_per_model": ["starting_max5_sft", "additional_sft_preferred_control", "dpo_official_form"],
        "initial_policy_and_dpo_reference": "same exact max-five SFT that generated the frozen native bank",
        "proposed_shared_hyperparameters": {"seed": 3407, "beta": .1, "learning_rate": 1e-6, "epochs": 1},
        "preflight_required": ["verify original Gadi bank and SFT adapter hashes", "render native chat templates",
            "count prompt plus completion tokens on both sides; reject over-context records without truncation",
            "GPU smoke before training; frozen SFT reference log probabilities before updates"],
        "pair_loss_panel": "inner_validation contains fitting questions only; shared split across all models",
        "generation_selection_panel": "all 144 validation questions, including uncovered questions; epoch zero remains a choice",
        "primary_metric": "official full-pool coverage at fixed generation budget",
        "secondary_metrics": ["MRR@5", "top-five coverage", "per-question gains and losses", "tokens and unique pool size"],
        "inference": "greedy and independent sampling; identical prompts, seeds, budgets, companion models and frozen reranker across arms",
        "first_generation_comparison": "four sampled draws per model matches source banks; ten-sample confirmation is a separate fixed-budget comparison",
        "model_comparison_limit": "Reviewed counts differ by generator; compare DPO with each model's own controls rather than attributing differences solely to backbone",
        "first_evaluation": "Swap one generator at a time; keep the two companion generators at their starting SFTs. An all-DPO ensemble is a separate subsequent comparison.",
        "held_out": "No dev160/test95 preferences; defer another test95 evaluation until protocol fixed",
        "scope": "Response-level official acceptance; this is not pool-aware generation or joint system-level DPO",
        "interpretation": "Do not equate preference-loss gains or saved-response swaps with improved unseen coverage"}
    data.write(output / "pilot_plan.json", plan)
    lines = ["# Inspected official-answer-form generator pilot", "",
        f"Programmatic checks covered {len(pairs)} shortlisted pairs. Assistant triage examined {len(sampled)} fitting questions",
        f"({summary['inspected_pairs']} pairs), retaining {len(selected)} unique native preferences on {len(split)} questions.",
        "This is assistant evidence inspection following automated judgments, not expert biomedical certification.", "",
        "| Model | Training pairs / questions | Inner holdout pairs / questions |", "|---|---:|---:|"]
    for m, c in counts.items():
        lines.append(f"| {m} | {c['train']['pairs']} / {c['train']['questions']} | {c['inner_validation']['pairs']} / {c['inner_validation']['questions']} |")
    lines += ["", "The inner holdout comes exclusively from the 1,296 fitting questions. It is not validation144.",
        "Native prompts and completions are unchanged. Official aliases and review decisions are not part of model messages.",
        "Repeated identical preferences are collapsed across draws/objectives; the retained source IDs preserve their provenance.", "",
        "## What the experiment tests", "",
        "Both sides were judged to supply supported answers. DPO prefers a response containing an officially accepted surface.",
        "This may improve benchmark-form coverage; it is not supervision for choosing a different biomedical concept.",
        f"Only {summary['representative_saved_swap_gain_pairs']} representative pairs on {summary['representative_saved_swap_gain_questions']} question(s) show fixed-companion coverage gain in saved-response swaps. These are fitting diagnostics, not training results.",
        "Chosen responses can be longer and contain more candidates. There is no preference for compactness in this objective.",
        "For the retained PAR2 question, a swap changes '320 kbp' to accepted '320kbp': a scoring-form effect, not a different quantity or new knowledge.",
        "Compare each model's unchanged SFT, additional preferred-only SFT, and DPO from the same SFT. Freeze generation budgets and the reranker.",
        "Keep scope, numerical-range, precision, answer-type and insufficient-evidence cases out of this pilot.", "",
        "## Status", "", "Data and experimental plan only. No API calls, training, submission or held-out evaluation occurred.",
        "training_ready=false until native-source and tokenizer/context preflight plus a GPU smoke check are implemented and pass.",
        "Preference counts support an exploratory pilot, not a reliable effectiveness claim or full-scale training recommendation.",
        ("All shortlisted records have been inspected. Held pairs require a different objective or further evidence review; do not add them automatically."
         if full else "Expand inspection of the remaining shortlist before spending GPU time on substantive coverage training. Do not train on all 825 by default."), "",
        "## Inspection decisions", "", "| Question | Decision | Reason |", "|---|---|---|"]
    lines += [f"| {q} | {d['decision']} | {d['reason']} |" for q, d in sorted(inspection["decisions"].items())]
    if full:
        lines[0] = "# Full inspection of official-answer-form generator preferences"
        lines[4:4] = ["All 825 shortlisted records, including the previously sampled records, were assessed against explicit question/surface rules.",
            "The assistant inspected every candidate surface and cited evidence excerpt, opening full snippets for ambiguous cases. This is not expert verification of every underlying biomedical claim.",
            "Each retained pair requires both full candidate lists to pass the same rule. A problematic response does not discard unrelated clean pairs for that question.",
            f"Retained source records: {len(retained_ids)}; held: {len(pairs)-len(retained_ids)}; retained exact duplicates collapsed: {len(retained_ids)-len(selected)}; uninspected: 0.",
            "See pair_decisions.jsonl and pair_decisions.md for one decision per original pair; full native records are in inspection_cases.jsonl.", ""]
        cases = ["# Pair-by-pair assistant inspection", "",
            "All records are fitting-only. Chosen/rejected are optimization roles; rejected does not mean medically wrong.",
            "Both original candidate lists and their relation tags are shown. Complete native strings, evidence and provenance are in inspection_cases.jsonl.", ""]
        for i, (p, d) in enumerate(zip(pairs, pair_decisions), 1):
            cases += [f"## Pair {i}: {p['question_id']} / {p['model_key']}", "",
                f"Pair ID: `{p['pair_id']}`", "", p["question"], "",
                f"Decision: **{d['decision']}**. {d['reason']}",
                f"Flagged expressions: {json.dumps(d['flagged_surfaces'], ensure_ascii=False)}", "",
                "Chosen candidate list:", "```json", json.dumps(p["chosen_candidates"], ensure_ascii=False, indent=2), "```",
                "Rejected candidate list:", "```json", json.dumps(p["rejected_candidates"], ensure_ascii=False, indent=2), "```", "",
                "Evidence checked:", *[f"- Snippet {e['snippet_id']}: {e['quote']}" for e in d["evidence"]], ""]
        (output / "pair_decisions.md").write_text("\n".join(cases), encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {p.relative_to(output).as_posix(): data.digest(p) for p in output.rglob("*") if p.is_file()}
    data.write(output / "manifest.json", {"schema": SCHEMA, **summary,
        "source_manifest_sha256": data.digest(source / "manifest.json"),
        "review_manifest_sha256": data.digest(reviewed / "manifest.json"),
        "selector_manifest_sha256": data.digest(selector / "manifest.json"),
        "inspection_sha256": data.digest(inspection_path), "files_sha256": hashes})
    validate(output)
    if archive:
        if archive.exists():
            raise FileExistsError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    handle.add(path, arcname=path.relative_to(output).as_posix())
    return summary


def validate(output):
    state = data.read(output / "manifest.json")
    if state.get("schema") != SCHEMA or state.get("training_ready") is not False:
        raise ValueError("Wrong pilot status")
    data.verify_files(output, state["files_sha256"])
    selected = data.records(output / "selected_pairs.jsonl")
    split = data.read(output / "question_split.json")
    inspection = data.read(output / "inspection.json")
    if inspection.get("scope") == "all_pairs":
        reconstructed, excluded, _ = select(data.records(output / "inspection_cases.jsonl"), inspection)
        if reconstructed != selected or excluded != data.records(output / "exclusions.jsonl"):
            raise ValueError("Full-review selections do not reconstruct")
        decisions = data.records(output / "pair_decisions.jsonl")
        cases = data.records(output / "inspection_cases.jsonl")
        if (len(decisions) != inspection["expected_pair_count"] or
                [d["pair_id"] for d in decisions] != [p["pair_id"] for p in cases]):
            raise ValueError("Incomplete pair-decision coverage")
        retained_ids = {pid for p in selected for pid in p["source_pair_ids"]}
        representatives = {p["pair_id"] for p in selected}
        by_exclusion = {e["pair_id"]: e for e in excluded}
        for p, d in zip(cases, decisions):
            rule = inspection["decisions"][p["question_id"]]
            expected = {k: p[k] for k in ("pair_id", "question_id", "model_key", "chosen_id", "rejected_id")}
            expected.update(decision="retain_for_form_pilot" if p["pair_id"] in retained_ids else "hold",
                duplicate_of_retained_preference=p["pair_id"] in retained_ids - representatives,
                reason=rule["reason"], trigger=by_exclusion.get(p["pair_id"], {}).get("reason"),
                flagged_surfaces=by_exclusion.get(p["pair_id"], {}).get("flagged_surfaces", []),
                evidence=[rule["evidence"], *rule.get("additional_evidence", [])], expert_certified=False)
            if d != expected:
                raise ValueError("Pair decision differs from inspected native records/rules")
    if split != assign_splits(selected, inspection["seed"]):
        raise ValueError("Question split changed")
    for model in data.MODELS:
        for name in ("train", "inner_validation"):
            dpo = data.records(output / model / f"dpo_{name}.jsonl")
            sft = data.records(output / model / f"sft_{name}.jsonl")
            rows = [p for p in selected if p["model_key"] == model and split[p["question_id"]] == name]
            expected_dpo = [{k: p[k] for k in ("question_id", "prompt", "chosen", "rejected", "source_pair_ids", "source_objectives")} for p in rows]
            if dpo != expected_dpo or sft != [{"question_id": p["question_id"], "source_pair_ids": p["source_pair_ids"],
                "messages": [*p["prompt"], {"role": "assistant", "content": p["chosen"]}]} for p in rows]:
                raise ValueError("Training files differ from inspected native strings")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--selector-audit", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    a = parser.parse_args()
    print(json.dumps(build(a.source.resolve(), a.reviewed.resolve(), a.selector_audit.resolve(),
        a.output.resolve(), a.inspection.resolve(), a.archive.resolve() if a.archive else None), indent=2))
