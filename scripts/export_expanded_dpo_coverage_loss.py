#!/usr/bin/env python3
"""Export expanded-DPO validation coverage losses using saved responses and labels only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
import types


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def key(answer):
    return " ".join(answer.split()).casefold()


def seed(qid, draw):
    return int.from_bytes(hashlib.sha256(f"3407:{qid}:{draw}".encode()).digest()[:4], "big")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def by_question(rows, ids):
    result = {r["question_id"]: r for r in rows}
    if len(result) != len(rows) or set(result) != ids:
        raise ValueError("Missing or duplicate per-question results")
    return result


def pool_summary(draws):
    unique = {}
    for draw in draws:
        for candidate in draw["answers"]:
            unique.setdefault(key(candidate["answer"]), candidate)
    rows = list(unique.values())
    rank = next((i for i, r in enumerate(rows, 1) if r["official_accepted"]), None)
    return {"unique_candidates": rows, "first_accepted_rank": rank,
            "mrr_at5": 1/rank if rank and rank <= 5 else 0.,
            "coverage_at5": float(rank is not None and rank <= 5), "coverage_full_pool": float(rank is not None),
            "accepted_initial_draws": sum(bool(d["answers"] and d["answers"][0]["official_accepted"]) for d in draws)}


def verify_metrics(calculated, saved):
    if any(abs(calculated[k]-saved[k]) > 1e-12 for k in ("mrr_at5", "coverage_at5", "coverage_full_pool")):
        raise ValueError("Raw response/ordering labels do not reproduce saved question metrics")


def compare_draws(before, after):
    result = []
    for a, b in zip(before, after):
        if a["draw"] != b["draw"] or a["seed"] != b["seed"]:
            raise ValueError("SFT/DPO draw indices or seeds differ")
        aa, bb = a["answers"], b["answers"]
        same_initial = bool(aa and bb and key(aa[0]["answer"]) == key(bb[0]["answer"]))
        kept = {key(c["answer"]) for c in bb}
        result.append({"draw": a["draw"], "same_initial_text": same_initial,
            "sft_initial": aa[0] if aa else None, "dpo_initial": bb[0] if bb else None,
            "removed_accepted_expressions": [c for c in aa if c["official_accepted"] and key(c["answer"]) not in kept],
            "accepted_alternative_removed_with_same_initial": same_initial and any(
                c["official_accepted"] and key(c["answer"]) not in kept for c in aa[1:]),
            "sft": a, "dpo": b})
    return result


def report(cases, epoch, selected):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = [f"# Llama validation coverage losses: epoch {epoch}", "",
        f"Selected epoch remains {selected}. Read-only, outcome-selected internal-validation diagnostics.",
        "Official acceptance does not establish semantic correctness or incorrectness. No new judging, generation, training or selection.", ""]
    for c in cases:
        lines += [f"## {c['question_id']}", "", c["question"], "", "Gold aliases: "+"; ".join(c["gold_aliases"]), "",
            "### Four paired sampled responses", "",
            "| Draw | SFT initial (accepted?) | Epoch initial (accepted?) | SFT full response | Epoch full response |",
            "|---|---|---|---|---|"]
        for d in c["paired_draws"]:
            def initial(x):
                return f"{x['answer']} ({x['official_accepted']})" if x else "EMPTY / unparseable"
            def answers(x):
                return "; ".join(f"{a['answer']} [accepted={a['official_accepted']}]" for a in x["answers"]) or "EMPTY / unparseable"
            lines.append("| "+" | ".join(map(cell, [d["draw"], initial(d["sft_initial"]), initial(d["dpo_initial"]), answers(d["sft"]), answers(d["dpo"])]))+" |")
        for policy in ("sft", "dpo"):
            s = c["llama_sampling4"][policy]
            accepted = [r["answer"] for r in s["unique_candidates"] if r["official_accepted"]]
            lines += ["", f"{policy}: accepted expressions={accepted}; accepted initial draws={s['accepted_initial_draws']}/4."]
        lines += ["", "### Frozen ensemble top five", "", "| Policy | Rank | Expression | Accepted | Sources |", "|---|---:|---|---|---|"]
        for policy, rr in c["ensemble_sampling4"].items():
            for row in rr[:5]:
                lines.append("| "+" | ".join(map(cell,[policy,row["rank"],row["answer"],bool(row["label"]),", ".join(row["sources"])]))+" |")
        lines += ["", "Greedy responses, all raw strings, companion responses and full evidence are included in the JSON files.", ""]
    return "\n".join(lines)


def export(comparison, output, archive=None, epoch=1):
    if epoch not in (1, 2, 3):
        raise ValueError("Expected a retained trained epoch: 1, 2 or 3")
    if output.exists() or output.is_relative_to(comparison) or (archive and (archive.exists() or archive.is_relative_to(comparison))):
        raise ValueError("Choose new export paths outside the original selection directory")
    state = read(comparison / "manifest.json")
    if state.get("status") != "complete" or state.get("smoke_test") is not False or state["validation_question_count"] != 144:
        raise ValueError("Expected completed full validation144 selection")
    source_hashes = {str(comparison / "manifest.json"): digest(comparison / "manifest.json")}
    def checked(path, expected):
        if digest(path) != expected:
            raise ValueError(f"Saved source changed: {path}")
        source_hashes[str(path)] = expected
        return path
    def saved(relative):
        path = (comparison / relative).resolve()
        if not path.is_relative_to(comparison): raise ValueError("Unsafe selection output path")
        return checked(path, state["output_sha256"][relative])
    summary = read(saved("comparison_summary.json"))
    scores = read(saved("comparison_per_question.json"))
    ids = set(state["question_ids"])
    if len(ids) != 144 or summary["question_count"] != 144 or summary["selected_epoch"] != state["selected_epoch"]:
        raise ValueError("Selection membership or selected epoch differs")
    a = by_question(scores["llama_epoch0_sampling4"], ids)
    b = by_question(scores[f"llama_epoch{epoch}_sampling4"], ids)
    lost = sorted(q for q in ids if a[q]["coverage_full_pool"] and not b[q]["coverage_full_pool"])
    recorded = next(r for r in summary["checkpoint_scores"] if r["epoch"] == epoch)
    if len(lost) != recorded["llama_full_pool_lost"]:
        raise ValueError("Reported loss count differs from per-question transitions")
    selected = set(lost)
    examples = by_question(records(saved("validation_inputs.jsonl")), ids)
    prepared = read(checked(Path(state["submission"]) / "prepared.json", state["prepared_sha256"]))
    checked(Path(state["training"]) / "manifest.json", state["training_manifest_sha256"])
    parser_name = "gadi_sft_8b_starter/scripts/run_extractive_expansion_8b.py"
    parser_path = checked(Path(state["submission"]) / "code" / parser_name, prepared["code_hashes"][parser_name])
    # Execute the hashed pure parser without creating __pycache__ in the submission.
    parser = types.ModuleType("frozen_coverage_loss_parser")
    parser.__file__ = str(parser_path)
    exec(compile(parser_path.read_text(encoding="utf-8"), str(parser_path), "exec"), parser.__dict__)
    mapping = read(saved("official/scorer_id_map.json"))
    official = records(saved("official/official_candidate_scores.jsonl"))
    gold = {r["id"]: r for r in read(saved("official/scorer_gold.json"))["questions"]}
    if len(official) != len(mapping) or {r["question_id"] for r in official} != set(mapping) or set(gold) != set(mapping):
        raise ValueError("Incomplete or duplicate cached official labels")
    labels, aliases = {}, {}
    for row in official:
        qid, answer = mapping[row["question_id"]]
        if row["lenient_accuracy"] not in (0, 1): raise ValueError("Invalid official acceptance label")
        if qid not in selected: continue
        identity = (qid, key(answer)); accepted = bool(row["lenient_accuracy"])
        if identity in labels and labels[identity] != accepted: raise ValueError("Normalization changes acceptance")
        labels[identity] = accepted
        g = gold[row["question_id"]]
        if g["body"] != examples[qid]["question"]: raise ValueError("Question text differs from official scoring")
        aa = g["exact_answer"][0]
        if qid in aliases and aliases[qid] != aa: raise ValueError("Gold aliases differ within question")
        aliases[qid] = aa
    def annotate(row, draw, actual_seed, raw, tokens, parsed=None):
        try: answers = parser.parse_equivalent_response(raw, 5)[0]
        except ValueError: answers = []
        if parsed is not None and answers != parsed: raise ValueError("Saved parsed answers differ from raw response")
        if actual_seed != seed(row, draw+1): raise ValueError("Saved sample seed differs from native bank")
        return {"draw": draw, "seed": actual_seed, "raw_response": raw, "output_tokens": tokens,
                "answers": [{**c, "official_accepted": labels[(row, key(c["answer"]))]} for c in answers]}
    raw = {q: {"sft": {}, "dpo": {}} for q in selected}
    source_responses = []
    for model, bank in prepared["native_banks"].items():
        path = Path(bank["path"])
        checked(path / "manifest.json", bank["manifest_sha256"])
        checked(path / "responses.jsonl", bank["responses_sha256"])
        seen = set()
        for row in records(path / "responses.jsonl"):
            qid = row["question_id"]
            if qid not in selected or row["split"] != "validation": continue
            if row["model_key"] != model or (qid, row["draw"]) in seen: raise ValueError("Changed/duplicate source response")
            seen.add((qid, row["draw"]))
            context = json.loads(row["prompt"][1]["content"])
            expected = {"question": examples[qid]["question"], "snippets": [
                {"id": s["snippet_id"], "text": s["text"]} for s in examples[qid]["snippets"]]}
            if context != expected or row["prompt"][0]["content"] != prepared["source"]["system_prompt"]:
                raise ValueError("Source question/evidence/prompt differs")
            r = annotate(qid, row["draw"], row["seed"], row["raw_response"], row["output_tokens"])
            source_responses.append({"question_id": qid, "policy": "sft", "model_key": model, **r})
            if model == "llama31": raw[qid]["sft"][row["draw"]] = r
        if seen != {(q,d) for q in selected for d in range(5)}: raise ValueError("Incomplete source draws")
    for mode, count in (("greedy", 1), ("sampling4", 4)):
        root = f"epoch-{epoch}/{mode}"
        config = read(saved(root+"/config.json"))
        if (config["num_generations"] != count or config["seed"] != 3407 or config["max_candidates"] != 5
                or config["sample_seed_offset"] != (0 if mode == "greedy" else 1)):
            raise ValueError("Epoch decoding budget differs")
        if read(saved(root+"/status.json"))["status"] != "complete": raise ValueError("Incomplete epoch generation")
        for qid, row in by_question(records(saved(root+"/generations.jsonl")), ids).items():
            if qid not in selected: continue
            if row["snippets_truncated"] or row["included_snippets"] != len(examples[qid]["snippets"]) or row["included_snippets"] != row["total_snippets"]:
                raise ValueError("Epoch dropped evidence")
            if len(row["samples"]) != count: raise ValueError("Incomplete epoch draws")
            for i, sample in enumerate(row["samples"], 1):
                if sample["draw"] != i: raise ValueError("Duplicate or unordered epoch draws")
                draw = 0 if mode == "greedy" else i
                r = annotate(qid, draw, sample["seed"], sample["raw_response"], sample["output_tokens"], sample["answers"])
                raw[qid]["dpo"][draw] = r
                source_responses.append({"question_id": qid, "policy": "dpo", "model_key": "llama31", **r})
    rankings = records(saved("rankings.jsonl"))
    cases = []
    for qid in lost:
        sampled = {p: [raw[qid][p][i] for i in range(1,5)] for p in ("sft", "dpo")}
        solo = {p: pool_summary(rr) for p, rr in sampled.items()}
        verify_metrics(solo["sft"], a[qid]); verify_metrics(solo["dpo"], b[qid])
        ensembles = {}
        for policy, e in (("sft", 0), ("dpo", epoch)):
            condition = f"epoch{e}_sampling4_max5_reranker"
            rr = sorted([r for r in rankings if r["condition"] == condition and r["question_id"] == qid], key=lambda r:r["rank"])
            if [r["rank"] for r in rr] != list(range(1,len(rr)+1)) or len({key(r["answer"]) for r in rr}) != len(rr):
                raise ValueError("Incomplete/duplicate ensemble ordering")
            if any(bool(r["label"]) != labels[(qid,key(r["answer"]))] or not math.isfinite(r["score"]) for r in rr):
                raise ValueError("Ensemble labels or scores differ")
            calculated = pool_summary([{"answers": [{**r, "official_accepted": bool(r["label"])} for r in rr]}])
            verify_metrics(calculated, by_question(scores[condition], ids)[qid])
            ensembles[policy] = rr
        cases.append({**examples[qid], "gold_aliases": aliases[qid], "epoch": epoch,
            "llama_sampling4": solo, "paired_draws": compare_draws(sampled["sft"], sampled["dpo"]),
            "greedy": {p: raw[qid][p][0] for p in ("sft", "dpo")}, "ensemble_sampling4": ensembles,
            "semantic_interpretation": "Unresolved: official nonacceptance alone does not imply an incorrect initial answer or expansion."})
    for path, expected in source_hashes.items(): checked(Path(path), expected)
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("cases.jsonl", cases), ("source_responses.jsonl", source_responses)):
        (output / filename).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    (output / "report.md").write_text(report(cases, epoch, state["selected_epoch"]), encoding="utf-8")
    write(output / "summary.json", {"question_ids": lost, "epoch": epoch, "selected_epoch_unchanged": state["selected_epoch"],
        "training_ready": False, "new_generation": False, "new_judging": False,
        "selection": "All validation144 Llama full-pool coverage losses versus epoch zero, four sampled responses"})
    write(output / "manifest.json", {"status": "complete", "comparison": str(comparison), "source_sha256": source_hashes,
        "export_code_sha256": digest(Path(__file__)), "output_sha256": {p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(output.iterdir()): handle.add(path, arcname=path.name)
    print((output / "report.md").read_text(encoding="utf-8"))
    print(f"Exported {len(cases)} question(s) to {output}")
    if archive: print(f"Download: {archive}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--epoch", type=int, default=1)
    a = p.parse_args()
    export(a.comparison.resolve(), a.output.resolve(), a.archive.resolve() if a.archive else None, a.epoch)
