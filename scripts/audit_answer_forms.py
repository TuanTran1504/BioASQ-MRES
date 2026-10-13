#!/usr/bin/env python3
"""Audit official alias forms; optionally compare gold-blind ordering on saved validation pools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / "data/BioASQ_expansion_sft/seed3407_teacher_strict_v1"
FORMS = ("abbreviation", "full_name", "combined_form", "numerical_expression", "descriptive_phrase", "other")
ORDERS = {
    "full_name_first": ("full_name", "combined_form", "abbreviation", "numerical_expression", "descriptive_phrase", "other"),
    "abbreviation_first": ("abbreviation", "combined_form", "full_name", "numerical_expression", "descriptive_phrase", "other"),
    "combined_first": ("combined_form", "full_name", "abbreviation", "numerical_expression", "descriptive_phrase", "other"),
}
VERSION = "answer-form-heuristics-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def abbreviation(text):
    """Surface abbreviation/symbol cue; does not infer biomedical equivalence."""
    text = text.strip()
    if len(text) > 24 or len(text.split()) > 3 or not re.fullmatch(r"[\wα-ωΑ-Ω+./−–-]+(?: [IVX0-9]+)?", text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return (sum(c.isupper() for c in letters) / len(letters) >= .6
            or bool(re.fullmatch(r"(?:miRNA|lncRNA|mRNA|siRNA|tRNA|rRNA|piRNA)s?", text)))


def classify(text):
    """Mutually exclusive primary form plus overlapping flags and review reasons."""
    s = re.sub(r"\s+", " ", text).strip()
    flags, review = [], []
    numeric = bool(re.fullmatch(
        r"(?:[~≈<>≤≥]|about |approximately )?\d[\d.,:%/\s–−-]*(?:to \d[\d.,]*)?"
        r"(?:\s*(?:%|percent|years?|months?|weeks?|days?|hours?|minutes?|seconds?|nm|mm|cm|m|"
        r"mg|g|kg|ug|μg|µg|ml|mL|L|kb|kDa|bp|base pairs?))?", s, re.I))
    if s.lower() in {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "fifteen"}:
        numeric = True
    numeric = numeric or bool(re.fullmatch(r"O\(.+\)", s))
    if numeric:
        flags.append("numerical_expression")
    abbr = abbreviation(s)
    if abbr:
        flags.append("abbreviation")
    combined = False
    for match in re.finditer(r"\(([^()]*)\)", s):
        inside, outside = match.group(1), (s[:match.start()] + s[match.end():]).strip()
        if outside and ((abbreviation(inside) and len(outside.split()) >= 2)
                        or (abbreviation(outside) and len(inside.split()) >= 2)):
            combined = True
    if combined:
        flags.append("combined_form")
    descriptive = bool(re.search(r"\b(?:is|are|was|were|contains?|causes?|binds?|encodes?|involves?|"
                                r"results?|occurs?|occur|required|responsible|due to|used for)\b", s, re.I))
    descriptive = descriptive or bool(re.match(
        r"(?:mutations? (?:in|of)|absence of|loss of|lack of|inhibition of|"
        r"(?:assigning|regulating|increasing|decreasing|induction of|activation of)\b)", s, re.I))
    descriptive = descriptive or len(s.split()) >= 14
    if abbr:
        descriptive = False  # Uppercase IS/ARE can be symbols, not sentence verbs.
    if descriptive:
        flags.append("descriptive_phrase")
    if re.search(r"\bOR\b|\bor/and\b|;", s):
        review.append("Multiple alternatives or clauses in one recorded alias")
    if len(s.split()) >= 14:
        review.append("Long expression: description versus formal name needs review")
    if not s:
        category = "other"
        review.append("Empty alias")
    elif numeric:
        category = "numerical_expression"
    elif descriptive:
        category = "descriptive_phrase"
    elif combined:
        category = "combined_form"
    elif abbr:
        category = "abbreviation"
        if len(s) <= 2:
            review.append("Short symbol may not be an abbreviation")
    elif re.search(r"[A-Za-z]", s):
        category = "full_name"
        review.append("Plain name/expression; surface rules cannot prove it is an expanded full name")
    else:
        category = "other"
        review.append("Notation not recognised")
    return {"primary_form": category, "form_flags": flags or [category],
            "needs_review": bool(review), "review_reasons": review}


def order_answers(answers, policy):
    if policy == "original":
        return list(answers)
    priority = {form: i for i, form in enumerate(ORDERS[policy])}
    return sorted(answers, key=lambda answer: priority[classify(answer)["primary_form"]])


def load_panel(source):
    manifest = read(source / "manifest.json")
    for name in ("audit.jsonl", "train.jsonl", "validation.jsonl", "dev_question_ids.json"):
        if digest(source / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Source hash changed: {name}")
    audits = records(source / "audit.jsonl")
    indexed = {r["question_id"]: r for r in audits}
    if len(indexed) != len(audits):
        raise ValueError("Duplicate audit questions")
    result = []
    for split, expected in (("train", 1296), ("validation", 144)):
        rows = records(source / f"{split}.jsonl")
        if len(rows) != expected or len({r["question_id"] for r in rows}) != expected:
            raise ValueError("Wrong fitting/validation question count")
        for row in rows:
            a = indexed[row["question_id"]]
            if a["internal_split"] != split:
                raise ValueError("Audit split differs")
            user = json.loads(row["messages"][1]["content"])
            result.append({"question_id": row["question_id"], "split": split,
                "question": user["question"], "gold_aliases": a["gold_aliases"], "snippets": user["snippets"]})
    if len({r["question_id"] for r in result}) != 1440 or set(indexed) != {r["question_id"] for r in result}:
        raise ValueError("Split membership differs")
    if {r["question_id"] for r in result} & set(read(source / "dev_question_ids.json")):
        raise ValueError("Development overlap")
    return result, manifest


def alias_audit(panel):
    aliases, questions, summaries = [], [], {}
    for row in panel:
        forms = []
        for index, alias in enumerate(row["gold_aliases"]):
            item = {"question_id": row["question_id"], "split": row["split"], "question": row["question"],
                    "alias_index": index, "accepted_alias": alias, **classify(alias)}
            aliases.append(item)
            forms.append(item["primary_form"])
        questions.append({k: row[k] for k in ("question_id", "split", "question", "gold_aliases")} |
            {"alias_forms": forms, "available_forms": sorted(set(forms)),
             "multiple_aliases": len(forms) > 1, "multiple_forms": len(set(forms)) > 1})
    for split in ("train", "validation"):
        aa = [a for a in aliases if a["split"] == split]
        qq = [q for q in questions if q["split"] == split]
        counts = Counter(a["primary_form"] for a in aa)
        summaries[split] = {"question_count": len(qq), "accepted_alias_count": len(aa),
            "alias_form_counts": {f: counts[f] for f in FORMS},
            "questions_with_each_form": {f: sum(f in q["available_forms"] for q in qq) for f in FORMS},
            "questions_with_multiple_aliases": sum(q["multiple_aliases"] for q in qq),
            "questions_with_multiple_forms": sum(q["multiple_forms"] for q in qq),
            "first_recorded_alias_forms": dict(Counter(q["alias_forms"][0] for q in qq)),
            "aliases_flagged_for_review": sum(a["needs_review"] for a in aa)}
    return aliases, questions, summaries


def compare_validation(run, panel, output):
    """Ranking reads only candidate surface forms; gold is used after ordering."""
    from scripts.analyze_matched_qwen_expansion import paired_contrast
    from src.notebook_workflows.coverage_comparison import official_candidate_matches
    from run_single_answer_sampling import expansion_candidates
    from run_extractive_expansion_8b import parse_equivalent_response
    config, status = read(run / "config.json"), read(run / "status.json")
    if status["status"] != "complete" or config["max_candidates"] != 5 or config["num_generations"] != 10:
        raise ValueError("Expected completed max-five ten-sample validation inference")
    examples, generations = records(run / "examples.jsonl"), records(run / "generations.jsonl")
    expected = {r["question_id"]: r for r in panel if r["split"] == "validation"}
    if len(examples) != 144 or len(generations) != 144:
        raise ValueError("Keep all 144 validation questions, including failed outputs")
    if any({r['question_id'] for r in rr} != set(expected) for rr in (examples, generations)):
        raise ValueError("Validation IDs differ from audited split")
    def evidence(items):
        return [r['text'] for r in items]
    for e in examples:
        audited = expected[e['question_id']]
        if (e['question'] != audited['question'] or e['gold_aliases'] != audited['gold_aliases']
                or evidence(e['snippets']) != evidence(audited['snippets'])):
            raise ValueError("Validation question, gold or evidence differs")
    candidates, pools, first_draws = [], {}, {}
    for row in generations:
        if len(row['samples']) != 10 or row['snippets_truncated'] or row['included_snippets'] != row['total_snippets']:
            raise ValueError("Incomplete draws/evidence")
        for draw, s in enumerate(row['samples'], 1):
            if s['draw'] != draw:
                raise ValueError("Sample order differs")
            try:
                parsed, _, _, _ = parse_equivalent_response(s['raw_response'], 5)
            except ValueError:
                parsed = []
            if parsed != s['answers']:
                raise ValueError("Saved candidates differ from raw parsing")
        unique = expansion_candidates(row['samples'])
        qid = row['question_id']
        pools[qid] = [c['answer'] for c in unique]
        first_draws[qid] = list(dict.fromkeys(c['answer'] for c in row['samples'][0]['answers']))
        candidates.extend({'question_id': qid, **c} for c in unique)
    if candidates != records(run / 'candidates.jsonl'):
        raise ValueError("Archived pooled candidates differ")
    matches = official_candidate_matches(examples, candidates, output,
        jar_path=ROOT/'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar') if candidates else {}
    result, all_scored = {}, {}
    for mode, values in (('first_stochastic_draw', first_draws), ('ten_sample_pool', pools)):
        mode_result, scored = {}, {}
        for policy in ('original', *ORDERS):
            scored[policy] = []
            for qid, answers in values.items():
                ordered = order_answers(answers, policy)
                rank = next((i for i, answer in enumerate(ordered, 1) if matches[(qid, answer)]), None)
                scored[policy].append({'question_id': qid, 'mrr_at5': 1/rank if rank and rank <= 5 else 0.,
                    'coverage_at1': float(rank == 1), 'coverage_at5': float(bool(rank and rank <= 5)),
                    'coverage_full_pool': float(rank is not None)})
            metrics = {k: sum(r[k] for r in scored[policy])/144 for k in scored[policy][0] if k != 'question_id'}
            mode_result[policy] = {'metrics': metrics}
            if policy != 'original':
                pairs = paired_contrast(scored['original'], scored[policy],
                    metrics=('mrr_at5', 'coverage_at1', 'coverage_at5', 'coverage_full_pool'))
                mode_result[policy]['minus_original'] = {
                    metric: {'difference': value['sft_minus_base'],
                             **{k: v for k, v in value.items() if k != 'sft_minus_base'}}
                    for metric, value in pairs.items()}
        result[mode] = mode_result
        all_scored[mode] = scored
    write(output / 'ordering_per_question.json', all_scored)
    return {'scope': 'Internal-validation diagnostic, not held-out test; first draw is stochastic, not greedy',
            'source_files_sha256': {f: digest(run/f) for f in ('config.json','examples.jsonl','generations.jsonl','candidates.jsonl','status.json')},
            'question_count': 144, 'rules_use_gold': False, 'candidate_pool_unchanged': True,
            'results': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path, default=ROOT/'Artifacts/answer_form_audit_v1')
    p.add_argument('--validation-run', action='append', default=[], help='MODEL=path/to/epoch-N/sampling10')
    args = p.parse_args()
    panel, _ = load_panel(args.source)
    aliases, questions, summary = alias_audit(panel)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rr in (('accepted_alias_forms.jsonl', aliases), ('question_forms.jsonl', questions),
                     ('review_queue.jsonl', [a for a in aliases if a['needs_review']])):
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr), encoding='utf-8')
    summary = {'classifier': VERSION, 'classification': 'Surface heuristics, not human/LLM-validated semantic annotation; full_name includes plain names/expressions',
        'splits': summary, 'source_manifest_sha256': digest(args.source/'manifest.json'),
        'development_questions_used': 0, 'ordering_comparisons': {}}
    sys.path.insert(0, str(ROOT/'gadi_sft_8b_starter/scripts'))
    for supplied in args.validation_run:
        model, path = supplied.split('=', 1)
        if model not in ('llama31','qwen3','ministral3') or model in summary['ordering_comparisons']:
            raise ValueError('Expected distinct supported model names')
        destination = args.output/model
        destination.mkdir(exist_ok=True)
        result = compare_validation(Path(path), panel, destination)
        write(destination/'ordering_summary.json', result)
        summary['ordering_comparisons'][model] = result
    write(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
