#!/usr/bin/env python3
"""Add official candidate matches to a training-only biomedical review template."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def add_matches(rows, matches):
    for row in rows:
        for candidate in row["candidates"]:
            accepted = matches[(row["question_id"], candidate["answer"])]
            candidate["official_accepted"] = accepted
            candidate["class"] = "C3" if accepted else None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/training13b.json")
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # The pinned original corpus supplies aliases, never generation prompts.
    if hashlib.sha256(args.gold.read_bytes()).hexdigest() != "38f1a4a54e90d6ee8b668f37f8d3d0be2df36dd4c89b26fa1ccceceb3861eaad":
        raise ValueError("Original gold corpus differs from the SFT source")
    bundle = ROOT / "gadi_sft_8b_starter"
    sys.path.insert(0, str(bundle / "scripts"))
    from expansion_dpo_data import DEFAULT_CONFIG, split_rows
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    allowed = {s: {r["question_id"] for r in rows} for s, rows in split_rows(config, full=True).items()}
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r["question_id"] not in allowed.get(r["split"], set()) for r in rows):
        raise ValueError("Only fitting/internal validation questions may be annotated")
    gold = {q["id"]: q for q in json.loads(args.gold.read_text(encoding="utf-8"))["questions"]}
    examples = []
    for qid in sorted({r["question_id"] for r in rows}):
        question = gold[qid]
        aliases = question["exact_answer"]
        aliases = [a for item in aliases for a in (item if isinstance(item, list) else [item])]
        examples.append({"question_id": qid, "question": question["body"], "gold_aliases": aliases})
    candidates = [{"question_id": r["question_id"], "answer": c["answer"]} for r in rows for c in r["candidates"]]
    report = args.output.parent / (args.output.stem + "-official-audit")
    report.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    from src.notebook_workflows.coverage_comparison import official_candidate_matches
    matches = official_candidate_matches(examples, candidates, report, jar_path=args.jar) if candidates else {}
    add_matches(rows, matches)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Saved {args.output}. Review semantic correctness, support and equivalence; unmatched classes remain unresolved.")


if __name__ == "__main__":
    main()
