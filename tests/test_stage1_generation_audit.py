"""CPU checks: python -m unittest discover -s tests -p 'test_stage1_generation_audit.py'."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cse_dpo import audit_stage1_training_generations as audit


def question(qid="q1"):
    return dict(question_id=qid, question="When?", gold_aliases=["first trimester of pregnancy", "early pregnancy"],
                supported_gold_aliases=["first trimester of pregnancy"],
                snippets=[{"text": "during the first trimester of pregnancy"}], gold_evidence=[],
                prompt="Question", prompt_tokens=8, prompt_sha256="prompt-hash")


class AuditTests(unittest.TestCase):
    def test_weak_matching_preserves_punctuation_and_boundaries(self):
        self.assertTrue(audit.contains_span("An IL-6  receptor.", "il-6 receptor"))
        self.assertFalse(audit.contains_span("An IL-6 receptor.", "IL 6 receptor"))
        self.assertFalse(audit.contains_span("secreted", "RET"))
        snippets = audit.marked_snippets(["[BS]first trimester[ES] [BS]of pregnancy[ES]"])
        self.assertFalse(any(audit.contains_span(s["text"], "first trimester of pregnancy") for s in snippets))

    def test_all_aliases_and_boundary_candidates(self):
        q = question()
        self.assertEqual(audit.classify_prediction("[BE]early pregnancy[EE]", q)["category"], "correct")
        short = audit.classify_prediction("[BE]first trimester[EE]", q)
        self.assertEqual(short["category"], "too_short_candidate")
        self.assertTrue(short["needs_semantic_review"])
        self.assertEqual(audit.classify_prediction("[BE]during the first trimester of pregnancy[EE]", q)["category"], "too_long_candidate")
        self.assertEqual(audit.classify_prediction("[BE]first[EE][BE]early pregnancy[EE]", q)["category"], "format_violation")
        self.assertEqual(audit.classify_prediction("[BE]the first[EE]", q)["category"], "other_extractive_mismatch_review")

    def test_eligibility_excludes_eval_and_uses_all_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_dir = root / "run/stage_1_concept_learning"
            stage_dir.mkdir(parents=True)
            audit.write_json(stage_dir.parent / "config.json", {"global_eval_question_ids": ["heldout"]})
            dev = root / "dev.json"
            audit.write_json(dev, [{"id": "dev"}])
            ids = ["ok", "heldout", "dev", "missing", "cross_snippets"]
            rows = [{"id": qid, "type": "factoid", "input_1": "Question?", "input_2": "[BS]IL-6 receptor[ES]", "output": "wrong first alias"} for qid in ids]
            rows[-1]["input_2"] = "[BS]IL-6[ES] [BS]receptor[ES]"
            audit.write_json(root / "train.json", rows)
            audit.write_json(root / "gold.json", {"questions": [{"id": qid, "exact_answer": [["absent", "IL-6 receptor"]]} for qid in ids if qid != "missing"]})
            qs, excluded, meta = audit.prepare_questions(root, {"stage_dir": str(stage_dir), "config": {"dev_source_input": str(dev)}}, root / "train.json", root / "gold.json")
            self.assertEqual([q["question_id"] for q in qs], ["ok"])
            self.assertEqual(qs[0]["supported_gold_aliases"], ["IL-6 receptor"])
            self.assertEqual({r["reason"] for r in excluded}, {"curriculum_heldout", "official_dev_or_test", "missing_gold_question", "no_gold_alias_in_marked_snippet"})

    def test_journal_recovers_torn_tail_but_rejects_other_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "generations.jsonl"
            audit.append_result(p, {"question_id": "done", "status": "complete"})
            with p.open("ab") as f:
                f.write(b'{"question_id": "unfinished')
            self.assertEqual(list(audit.read_journal(p)), ["done"])
            self.assertTrue(p.read_bytes().endswith(b"\n"))
            with p.open("ab") as f:
                f.write(b'not json\n')
            with self.assertRaises(json.JSONDecodeError):
                audit.read_journal(p)

    def test_manual_review_survives_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            q = question()
            records = {"q1": {"status": "complete", **audit.classify_prediction("[BE]first trimester[EE]", q)}}
            audit.export_audit(out, [q], records)
            with (out / "manual_review.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            rows[0]["review_notes"] = "Check modifier in snippet"
            rows[0]["approved_for_training"] = "no"
            audit.write_csv(out / "manual_review.csv", rows)
            audit.export_audit(out, [q], records)
            with (out / "manual_review.csv").open(encoding="utf-8-sig") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["review_notes"], "Check modifier in snippet")
            self.assertEqual(row["approved_for_training"], "no")

    def test_resume_reuses_completed_and_retries_oom(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            qs = [question("q1"), question("q2"), {**question("long"), "prompt_tokens": 500}]
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024))
            answer = dict(raw_output="[BE]first trimester[EE]", generated_tokens=10, hit_generation_limit=False)
            with patch.object(audit, "generate_one", side_effect=[answer, torch.OutOfMemoryError("fake OOM")]) as generate:
                summary = audit.run_audit(model, None, qs, out, max_prompt_tokens=100, max_new_tokens=20)
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(summary["status_counts"], {"complete": 1, "oom": 1, "skipped_context_budget": 1})
            with patch.object(audit, "generate_one", return_value=answer) as generate:
                summary = audit.run_audit(model, None, qs, out, max_prompt_tokens=100, max_new_tokens=20)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(summary["status_counts"]["complete"], 2)

    def test_resume_refuses_changed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base").mkdir()
            audit.write_json(root / "base/config.json", {})
            stage = {"base_model": str(root / "base"), "adapter": "adapter", "checkpoint_sha256": {"weights": "sha"}}
            audit.initialize_audit(root / "output", stage, [question()], {"max_new_tokens": 64}, {})
            audit.initialize_audit(root / "output", stage, [question()], {"max_new_tokens": 64}, {})
            with self.assertRaises(ValueError):
                audit.initialize_audit(root / "output", stage, [question()], {"max_new_tokens": 128}, {})

    def test_scorer_preserves_all_aliases_and_omits_unfinished_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            q = {**question(), "resources": [], "source_path": "train.json",
                 "raw_question": {"id": "q1", "type": "factoid", "exact_answer": ["first trimester of pregnancy", "early pregnancy"]}}
            audit.append_result(out / "generations.jsonl", {"question_id": "q1", "status": "complete", "prediction": "[BE]early pregnancy[EE]"})
            with patch("src.utility.bioasq_official.evaluate_with_bioasq_java", return_value={}) as scorer:
                audit.score_completed_questions(out, out, [q, question("pending")])
                args = scorer.call_args.kwargs
                self.assertEqual(len(args["prediction_rows"]), 1)
                raw_gold = args["examples_by_key"][("q1", "factoid")].raw_question
                self.assertEqual(raw_gold["exact_answer"], [q["gold_aliases"]])
                self.assertEqual(q["raw_question"]["exact_answer"], q["gold_aliases"])


if __name__ == "__main__":
    unittest.main()
