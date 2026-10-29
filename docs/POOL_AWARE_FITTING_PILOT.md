# Prompt-only pool-aware generation: fitting pilot

This pilot asks whether showing previous proposals to a frozen max-five SFT
generator recovers accepted answers missed by independent sampling at the same
**response/output-allowance budget**. No model training or paid judging runs.

## Conditions and budget

Use the exact original Llama, Qwen and Ministral max-five SFT checkpoints and
the saved four-sample fitting banks. For every question, reuse sampled draws
1–3 from each model as the **same nine-response prefix** in every condition.
Greedy outputs are not part of this prefix. Replace each model's fourth draw:

| Condition | Fourth response from each model | Final responses/question |
|---|---|---:|
| Independent | Reuse the original independent fourth response | 12 |
| Correction without pool | Fresh correction prompt with an empty previous-proposals list | 12 |
| Pool-aware correction | Same correction prompt, showing the full prefix proposal pool | 12 |

All three pool-aware requests see the **same frozen nine-response pool**; they do
not see one another's new responses. This permits parallel model jobs and avoids
giving one model extra intermediate information. Each replacement uses its native
fourth-slot seed, temperature 0.8, top-p 0.95, output limit 512 tokens and at most
five retained answer candidates. No sampling until success is permitted.

The no-pool correction control separates new prompt instructions from the
additional previous-answer information. The correction prompt allows revising an
incorrect initial concept, while still requiring equivalent forms within one
response. Repeating a supported existing answer is allowed; novelty and extra
candidates alone do not earn a coverage gain. Previous proposals are explicitly
unverified data, not evidence or instructions.

The model-visible pool is exact-string deduplicated, retains all parser-accepted
surfaces in raw positions 1–5, and uses a fixed question-seeded order independent
of labels. Case/spacing variants stay separate. There is no top-five prefilter,
gold-based routing, evidence truncation or oracle selection of questions to repair.
Every selected question receives both correction conditions.

**Input tokens are not matched:** a visible pool lengthens the prompt. The report
includes input/output counts and parsing/ceiling counters. Equal response/output
allowances do not imply equal FLOPs, wall time or total experiment costs. The
baseline is archived generation; inference-library/hardware drift can affect
exact replay even with identical seeds and checkpoint lineage. The fresh no-pool
control provides a same-runtime comparison for the effect of pool information.

## Fitting-only panel and prospective DPO use

Select **192 fitting questions by question-ID hash**, without using successes,
failures, gold aliases or candidate labels. Validation144, dev160 and test95 are
excluded. The prior 40-question fitting-only preference holdout is also excluded.
Reserve 160 selected questions for discovery and 32 for this pilot's audit.
The new 32 audit questions are excluded from future preference training.

All of these questions were seen by the original SFTs: these are **feasibility
diagnostics**, not held-out generalization results. The audit split is fitting-only
and small. Do not interpret exploratory confidence intervals as independent test
evidence after selecting an approach from these results.

The primary comparison is full-pool coverage **against all twelve original
independent responses**, not only against the nine-response prefix. A repair that
duplicates a success the independent fourth draws already supplied is not an
additional covered question. Gains and losses are both counted, with their IDs.

CPU scoring uses the complete hash-pinned official Java runtime and the frozen
original max-five logistic reranker. Top-five coverage and MRR are secondary.
Verbatim pools are rebuilt consistently across all conditions; previous normalized
pool scores cannot replace the new baseline. Focal comparisons also replace one
model's fourth response while keeping the other eleven original responses fixed.

Gold aliases appear only in separate CPU-scoring/review files. Model inputs are
whitelisted question, complete snippets and unlabelled proposal IDs/surfaces.
Generation files contain no gold labels, accepted-answer flags or review notes.

Potential recovered cases are saved for evidence/scope review, **not as trainable
DPO pairs**. An independent failure and a pool-aware success arose under different
prompts and are not automatically paired. The pilot reports whether the three
pool-aware replies contain eligible positive and negative completions under the
same pool-conditioned message input. Such cross-model completions still require
review and target-model reference scoring before any later DPO stage. If matching
negatives are absent, further sampling would be a separately recorded data step.
Official nonacceptance does not establish medical incorrectness. `training_ready`
remains false throughout this pilot.

## Run on Gadi

The existing reviewed package supplies immutable source provenance; no new archive
upload is needed. Preparation verifies all native banks/checkpoints, complete
target-tokenizer plans, excluded membership, the complete evaluator distribution,
and a real Java matching smoke before the first qsub.

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_pool_aware_fitting_pilot.sh \
  outputs/generator_form_designs/reviewed-20261029-v2
```

This creates three GPU smoke/full chains and one dependent CPU score job: seven
jobs. The full jobs generate **1,152 fresh responses total** (192 questions x
three models x two correction conditions). The independent arm's 2,304 responses
are reused. Each model's smoke additionally generates four responses on its two
longest selected questions; those smoke responses are excluded from scoring.
Full correction jobs load one model and execute both prompt conditions.

Each job's raw requests, seeds, prompt-token hashes, outputs and progress are saved.
Empty, recovered, capped and ceiling outputs remain in their logical denominators.
An incomplete generation job blocks scoring rather than dropping questions or
silently generating replacements. There is an explicit worker resume option for
technical interruptions; completed attempts are retained, including parse failures.
Inspect the existing submission before retrying, since repeating the launcher
creates a separate experiment.

Monitor the newest pilot from the starter directory:

```bash
module load python3/3.12.13
python3 - <<'PY'
import csv, json, subprocess
from pathlib import Path
records = sorted(Path('outputs/pool_aware_fitting_pilot').glob('submission-*/jobs.tsv'),
                 key=lambda p: p.stat().st_mtime)
if not records:
    raise SystemExit('No submission record yet; preparation may still be running.')
print('Record:', records[-1], flush=True)
pending = []
with records[-1].open() as f:
    for r in csv.DictReader(f, delimiter='\t'):
        out = Path(r['output'])
        manifest = out / 'manifest.json'
        d = json.loads(manifest.read_text()) if manifest.exists() else {}
        print(r['model'], r['stage'], r['job_id'], d.get('status', 'no saved status'), flush=True)
        if 'completed_requests' in d:
            print('  Requests:', d['completed_requests'], '/', d['expected_requests'], flush=True)
        if d.get('error'): print('  Error:', d['error'], flush=True)
        if (out / 'report.md').exists(): print((out / 'report.md').read_text(), flush=True)
        if d.get('status') != 'complete': pending.append(r['job_id'])
if pending: subprocess.run(['qstat', '-swx', *pending])
PY
```

Results are under `outputs/pool_aware_fitting_pilot/submission-.../comparison-<job>/`:

- `report.md`: three-condition comparison and full-pool gains/losses.
- `comparison_summary.json`: primary/secondary metrics, paired comparisons, split
  metrics/transitions, focal effects, costs and diagnostic counters.
- `comparison_per_question.json` / `rankings.jsonl`: all question outcomes and
  exact ordered candidate surfaces for inspecting changed cases.
- `recovery_review_cases.jsonl`: pool-aware versus independent coverage changes,
  evidence, original/new surfaces and same-prompt supervision opportunities.
  Audit cases are labelled ineligible for future training; no DPO files are made.

The methodological ingredients are inspired by
[Asking a Language Model for Diverse Responses](https://aclanthology.org/2025.uncertainlp-main.8/)
and [system-level DPO](https://papers.nips.cc/paper_files/paper/2025/hash/a5299935ef5fff4e33630ebc70babd30-Abstract-Conference.html).
This is our biomedical adaptation and an untested hypothesis, not a replication
or an established improvement from either paper.
