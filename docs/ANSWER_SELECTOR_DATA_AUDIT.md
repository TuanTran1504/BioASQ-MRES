# Max-five answer-selector data audit

This stage prepares a supervised answer-selector dataset and audits potential
ranking supervision. It runs official Java scoring on a CPU. It does not train
a selector or DPO model, generate answers, call a paid judge or evaluate test95.

## Run on Gadi

From `gadi_sft_8b_starter`:

```bash
git pull --ff-only
bash scripts/submit_answer_selector_audit.sh
```

The default source is the completed native SFT reranker archive
`outputs/sft_ensemble_reranker/train-181350943.gadi-pbs`. An alternative archive
of that same schema can be passed as the first argument. Its recorded preparation
identifies all three complete native SFT response banks. The worker validates
bank hashes, prompts, seeds, split membership and complete request counts.
Held-out exclusion uses IDs from the existing input-only test95 bundle, or an
existing SFT-test/ensemble submission's `test_inputs.jsonl`; no test gold is opened.

This first audit explicitly uses the existing **one greedy plus four stochastic
responses per model/question**. It does not wait for or silently substitute the
ongoing ten-sample adapted-reranker banks. Those banks include different generator
policies; a future ten-sample selector experiment needs a separately validated
all-SFT source. Greedy and sampling4 pools remain separate.

One CPU job writes:

```text
outputs/answer_selector_data/audit-JOBID.gadi-pbs/
```

It also packages that directory, with files at the archive root, as:

```text
/scratch/nl78/$USER/max5-selector-dataset-JOBNUMBER.tar.gz
```

Monitor the latest submission:

```bash
record=$(ls -t outputs/answer_selector_data/submission-*/jobs.tsv 2>/dev/null | head -n 1)
if [[ -n "$record" ]]; then
    cat "$record"
    job=$(awk -F '\t' 'NR==2 {print $2}' "$record")
    qstat -swx "$job"
    qstat -fx "$job" | grep -E 'Job Id|job_state|Exit_status|comment'
else
    echo "No selector data audit submission found."
fi
```

After completion:

```bash
number=${job%%.*}
cat "outputs/answer_selector_data/audit-${job}/report.md"
tail -n 60 "max5-selector-audit.o${number}"
tail -n 60 "max5-selector-audit.e${number}"
```

Download the named archive in PowerShell, replacing `JOBNUMBER`:

```powershell
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/max5-selector-dataset-JOBNUMBER.tar.gz .\Artifacts\
if ($LASTEXITCODE -eq 0) {
    New-Item -ItemType Directory -Force .\Artifacts\answer_selector_data_v1 | Out-Null
    tar -xzf .\Artifacts\max5-selector-dataset-JOBNUMBER.tar.gz -C .\Artifacts\answer_selector_data_v1
}
```

## Output contract

- `summary.json` / `report.md`: counts by fitting/validation and greedy/sampling4;
  accepted/unaccepted candidates; covered, uncovered and empty questions; potential
  preference questions; pool sizes; input character lengths; parser warnings.
- `train_inputs.jsonl` / `validation_inputs.jsonl`: every question/pool, even if
  empty or uncovered. `messages` contains only instructions, question, all supplied
  snippets, and numbered candidate surfaces. It contains no gold aliases, acceptance
  labels, semantic judgments or model-generated relation tags.
- `train_supervision.jsonl` / `validation_supervision.jsonl`: all recorded gold
  aliases and generated candidates, official acceptance labels, native draw/position
  provenance, original-style source ranks/model votes/draw frequencies, accepted IDs,
  target IDs and eligibility. These metadata allow recomputing a logistic baseline
  on the same prospective pools. This is a sidecar, not input
  to the selector. Do not concatenate it into prompts.
- `train_sft.jsonl` / `validation_sft.jsonl`: chat messages with a target JSON array
  of `min(5, pool size)` distinct candidate IDs for pools with an accepted answer.
  Accepted IDs precede unaccepted IDs; order within each class is randomized with a
  reproducible seed. This is one valid weakly ordered target, not a fully graded gold
  ranking. A future trainer should support tied positives/target order augmentation.
  A row has weight 0.5 so each mode has equal weight; incomplete coverage means some
  questions have only one eligible mode. Two views are not independent questions.
- `review_candidates.jsonl`: up to three closest lexical accepted/unaccepted
  comparisons per question/pool, with question, snippets and aliases for review.
  `review_sample.jsonl` selects up to 100 per split. Every semantic category is
  initially `unresolved`; supported synonyms must not be labeled medically wrong
  just because they fail official matching. Numbers and punctuation are review
  hints, not an equivalence classifier. Review does not alter official labels.
- `response_audit.jsonl`: recovery issues, empty responses, candidate cap exclusions
  and 512-token ceiling warnings. Requested failed draws remain accounted for.
- `official/`: scoring inputs/results from the existing official Java matcher;
  every recorded alias is supplied for each question.
- `manifest.json`: complete/failed state and output hashes. `source_provenance.json`
  records source hashes. `heldout_ids.json` contains exclusion membership only.

All supplied snippets are preserved without selecting evidence using gold.
Candidate positions and IDs are randomized independently of acceptance labels.
All generated candidates beyond the logistic top five remain available.
Gold aliases that were not generated are never inserted into pools.

Exact-string pooling preserves original spacing, Unicode, capitalization and
punctuation. Near-duplicate strings are not silently collapsed. This deliberately
differs from the historical normalized pooling policy: historical scores must not
be claimed as results on these prospective pools. Recompute logistic/other baselines
on the same pools before a selector comparison.

No DPO pairs are created. Potential accepted/unaccepted comparison counts only
describe available supervision, not actual sampled preferred/rejected rankings.
Questions with no accepted candidate cannot supply an MRR-positive target, but
remain in evaluation inputs and coverage denominators.

## Before model training

1. Read the counts and review sample. Distinguish equivalent wording, substantive
   scope/number/entity distinctions, and unresolved cases; automatic flags are not
   completed semantic annotation.
2. Choose a selector backbone/tokenizer and measure actual context lengths. The
   export is model-independent and untokenized; it applies no truncation. If it does
   not fit, define a gold-blind policy on fitting/validation and audit coverage loss.
3. Train/select using fitting1296/validation144 only, preserving question-disjoint
   membership. The existing SFT generators saw fitting questions; this is ordinary
   stacked-selector training, not out-of-fold generator data.
4. Compare supervised selector versus logistic on identical pools before adding DPO.
   DPO should then use the selector's sampled ranking errors, exclude ties and count
   distinct questions, rather than manufacture many near-identical permutations.
5. Keep dev160/test95 out of training. test95 has been inspected repeatedly and is
   exploratory, not a fresh confirmatory evaluation.
