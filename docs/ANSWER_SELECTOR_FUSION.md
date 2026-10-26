# Fusion of saved answer-selector rankings

This CPU-only audit reuses the selected Llama, Qwen and Ministral validation
predictions and the logistic rankings from the same pilot submissions. It does
not load model weights, train a model, generate answers, call a paid API, rerun the
Java scorer, construct DPO pairs or evaluate test95. Official labels come from the
existing frozen, audited selector data.

## Submit on Gadi

From `gadi_sft_8b_starter`:

```bash
git pull --ff-only
bash scripts/submit_answer_selector_fusion.sh \
  outputs/answer_selector_pilot/comparison-20261026-085714/comparison_summary.json
```

The comparison path above is also the wrapper's default. All three source pilots
must have completed training and reports. The wrapper snapshots the source
comparison, code and CPU job, then submits **one normal-queue job**, requesting one
CPU, 16GB RAM and a two-hour walltime limit. The worker validates all frozen
sources before scoring. Preparation failure submits no job, and a submission
failure preserves the partial record.

The existing comparison pins its three submissions and chosen epochs. This audit
uses those exact sources, rather than scanning for newer runs or choosing epochs
again. It verifies source hashes, common data/configuration, selected checkpoints,
all144 validation questions in both pool modes, exclusion of dev160/test95,
display-ID mappings, exact candidate surfaces and the saved per-question metrics.
Each independently fitted logistic baseline must reproduce the same top-five
ranking. Inconsistent, changed, failed or missing sources stop the audit.

## Fixed fusion conditions

Each system contributes exactly its first `min(5, pool size)` source candidate
IDs. The logistic baseline has a full ranking available, but **only its top five
enter fusion**, matching the LLM selectors' output depth. There are no learned
weights, smoothing constants, score calibration or validation weight searches.

| Condition | Inputs | Primary ranking rule | Secondary tie rule |
|---|---|---|---|
| selectors3_model_vote | Three selectors | Count systems selecting the candidate | Sum of reciprocal ranks |
| selectors3_rank_vote | Three selectors | Sum `1/rank` across systems selecting it | Count systems selecting it |
| selectors3_logistic_model_vote | Three selectors and logistic | Count systems selecting the candidate | Sum of reciprocal ranks |
| selectors3_logistic_rank_vote | Three selectors and logistic | Sum `1/rank` | Count systems selecting it |

Remaining exact score ties use a fixed SHA256 key of seed, question ID, pool and
answer surface, never labels, accepted aliases or expert order. Reciprocal scores
use exact rational arithmetic internally. This is the project's unsmoothed rank
voting rule, not the conventional smoothed RRF formula `1/(k+rank)`.

The three model tokenizers use different display codes. Stored
`selected_source_ids` map their choices back to common candidate identities, and
the worker verifies those mappings against each frozen input. Fusion never votes
directly over display tokens. Case, punctuation, Unicode and whitespace variants
remain separate candidates. Identical source candidates get one vote per expert.
No semantic synonym merging or rewriting occurs.

Only the union of the supplied top-five lists is eligible for the final top five.
Candidates absent from all input lists receive no contribution and are not
appended as an artificial tail. Small and empty original pools remain in the
denominator. Every condition retains all144 validation questions per mode.

## Results and interpretation

The result directory is printed by the submission wrapper:

`outputs/answer_selector_fusion/submission-TIMESTAMP-PID/fusion-JOBID/`

- `report.md`: eight conditions per mode (four individual systems and four
  fusions), MRR@5, top-five hit counts, gains/losses against logistic, complementary
  coverage and descriptive paired MRR intervals.
- `comparison_summary.json`: metrics, means, fixed fusion rules, source pins,
  10,000-resample paired contrasts and coverage transitions against logistic and
  the best individual selector by the previously used validation-mean criterion.
- `comparison_per_question.json`: each condition's scores for every question.
- `question_audit.jsonl`: all288 validation question/pool views, supplied context,
  accepted aliases, original candidates, each system's top five, fusion support,
  per-question scores and changes against the two baselines.
- `changed_cases.jsonl`: the subset whose MRR changes for any fusion/baseline
  comparison, retaining both improvements and regressions.
- `manifest.json`: completion/failure and output hashes.

Complementarity diagnostics count questions with accepted answers selected by
zero, one, two or all three selectors, each selector's unique top-five hits,
selector-union hits missed by logistic and logistic hits missed by all selectors.
All such counts include question IDs in the JSON files.

Three coverage concepts remain distinct:

1. **Original full-pool coverage:** an accepted candidate exists in the generator
   pool. This remains fixed across conditions.
2. **Input-union coverage:** an accepted candidate exists somewhere in the lists
   available to a particular fusion. This is its oracle selection bound, not a
   deployed system score or a guaranteed improvement.
3. **Output coverage@5:** the fused five actually contain an accepted candidate.

Higher union coverage does not ensure better fusion MRR. Voting can suppress a
rare accepted candidate, and the fourth expert can improve or hurt ranking.
Official nonacceptance does not establish biomedical incorrectness. No semantic
verdict is inferred from mechanical gains or losses.

These comparisons reuse the same validation set that selected checkpoints and
backbones. The paired intervals are descriptive, not independent test evidence.
The two pool views are not independent questions; contrasts are computed
separately by mode, and the mean across modes is a descriptive summary only.

All exported cases have `split=validation` and `DPO_training_eligible=false`.
An interesting validation rescue motivates a later fitting-only rollout audit;
it must never be copied into a DPO fitting dataset. This job produces no DPO pairs
and establishes neither preference availability nor held-out improvement.

## Monitor

```bash
record=$(ls -t outputs/answer_selector_fusion/submission-*/jobs.tsv 2>/dev/null | head -n 1)
cat "$record"
job=$(awk -F '\t' '$1=="fusion" {print $2}' "$record")
qstat -swx "$job"
```

After completion, or if PBS history is unavailable:

```bash
submission=$(dirname "$record")
cat "$submission/fusion-${job}/manifest.json"
cat "$submission/fusion-${job}/report.md"
```

For an explicitly local CPU run, the same worker can be invoked without PBS:

```bash
python3 ../scripts/fuse_answer_selector_pilots.py --mode run \
  --comparison outputs/answer_selector_pilot/comparison-20261026-085714/comparison_summary.json \
  --output "outputs/answer_selector_fusion/manual-$(date -u +%Y%m%d-%H%M%S)"
```

The CPU job is the recommended route on Gadi. No remote job is submitted by the
assistant; the user runs the wrapper.
