# Contextual max-five SFT reranker pilot

This pilot tests whether a biomedical transformer can improve selection from the
**same existing max-five SFT ensemble candidate pools**. It does not change the
generators, generate additional responses, modify aliases, perform DPO or use paid
judging. Gadi submission is performed by the user.

## Comparisons

| Condition | How it ranks |
|---|---|
| Logistic | Existing frozen max-five logistic reranker |
| Contextual | Fine-tuned BiomedBERT reading question, candidate and selected evidence |
| Hybrid | Validation-selected reciprocal-rank fusion of the two rankings |

The hybrid uses the complete logistic ranking: TF-IDF text, character features,
source frequencies, provenance, relation labels and other existing metadata. It
does not isolate only frequency/provenance, and it does not fit a second classifier
on in-sample neural predictions.

The source baseline is
`outputs/sft_ensemble_reranker/train-181350943.gadi-pbs`. Its previously validated
fitting/validation candidate labels and input contexts are reused. The saved test
comparison is `outputs/sft_ensemble_test/comparison-181287698.gadi-pbs`.

## Data and selection

| Partition | Questions | Use |
|---|---:|---|
| Fitting | 1,296 | Update encoder/classification weights |
| Internal validation | 144 | Select epoch and hybrid weight |
| Dev160 | 160 | Excluded |
| Historical official test | 95 | Score frozen choices on cached pools |

This controlled pilot deliberately uses the **completed four-sample fitting and
validation banks used by the existing logistic baseline**. Greedy and sampling4
remain separate pools. Test uses the existing greedy and sampling10 pools, equally
for both rankers. The four-to-ten sample shift is a limitation shared by the two
rankers in this comparison.

This pilot is separate from the already prepared ten-sample strict/alternative
DPO-bank workflow. It does not replace, resubmit or modify those jobs. Comparing
two rankers trained from the same existing banks isolates this pilot from that
larger generation change.

Every original candidate is retained, including officially nonaccepted variants.
Questions with empty pools remain in evaluation. Labels mean **official scorer
acceptance**, not semantic correctness. No semantic labels or new hard-negative
sampling are introduced in this first comparison.

Official aliases and acceptance labels never enter the encoder's tokenized input.
The training worker loads fitting and validation data for learning/selection; test
labels are used only in the separate scoring stage. Preparation validates test
membership/source lineage, and integrity checks hash all prepared files.

Preparation checks fitting/validation/test IDs and excludes dev160. It also checks
normalized question-text overlap between fitting/validation and test95. The
historical test95 has already been examined in this project; its results remain
exploratory rather than a fresh confirmatory evaluation.

## Encoder and evidence policy

Backbone: **Microsoft BiomedBERT, abstracts + full text**, formerly PubMedBERT,
approximately 110M parameters. Revision:
`e1354b7a3a09615f6aba48dfad4b7a613eef7062`.
[Model card](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext).

The backbone starts from its biomedical masked-language-model checkpoint with a
new scalar classification head. All encoder parameters are fine-tuned. It is not
an off-the-shelf trained answer reranker or a copy of Prompt-BioEL/ENR.

The native input budget is **512 tokens**:

1. Segment A preserves the complete question and candidate answer.
2. Up to three snippets are selected using a fixed lexical overlap rule over the
   question and candidate. No gold aliases, acceptance labels, relation tags or
   classifier predictions are used for selection.
3. The remaining budget is divided among the selected snippets. Long snippets
   contribute the most lexically relevant token window; fixed original-order ties
   are used. Native BERT segment/special-token framing is preserved.
4. A question/candidate pair that leaves fewer than 32 evidence tokens fails
   preparation instead of silently truncating the question or answer.

The encoder therefore **does not read all evidence**. The generators' full evidence
and their candidate pools remain unchanged. The selected-evidence policy is part
of the new reranking pipeline; improvements cannot be attributed solely to the
transformer architecture independently of this policy.

`train_evidence_audit.jsonl`, `validation_evidence_audit.jsonl` and
`test_evidence_audit.jsonl` record selected snippet IDs, window offsets and retained
token counts for every candidate. `evidence_summary.json` reports omission counts.

## Training protocol

- Three epochs, learning rate 0.00002, batch size 8, gradient accumulation 4.
- Full fine-tuning with AdamW defaults, weight decay 0.01, linear schedule and
  10% warm-up; seed 3407; gradient clipping at 1.0.
- FP16 on Gadi V100 and gradient checkpointing.
- Weighted binary cross-entropy: equal initial weight per question/pool, followed
  by balancing the total weight of accepted and nonaccepted fitting candidates.
  All-negative question/pools are retained.
- Evaluate after each epoch. Select the highest mean validation MRR@5 across
  greedy and sampling4, equally weighted. Save the selected model and tokenizer.
- On that selected checkpoint's validation predictions, choose hybrid weight
  alpha from `[0, 0.25, 0.5, 0.75, 1]`:
  `score = (1-alpha)/logistic_rank + alpha/contextual_rank`.
  Lower alpha wins ties. Alpha zero is the frozen logistic fallback.

No test-based checkpoint or blend selection occurs. Validation results are used
for selection and are not independent performance estimates. Software versions,
weight hashes and selection metadata are saved.

## Submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_contextual_sft_reranker.sh
```

Preparation validates the original sources, reproduces the frozen logistic
validation scores and downloads only the pinned encoder checkpoint if missing
(about 440 MB). It encodes the candidate inputs and freezes data, code and job
files before submitting anything. The first preparation may take several minutes.
It uses the existing `bioasq-8b` environment; it does not install or upgrade packages.

Three GPU jobs are submitted:

| Stage | Budget | Purpose |
|---|---|---|
| Smoke | 1 hour | Two optimizer steps on fitting questions and a small validation subset |
| Train | 8 hours | Full fitting, validation checkpoint selection and hybrid selection |
| Score | 2 hours | Apply frozen choices to unchanged test95 pools |

Each stage depends on successful completion of the previous stage. The full model
starts from the biomedical backbone, not the smoke checkpoint. Offline mode is
enabled on workers. These time limits are resource requests, not runtime estimates.

The launcher also accepts explicit baseline and test-comparison directories as
its first and second arguments. It will reject mismatched sources.

## Monitor

```bash
record=$(ls -t outputs/contextual_sft_reranker/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Training logs:

```bash
job=$(awk -F '\t' '$1=="train" {print $2}' "$record")
number=${job%%.*}
tail -n 60 max5-ce-train.o"$number"
tail -n 60 max5-ce-train.e"$number"
```

If a stage fails, inspect its error log. Do not submit duplicate chains merely
because training or scoring is held behind a dependency.

## Read the results

```bash
module load python3/3.12.13
python3 - <<'PY'
import json
from pathlib import Path
paths = sorted(Path('outputs/contextual_sft_reranker').glob(
    'submission-*/comparison-*/comparison_summary.json'), key=lambda p: p.stat().st_mtime)
if not paths:
    raise SystemExit('No completed score summary yet; inspect the job status/logs.')
p = paths[-1]
d = json.loads(p.read_text())
print('Result:', p)
print('Validation-selected hybrid alpha:', d['blend_alpha'])
for pool in ('max5_greedy', 'max5_sampling10'):
    print('\n' + pool)
    for method in ('logistic', 'contextual', 'hybrid'):
        m = d['conditions'][f'{pool}_{method}']['metrics']
        print(f"  {method}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, full pool={m['coverage_full_pool']:.4f}")
    for method in ('contextual', 'hybrid'):
        name = f'{pool}_{method}_minus_logistic'
        c = d['paired_contrasts'][name]['metrics']['mrr_at5']
        t = d['coverage_at5_transitions'][name]
        print(f"  {method} minus logistic: {c['difference']:+.6f}, CI={c['paired_bootstrap_95ci']}, top-five gained={len(t['gained'])}, lost={len(t['lost'])}")
PY
```

The result includes per-batch scores, paired bootstrap intervals, question-level
coverage gains/losses and full ranked pools. Full-pool coverage must be identical
across all three methods. No candidate strings are rewritten or removed.

## Verification and limits

Local tests cover native tokenizer framing, gold-blind evidence selection, input
alignment, partition separation, frozen artifacts, loss weighting, ranking and
blend selection. A real tiny BERT model is trained, selected, saved and reloaded
on CPU, including an end-to-end synthetic preparation/train/test95 scoring run.
These tests verify the workflow; they do not establish performance for the real
BiomedBERT checkpoint or successful operation on Gadi GPUs. The smoke job provides
the hardware/runtime check.

The design draws on [contextual candidate comparison](https://ojs.aaai.org/index.php/AAAI/article/view/26624).
Error-derived hard-negative emphasis, semantic auxiliary labels and synonym
grouping are reserved for later controlled comparisons.
