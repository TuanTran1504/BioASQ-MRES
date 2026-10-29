# Comparing model-specific and shared generator DPO preferences

This experiment compares two ways to use the fully inspected official-answer-form
preferences. It trains the **generation models**, starting from their original
max-five SFT adapters. It does not train the answer selectors or logistic reranker.

| Design | Llama training preferences | Qwen training preferences | Ministral training preferences |
|---|---|---|---|
| Model-specific | 132 own-model pairs | 132 own-model pairs | 132 own-model pairs |
| Shared | Same pooled 132 pairs | Same pooled 132 pairs | Same pooled 132 pairs |

The shared file has 44 sampling slots per response author, with exact duplicate
preferences across authors collapsed. All original authors and response IDs remain
recorded. Each target serializes both completions with its own tokenizer and chat
template and calculates its own frozen starting-SFT reference log probabilities.
Reference scores from a different model are never reused.

## Controls and limits

- All six runs use beta 0.1, learning rate 1e-6, seed 3407, three epochs, batch size
  one and accumulation four: **99 optimizer updates per run**. Full sequences are
  checked with each target tokenizer. One common context allocation covers the
  longest pair across all six runs; preparation rejects overlong inputs rather
  than truncating evidence or completions.
- Model-specific samples span 75 Llama, 73 Qwen and 81 Ministral fitting questions;
  shared spans 106. There are only 14 questions common to the three selected
  model-specific samples. This is a practical comparison of data designs with
  equal record/update counts, **not an isolated test of authorship**, nor a matched
  token/FLOP budget. Broader question coverage can contribute to a shared advantage.
- All runs have the same **96 preference holdout pairs from 40 fitting questions**,
  disjoint from their preference-training questions. This holdout is not the
  separate validation144 panel. No validation144, dev160 or test95 question enters
  preference construction.
- These are reviewed **official-form preferences**. Both alternatives can be valid
  biomedical wording. Better official acceptance does not establish that the
  rejected response is medically false. This experiment does not reward shorter
  responses or claim semantic correction.
- Equal exposure is achieved by deterministic question-balanced downsampling,
  without duplicating records. The original full reviewed export remains intact.
- The frozen source banks, SFT adapter lineage, reranker, evaluator JAR, package
  contents, inference workers and code snapshots are checked before execution.
  This launch requires no new API calls, model downloads or paid judging.

## Evaluation and checkpoint selection

Each model/design has GPU smoke -> DPO training -> validation generation/scoring.
The launcher submits six such chains plus one dependent CPU comparison (19 jobs).
The smoke stage uses the longest pairs to exercise memory use before full training.

Every saved DPO epoch is evaluated on all **144 internal-validation questions**
using greedy generation and **four sampled responses/model/question** (temperature
0.8, top-p 0.95), matching the saved source banks. Epoch zero reuses SFT outputs.
No test95 evaluation is submitted. Greedy and sampling outputs are separate pools.

For each focal model, the other two generators remain SFT and the original max-five
logistic reranker stays frozen. Candidate pools retain exact strings, including
case/spacing variants. All arms, including the baseline, use this same policy;
historical normalized-pool scores cannot be substituted for this baseline.

Select on sampling4 using this predeclared order:

1. Require no increase versus SFT in empty, output-ceiling or surplus responses.
2. Maximize ensemble full-pool official-answer coverage.
3. Then maximize focal-model full-pool coverage.
4. Then maximize final ensemble MRR@5; break remaining ties with the earliest epoch.

Epoch zero is eligible. Unlike the compactness experiment, selection does not
require shorter output or zero individual question losses. Coverage gains/losses
are reported explicitly, and MRR may decrease under this coverage-first objective.

The CPU report compares each focal model/design and also combines all three
selected model-specific outputs versus all three selected shared outputs. It also
reports **fixed epoch-three** comparisons, so checkpoint-selection effects are
visible. Joint combinations use saved outputs; they do not search checkpoint
combinations or generate again. Paired differences include MRR@5, top-five/full-pool
coverage and question-level gains/losses.

Validation selects checkpoints and permits choosing a design. Its confidence
intervals are descriptive. This cannot establish held-out superiority, and these
six runs do not isolate DPO versus additional preferred-only SFT training.

## Prepare and upload the reviewed package

The local package is already prepared at
`Artifacts/generator_form_designs_20261029_v2`; its archive is
`Artifacts/generator-form-designs-20261029-v2.tar.gz`.
Preparation alone launches no training. The package's `training_ready: false`
indicates that native Gadi checks and GPU smoke execution are still required.

To reproduce the package locally, use a **new output name**:

```powershell
.venv\Scripts\python.exe scripts/prepare_generator_form_designs.py `
  --inspection Artifacts/generator_form_full_inspection_20261029_v2 `
  --selector-audit Artifacts/answer_selector_data_v1 `
  --output Artifacts/generator_form_designs_20261029_v2 `
  --archive Artifacts/generator-form-designs-20261029-v2.tar.gz
```

From Windows PowerShell:

```powershell
cd C:\Users\Dustin\Projects\BioASQ-MRES
scp .\Artifacts\generator-form-designs-20261029-v2.tar.gz dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/
```

Then on Gadi:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
mkdir -p outputs/generator_form_designs/reviewed-20261029-v2
tar -xzf /scratch/nl78/$USER/generator-form-designs-20261029-v2.tar.gz \
  -C outputs/generator_form_designs/reviewed-20261029-v2
bash scripts/submit_generator_form_designs.sh \
  outputs/generator_form_designs/reviewed-20261029-v2
```

Preparation validates all six complete datasets/tokenizers before the first qsub.
It may be quiet while reading and hashing native banks or loading a tokenizer.
If it fails, no jobs have been submitted by that invocation. Do not repeatedly
launch while preparation is still running.

### Tokenizer return-format compatibility

The first Gadi preparation stopped on question `6206b5a8c9dfcb9c0900003a`
with `native assistant tokens do not extend the generation prompt`. The
underlying check also rejected non-list results, making its message ambiguous.
Local reproduction with the public Llama tokenizer showed that the prompt and
completion prefixes match, but Transformers returns `BatchEncoding` by default.
Its `input_ids` are the expected token list.

The DPO renderer now passes `return_dict=False` explicitly for both prompt and
completed conversations. It still rejects unexpected structures, genuine prefix
mismatches and overlength inputs. Completion masking, raw responses, preferences
and checkpoint selection are unchanged. Regression tests include a real fast
tokenizer and deliberately invalid return structures/prefixes. Public-tokenizer
checks are supplementary; Gadi still verifies its actual saved SFT tokenizers.

This failure occurred before submission creation/qsub. After pulling the fix,
rerun the same launcher with `reviewed-20261029-v2`. The existing extracted data
package can be reused; no new archive upload or paid review is needed.

## Monitor and read results

On Gadi, from the starter directory:

```bash
module load python3/3.12.13
python3 - <<'PY'
import csv, json, subprocess
from pathlib import Path
records = sorted(Path('outputs/generator_form_designs').glob('submission-*/jobs.tsv'),
                 key=lambda p: p.stat().st_mtime)
if not records:
    raise SystemExit('No submission record yet.')
record = records[-1]
print('Record:', record)
pending = []
with record.open() as f:
    for r in csv.DictReader(f, delimiter='\t'):
        path = Path(r['output']) / 'manifest.json'
        d = json.loads(path.read_text()) if path.exists() else {}
        print(r['model'], r['design'], r['stage'], r['job_id'], d.get('status', 'no saved status'))
        for k in ('global_steps', 'selected_epoch', 'error'):
            if k in d: print(' ', k, d[k])
        report = Path(r['output']) / 'report.md'
        if report.exists(): print(report.read_text())
        if d.get('status') != 'complete': pending.append(r['job_id'])
if pending:
    subprocess.run(['qstat', '-swx', *pending])
PY
```

Child training and selection outputs are under
`submission-.../<model>/<design>/train-<job>` and `validation-<job>`.
The combined result is under `submission-.../comparison-<job>/report.md`, with
`comparison_summary.json`, `comparison_per_question.json` and `joint_rankings.jsonl`.
Each validation summary also contains model-alone coverage and loss IDs.

If a stage fails, inspect its stderr before resubmitting. Existing completed
generation is preserved. A failed CPU comparison can be rerun alone using the
frozen `compare.pbs` and `FORM_COMPARISON` pointing to the parent submission;
this does not rerun generation or training. Repeating the main launcher creates
a new six-run experiment.
