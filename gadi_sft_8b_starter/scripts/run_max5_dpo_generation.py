#!/usr/bin/env python3
"""Generate audited max-five banks only; no DPO training or paid judging."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from max5_dpo_data import DEFAULT_CONFIG, MODELS, panels, selected_source, answers
from prepare_matched_8b_sft import ROOT, read, write, digest
from run_extractive_expansion_8b import configure_job_local_compiler_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    config=read(DEFAULT_CONFIG)
    source=selected_source(config,args.model)
    panel,_=panels(config)
    if args.validate_only:
        print(f"Validated {args.model}: loss-selected max-five SFT; 1296 fitting / 144 validation / no dev160 generation")
        return
    # The shared generator imports the portable training/runtime helpers at
    # execution time. Keep its src package ahead of the repository's src.
    sys.path.insert(0,str(ROOT))
    name=f"{args.model}-generate-{os.environ.get('PBS_JOBID','manual')}"+('-smoke' if args.smoke_test else '')
    output=ROOT/config['output_root']/name
    output.mkdir(parents=True,exist_ok=False)
    configure_job_local_compiler_cache()
    state={'status':'running','model_key':args.model,'smoke_test':args.smoke_test,'full_dataset':True,
           'configuration':config,'source':source,'gold_blind_generation':True,
           'dpo_config_sha256':digest(DEFAULT_CONFIG), 'formulation':'expansion',
           'worker_sha256':digest(Path(__file__)),
           'package_versions':{n:version(n) for n in ('torch','transformers','peft','unsloth')}}
    # Reuse generation and native loaders, but pass the exact max-five panels.
    # Generator's split_rows expects a training_matrix pointer via evaluation_config.
    import run_expansion_dpo_8b as worker
    old_split=worker.split_rows
    worker.split_rows=lambda cfg,full:panel
    try:
        worker.generate(argparse.Namespace(model=args.model,full_dataset=True,smoke_test=args.smoke_test),
                        config,source,output,state)
        if args.smoke_test:
            from prepare_matched_8b_sft import records
            rr=records(output/'responses.jsonl')
            for split in ('train','validation'):
                valid=0
                for row in rr:
                    if row['split']!=split:continue
                    try:answers(row['raw_response']);valid+=1
                    except (ValueError,TypeError):pass
                if not valid:raise ValueError(f'No strict max-five {split} response in smoke')
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        worker.split_rows=old_split
        write(output/'manifest.json',state)
    print(f"Complete bank: {output}")


if __name__=='__main__':main()
