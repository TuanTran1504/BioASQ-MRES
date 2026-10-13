import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gadi_sft_8b_starter/scripts'))
import original_dpo_checkpoint_selection as select
from prepare_matched_8b_sft import write, digest


def checkpoints():
    return [{'epoch':1.0, 'global_step':13, 'eval_loss':.6, 'validation_mrr':.7},
            {'epoch':2.0, 'global_step':26, 'eval_loss':.5, 'validation_mrr':.6}]


def test_independent_mrr_and_loss_selection_and_earliest_ties():
    rows = checkpoints()
    winners = select.choose_winners(rows, 2)
    assert winners['best_validation_mrr']['epoch'] == 1
    assert winners['best_eval_loss']['epoch'] == 2
    rows[1]['validation_mrr'] = .7
    assert select.choose_winners(rows, 2)['best_validation_mrr']['epoch'] == 1
    with pytest.raises(ValueError, match='every complete epoch'):
        select.choose_winners(rows[1:], 2)


def test_export_uses_actual_winning_weights_not_loss_weights(tmp_path):
    rows = checkpoints()
    for row in rows:
        checkpoint = tmp_path/'trainer_output'/f"checkpoint-{row['global_step']}"
        checkpoint.mkdir(parents=True)
        write(checkpoint/'trainer_state.json', {'global_step':row['global_step'],'epoch':row['epoch']})
        write(checkpoint/'adapter_config.json', {'base_model_name_or_path':'pinned'})
        (checkpoint/'adapter_model.safetensors').write_bytes(f"epoch-{row['epoch']}".encode())
    saved = tmp_path/'adapter'
    saved.mkdir()
    write(saved/'tokenizer_config.json', {'chat_template':'native original'})
    (saved/'adapter_model.safetensors').write_bytes(b'loaded-loss-winner')
    result = select.export_selected_adapters(tmp_path, rows, 2)
    for name, epoch in [('best_validation_mrr',1.0),('best_eval_loss',2.0)]:
        path=Path(result[name]['path'])
        assert (path/'adapter_model.safetensors').read_bytes() == f'epoch-{epoch}'.encode()
        assert json.loads((path/'tokenizer_config.json').read_text())['chat_template']=='native original'
        assert result[name]['files_sha256']['adapter_model.safetensors']==digest(path/'adapter_model.safetensors')


def test_scoring_retains_failed_questions_and_uses_official_mrr(tmp_path, monkeypatch):
    import src.utility.bioasq_official as official
    examples=[{'question_id':'a','question':'Which?','gold_aliases':['accepted']},
              {'question_id':'b','question':'Which?','gold_aliases':['other']}]
    generations=[{'question_id':'a','raw_response':'Answer: [BE]accepted[EE]','answer':'accepted'},
                 {'question_id':'b','raw_response':'bad format','answer':None}]
    def score(**kwargs):
        gold=json.loads(kwargs['gold_path'].read_text())['questions']
        assert len(gold)==1 and gold[0]['exact_answer']==[['accepted']]
        return [{'question_id':'a','mrr':1.0}]
    monkeypatch.setattr(official,'_run_official_per_question',score)
    metrics=select.score_answers(examples,generations,tmp_path)
    assert metrics['mrr']==.5 and metrics['question_count']==2 and metrics['parse_success_count']==1
    with pytest.raises(ValueError,match='retained exactly once'):
        select.score_answers(examples,generations[:1],tmp_path)
    generations[0]['answer']='changed'
    with pytest.raises(ValueError,match='raw tagged parse'):
        select.score_answers(examples,generations,tmp_path)


def test_real_official_java_single_answer_and_failure_denominator(tmp_path):
    examples=[{'question_id':'a','question':'Which?','gold_aliases':['accepted']},
              {'question_id':'b','question':'Which?','gold_aliases':['other']},
              {'question_id':'c','question':'Which?','gold_aliases':['missing']}]
    generations=[{'question_id':'a','raw_response':'Answer: [BE]accepted[EE]','answer':'accepted'},
                 {'question_id':'b','raw_response':'Answer: [BE]wrong[EE]','answer':'wrong'},
                 {'question_id':'c','raw_response':'bad format','answer':None}]
    metrics=select.score_answers(examples,generations,tmp_path)
    assert metrics['mrr']==pytest.approx(1/3) and metrics['correct_count']==1


def test_generation_restores_trainability_rng_and_native_flags_on_error():
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.ones(1))
            self.config=SimpleNamespace(use_cache=False)
            self.gradient_checkpointing='unsloth'
            self._gradient_checkpointing_func=object()
        @property
        def is_gradient_checkpointing(self):return bool(self.gradient_checkpointing)
        def gradient_checkpointing_disable(self):self.gradient_checkpointing=False
    model=Model()
    initial=torch.get_rng_state().clone()
    original_func=model._gradient_checkpointing_func
    restored=[]
    def native_restore():
        del model._flag_for_generation
        model.gradient_checkpointing=True
        restored.append(True)
    with pytest.raises(RuntimeError,match='test failure'):
        with select.generation_mode(model,torch,native_restore):
            torch.rand(3)
            model.weight.requires_grad_(False)
            model._flag_for_generation=True
            model._gradient_checkpointing_func=None
            assert not model.training and model.config.use_cache
            raise RuntimeError('test failure')
    assert model.training and model.weight.requires_grad and not model.config.use_cache
    assert model.gradient_checkpointing=='unsloth' and model._gradient_checkpointing_func is original_func
    assert not hasattr(model,'_flag_for_generation') and restored==[True]
    assert torch.equal(initial,torch.get_rng_state())


def test_common_144_panel_excludes_fitting_and_dev_and_teacher_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(select,'ROOT',tmp_path)
    source=tmp_path/'source';source.mkdir()
    rows=[{'question_id':f'v{i}', 'messages':[{'role':'system','content':'Single'},
        {'role':'user','content':json.dumps({'question':'Which?','snippets':[{'id':'s1','text':'evidence'}]})},
        {'role':'assistant','content':'teacher answer never used as gold'}]} for i in range(144)]
    audit=[{'question_id':r['question_id'],'internal_split':'validation','gold_aliases':['official']} for r in rows]
    (source/'audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit))
    write(source/'manifest.json',{'output_sha256':{'audit.jsonl':digest(source/'audit.jsonl')}})
    write(tmp_path/'eval.json',{'training_matrix':'matrix.json','input':'dev.jsonl'})
    write(tmp_path/'matrix.json',{})
    (tmp_path/'dev.jsonl').write_text(json.dumps({'question_id':'dev'})+'\n')
    panels={'train':[{'question_id':'fit'}],'validation':rows}
    monkeypatch.setattr(select,'split_rows',lambda *a:panels)
    monkeypatch.setattr(select,'load_source',lambda *a:(source,{},{}))
    examples,prompt=select.validation_examples({'evaluation_config':'eval.json'})
    assert len(examples)==144 and prompt=='Single'
    assert all(e['gold_aliases']==['official'] for e in examples)
    assert 'teacher answer' not in json.dumps(examples)
    (tmp_path/'dev.jsonl').write_text(json.dumps({'question_id':'v0'})+'\n')
    with pytest.raises(ValueError,match='disjoint'):
        select.validation_examples({'evaluation_config':'eval.json'})


def test_three_arm_validation_checks_third_prompt(tmp_path):
    from scripts.analyze_matched_qwen_expansion import validate_pair
    paths=[]
    for name in ('sft','loss','mrr'):
        path=tmp_path/name;path.mkdir();paths.append(path)
        write(path/'config.json',{'question_count':1,'prompt_sha256':'same'})
        (path/'examples.jsonl').write_text(json.dumps({'question_id':'q'})+'\n')
        (path/'generations.jsonl').write_text(json.dumps({'question_id':'q'})+'\n')
        write(path/'status.json',{'status':'complete'})
    validate_pair(paths,1)
    write(paths[2]/'config.json',{'question_count':1,'prompt_sha256':'different'})
    with pytest.raises(ValueError,match='settings differ'):
        validate_pair(paths,1)


def test_real_trainer_scores_each_epoch_and_resumes_dpo_updates(tmp_path,monkeypatch):
    """CPU integration of the actual Trainer override, including native mode changes."""
    import ast
    import inspect
    import textwrap
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    import run_expansion_dpo_8b as runner
    examples=[{'question_id':'a','question':'Question A','snippets':[{'snippet_id':'s1','text':'evidence'}],
               'gold_aliases':['GOLD_ACCEPTED']},
              {'question_id':'b','question':'Question B','snippets':[{'snippet_id':'s1','text':'evidence'}],
               'gold_aliases':['other']}]
    monkeypatch.setattr(select,'validation_examples',lambda c:(examples,'Single'))
    rendered=[]
    class Tokenizer:
        pad_token_id=0
        eos_token_id=9
        def apply_chat_template(self,messages,**kwargs):
            text=json.dumps(messages);rendered.append(text);return text
        def __call__(self,*,text,return_tensors=None,**kwargs):
            ids=[1,2 if 'Question A' in text else 3]
            return {'input_ids':torch.tensor([ids]) if return_tensors else ids}
        def decode(self,ids,**kwargs):
            return 'Answer: [BE]GOLD_ACCEPTED[EE]' if ids[0].item()==10 else 'bad format'
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(16,8)
            self.head=torch.nn.Linear(8,16)
            self.config=SimpleNamespace(use_cache=False)
            self.gradient_checkpointing='unsloth'
        @property
        def is_gradient_checkpointing(self):return bool(self.gradient_checkpointing)
        def gradient_checkpointing_disable(self):self.gradient_checkpointing=False
        def forward(self,input_ids,attention_mask=None,use_cache=False):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))
        def generate(self,input_ids,**kwargs):
            self._flag_for_generation=True
            for p in self.parameters():p.requires_grad_(False)
            torch.rand(3)
            last=torch.tensor([[10 if input_ids[0,-1].item()==2 else 11]])
            return torch.cat([input_ids,last],dim=1)
    model=Model();tokenizer=Tokenizer()
    def restore():
        if hasattr(model,'_flag_for_generation'):del model._flag_for_generation
    config={'max_seq_length':32,'max_new_tokens':4,'seed':3407}
    tracker=select.EpochMRRTracker(config,{'system_prompt':'Single','chat_template_kwargs':{}},tmp_path,
                                  restore_for_training=restore)
    cls=next(n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(runner.train))))
             if isinstance(n,ast.ClassDef) and n.name=='ExpansionDPOTrainer')
    namespace={'Trainer':Trainer,'tracker':tracker,'tokenizer':tokenizer,'torch':torch,
               'completion_logps':runner.completion_logps,'dpo_loss':runner.dpo_loss,'config':{'beta':.1}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'trainer-hook','exec'),namespace)
    rows=[]
    for accepted,rejected in [(4,5),(6,7),(8,9)]:
        row={}
        for side,value in [('chosen',accepted),('rejected',rejected)]:
            row[side+'_input_ids']=[1,2,value];row[side+'_labels']=[-100,-100,value]
            with torch.no_grad():
                row['ref_'+side]=runner.completion_logps(model(torch.tensor([[1,2,value]])).logits,
                                                       torch.tensor([[-100,-100,value]]),torch).item()
        rows.append(row)
    initial=model.head.weight.detach().clone()
    arguments=TrainingArguments(output_dir=str(tmp_path/'trainer_output'),use_cpu=True,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,num_train_epochs=2,
        gradient_accumulation_steps=2,
        learning_rate=.01,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',
        remove_unused_columns=False,prediction_loss_only=True,label_names=['chosen_labels','rejected_labels'],
        disable_tqdm=True)
    trainer=namespace['ExpansionDPOTrainer'](model=model,args=arguments,train_dataset=Dataset.from_list(rows),
        eval_dataset=Dataset.from_list(rows),data_collator=lambda r:runner.collate_pairs(r,0,torch))
    trainer.train()
    assert [r['epoch'] for r in tracker.rows]==[1.0,2.0]
    assert all(r['validation_mrr']==.5 for r in tracker.rows)
    assert trainer.state.global_step==4 and not torch.equal(initial,model.head.weight)
    assert all(p.requires_grad for p in model.parameters()) and not hasattr(model,'_flag_for_generation')
    assert all('GOLD_ACCEPTED' not in text and 'gold_aliases' not in text for text in rendered)


def test_dual_selected_dev_report_has_both_dpo_contrasts(tmp_path,monkeypatch):
    from scripts import analyze_original_sft_dpo as analyze
    import src.notebook_workflows.local_expansion as local
    from run_single_answer_sampling import unique_candidates
    manifest={'status':'complete','expected_questions':160,'smoke_test':False,'model_key':'llama31',
              'pair_strategy':'concept','runs':{},'arm_hashes':{}}
    examples=[{'question_id':str(i)} for i in range(160)]
    for name,correct in [('sft',40),('dpo_best_eval_loss',45),('dpo_best_validation_mrr',50)]:
        path=tmp_path/name;path.mkdir()
        write(path/'config.json',{'question_count':160,'prompt_sha256':'same'})
        generations=[];candidates=[]
        for i in range(160):
            answer='accepted' if i<correct else 'wrong'
            sample={'draw':1,'raw_response':f'Answer: [BE]{answer}[EE]','answer':answer}
            generations.append({'question_id':str(i),'samples':[sample],'snippets_truncated':False})
            candidates.extend({'question_id':str(i),**row} for row in unique_candidates([sample]))
        for filename,rows in [('examples.jsonl',examples),('generations.jsonl',generations),('candidates.jsonl',candidates)]:
            (path/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write(path/'status.json',{'status':'complete'})
        manifest['runs'][name]=name
        manifest['arm_hashes'][name]={f:digest(path/f) for f in ('config.json','examples.jsonl','generations.jsonl','candidates.jsonl','status.json')}
    write(tmp_path/'manifest.json',manifest)
    def scored(path,**kwargs):
        rows=[]
        for r in json.loads('['+','.join((path/'generations.jsonl').read_text().splitlines())+']'):
            answer=r['samples'][0]['answer']
            rows.append({'question_id':r['question_id'],'answers':[answer],
                         'matching_answers':[answer] if answer=='accepted' else []})
        return path,{},rows
    monkeypatch.setattr(local,'analyze_local_expansion',scored)
    monkeypatch.setattr(analyze,'efficiency',lambda p:{})
    # Bootstrap implementation is covered separately; preserve the actual deltas here.
    def paired(a,b):
        delta=sum(y['mrr_at5']-x['mrr_at5'] for x,y in zip(a,b))/160
        return {'mrr_at5':{'sft_minus_base':delta,'paired_bootstrap_95ci':[delta,delta]}}
    monkeypatch.setattr(analyze,'paired_contrast',paired)
    monkeypatch.setattr(sys,'argv',['analyze',str(tmp_path)])
    analyze.main()
    result=json.loads((tmp_path/'comparison_summary.json').read_text())
    assert result['conditions']['dpo_best_validation_mrr']['metrics']['mrr_at5']==50/160
    assert result['paired_contrasts']['dpo_best_eval_loss_minus_sft']['mrr_at5']['dpo_minus_sft']==5/160
    assert result['paired_contrasts']['mrr_selected_minus_loss_selected']['mrr_at5']['difference']==5/160
