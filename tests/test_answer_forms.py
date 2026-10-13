import json
from pathlib import Path
import pytest
from scripts import audit_answer_forms as audit


@pytest.mark.parametrize('text, expected', [
    ('PD-1', 'abbreviation'), ('BRCA1', 'abbreviation'), ('mRNA', 'abbreviation'),
    ('epidermal growth factor receptor', 'full_name'), ('Anakinra', 'full_name'),
    ('epidermal growth factor receptor (EGFR)', 'combined_form'),
    ('EGFR (epidermal growth factor receptor)', 'combined_form'),
    ('Etanercept (Enbrel)', 'full_name'), ('24–30 months', 'numerical_expression'),
    ('2 to 2.5 years', 'numerical_expression'), ('15', 'numerical_expression'),
    ('Hepcidin contains four disulfide bridges', 'descriptive_phrase'),
    ('poly(ADP-ribose) polymerase', 'full_name'),
])
def test_surface_forms_do_not_confuse_symbols_with_numbers_or_brands_with_abbreviations(text,expected):
    assert audit.classify(text)['primary_form'] == expected


def test_stable_ordering_preserves_exact_strings_and_full_pool():
    original=['EGFR','epidermal growth factor receptor','epidermal growth factor receptor (EGFR)', 'another name']
    assert audit.order_answers(original,'original')==original
    for policy in audit.ORDERS:
        ordered=audit.order_answers(original,policy)
        assert sorted(ordered)==sorted(original)
    assert audit.order_answers(original,'full_name_first')[:2]==[original[1],original[3]]
    assert audit.order_answers(original,'combined_first')[0]==original[2]
    assert audit.classify('Anakinra')['needs_review']  # Plain name is not a proven expanded abbreviation.


def test_all_accepted_aliases_are_retained_with_overlapping_question_forms():
    panel=[{'question_id':'a','split':'train','question':'Which?','gold_aliases':['EGFR','epidermal growth factor receptor']},
           {'question_id':'b','split':'validation','question':'Which?','gold_aliases':['Anakinra']}]
    aliases,questions,summary=audit.alias_audit(panel)
    assert [a['accepted_alias'] for a in aliases]==panel[0]['gold_aliases']+panel[1]['gold_aliases']
    assert questions[0]['multiple_forms'] and questions[0]['multiple_aliases']
    assert summary['train']['questions_with_each_form']['abbreviation']==1
    assert summary['train']['questions_with_each_form']['full_name']==1


def test_validation_ordering_on_identical_pools_and_stochastic_first_draw(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(audit.ROOT/'gadi_sft_8b_starter/scripts'))
    from src.notebook_workflows import coverage_comparison as coverage
    from scripts import analyze_matched_qwen_expansion as matched
    from run_single_answer_sampling import expansion_candidates
    run=tmp_path/'run';run.mkdir()
    output=tmp_path/'report';output.mkdir()
    panel=[{'question_id':str(i),'split':'validation','question':'Which receptor?',
            'gold_aliases':['EGFR'], 'snippets':[{'text':'supplied evidence'}]} for i in range(144)]
    generations,candidates=[],[]
    for example in panel:
        samples=[]
        for draw in range(1,11):
            values=[{'answer':'epidermal growth factor receptor','relation_type':'original'},
                    {'answer':'EGFR','relation_type':'abbreviation_expansion'}]
            raw=json.dumps({'answers':values})
            from run_extractive_expansion_8b import parse_equivalent_response
            parsed,*_=parse_equivalent_response(raw,5)
            samples.append({'draw':draw,'raw_response':raw,'answers':parsed})
        row={'question_id':example['question_id'],'samples':samples,
             'snippets_truncated':False,'included_snippets':1,'total_snippets':1}
        generations.append(row)
        candidates.extend({'question_id':example['question_id'],**c} for c in expansion_candidates(samples))
    audit.write(run/'config.json',{'max_candidates':5,'num_generations':10})
    audit.write(run/'status.json',{'status':'complete'})
    examples=[{k:v for k,v in e.items() if k!='split'} for e in panel]
    for filename,rr in [('examples.jsonl',examples),('generations.jsonl',generations),('candidates.jsonl',candidates)]:
        (run/filename).write_text(''.join(json.dumps(r)+'\n' for r in rr),encoding='utf-8')
    monkeypatch.setattr(coverage,'official_candidate_matches',lambda ee,cc,*a,**k:
        {(c['question_id'],c['answer']):c['answer']=='EGFR' for c in cc})
    monkeypatch.setattr(matched,'paired_contrast',lambda a,b,metrics:
        {m:{'sft_minus_base':sum(y[m]-x[m] for x,y in zip(a,b))/144} for m in metrics})
    result=audit.compare_validation(run,panel,output)
    assert result['rules_use_gold'] is False
    for mode in result['results'].values():
        assert mode['original']['metrics']['mrr_at5']==.5
        assert mode['abbreviation_first']['metrics']['mrr_at5']==1
        assert mode['abbreviation_first']['minus_original']['coverage_full_pool']['difference']==0
    # Evaluation rejects external/dev question membership before any matching.
    panel[0]['question_id']='development-id'
    with pytest.raises(ValueError,match='IDs differ'):
        audit.compare_validation(run,panel,output)
