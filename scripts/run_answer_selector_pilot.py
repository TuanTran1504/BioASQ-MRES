#!/usr/bin/env python3
"""Supervised, tie-aware 8B answer selectors; fitting/validation only, no DPO."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))
from scripts import prepare_answer_selector_data as data
from scripts import answer_selector_core as core

CONFIG = ROOT / "gadi_sft_8b_starter/configs/answer_selector_pilot.json"
POOLS = ("max5_greedy", "max5_sampling4")
MODELS = {
    'llama31': ('fast_language_model', {'llama'}),
    'qwen3': ('fast_language_model', {'qwen3'}),
    'ministral3': ('fast_model', {'mistral3', 'ministral3'}),
}


def configuration(repository, model):
    config = data.read(repository / 'gadi_sft_8b_starter/configs/answer_selector_pilot.json')
    if model not in MODELS:
        raise ValueError('Unknown selector backbone: ' + model)
    config['model_key'] = model
    return config


def hashes(directory):
    return {p.relative_to(directory).as_posix(): data.digest(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def verify_backbone(directory, expected):
    """Read-only HF blobs may be symlinks outside snapshots, inside their model cache."""
    directory = directory.resolve()
    cache_root = directory.parent.parent
    for name, digest in expected.items():
        relative = Path(name)
        path = directory / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve().is_relative_to(cache_root) or data.digest(path) != digest):
            raise ValueError("Pinned instruction backbone changed: " + name)


def verify_export(directory):
    state = data.read(directory / "manifest.json")
    if (state.get("status") != "complete" or state.get("schema") != "max5-answer-selector-audit-v1"
            or state.get("question_counts") != data.COUNTS or state.get("model_training")):
        raise ValueError("Need the completed fitting1296/validation144 selector audit")
    data.verify_files(directory, state["output_sha256"])
    inputs, supervision, question_ids = {}, {}, {}
    heldout = data.read(directory / "heldout_ids.json")
    for split in data.SPLITS:
        inputs[split] = data.records(directory / f"{split}_inputs.jsonl")
        supervision[split] = data.records(directory / f"{split}_supervision.jsonl")
        identity = lambda r: (r["question_id"], r["pool"])
        expected = {(r["question_id"], p) for r in inputs[split] for p in POOLS}
        if (len(inputs[split]) != 2 * data.COUNTS[split]
                or {identity(r) for r in inputs[split]} != expected
                or len(supervision[split]) != len(inputs[split])
                or {identity(r) for r in supervision[split]} != expected):
            raise ValueError("Missing, duplicate or unexpected question/pool view")
        question_ids[split] = {r["question_id"] for r in inputs[split]}
        if len(question_ids[split]) != data.COUNTS[split] or question_ids[split] & set(heldout["dev160"] + heldout["test95"]):
            raise ValueError("Wrong panel membership or held-out overlap")
        by_input = {identity(r): r for r in inputs[split]}
        for row in supervision[split]:
            payload = json.loads(by_input[identity(row)]["messages"][1]["content"])
            if (set(payload) != {"question", "snippets", "candidates"}
                    or row.get('split') != split or by_input[identity(row)].get('split') != split
                    or payload["candidates"] != [{"id": r["id"], "answer": r["answer"]} for r in row["candidates"]]
                    or len({r["id"] for r in row["candidates"]}) != len(row["candidates"])
                    or set(row['accepted_ids']) != {r['id'] for r in row['candidates'] if r['official_accepted']}
                    or any(type(r["official_accepted"]) is not bool for r in row["candidates"])):
                raise ValueError("Gold-blind candidate input/official labels differ")
    if question_ids["train"] & question_ids["validation"]:
        raise ValueError("Fitting/validation question overlap")
    return state, inputs, supervision


def validate_config(config):
    if (config.get("schema") != "tie-aware-answer-selector-v1" or config["model_key"] not in MODELS
            or config["batch_size"] != 1 or config["epochs"] != 3 or config["sampled_responses_per_model"] != 4
            or config["max_seq_length"] != 8704 or config["DPO_training"] or config["test_evaluation"]
            or not math.isfinite(config.get('gradient_scaler_initial_scale', 128.0))
            or config.get('gradient_scaler_initial_scale', 128.0) <= 0
            or type(config.get('max_overflow_retries', 12)) is not int
            or not 0 <= config.get('max_overflow_retries', 12) <= 12
            or config["learning_rate"] <= 0 or config["gradient_accumulation_steps"] < 1):
        raise ValueError("Unsupported supervised selector pilot configuration")


def prepare(repository, audit, submission, model_key='llama31'):
    """CPU preflight: local tokenizer only; no model weights loaded or downloaded."""
    config = configuration(repository, model_key)
    validate_config(config)
    state, inputs, supervision = verify_export(audit)
    from scripts.prepare_reranker_guided_dpo import configure_data_root
    native = configure_data_root(repository)
    generation = data.read(repository / "gadi_sft_8b_starter/configs/max5_dpo_banks_8b.json")
    source = native.selected_source(generation, model_key)
    backbone = Path(source["base_snapshot"]).resolve()
    if (source["model_loader"] != MODELS[model_key][0] or backbone.name != source["base_revision"]
            or not (backbone / "config.json").exists()
            or data.read(backbone / "config.json")["model_type"] not in MODELS[model_key][1]):
        raise ValueError("Need the already pinned cached instruction backbone for " + model_key)
    base_hashes = hashes(backbone)
    verify_backbone(backbone, base_hashes)
    # Set Transformers offline before importing the tokenizer stack on the login node.
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
    chat_kwargs = source.get('chat_template_kwargs', {})
    if model_key == 'qwen3' and chat_kwargs.get('enable_thinking') is not False:
        raise ValueError('Qwen selector requires explicit non-thinking chat rendering')
    maximum = max(len(r["candidates"]) for rows in supervision.values() for r in rows)
    spec = core.encoding_spec(tokenizer, maximum)
    # Verify framing at the tokenizer boundary, not by estimating token counts.
    import json
    first_code = next(iter(spec['codes']))
    example_json = tokenizer.decode(spec["prefix"] + [spec["codes"][first_code]] + spec["closing"])
    if json.loads(example_json) != [first_code]:
        raise ValueError("Tokenizer does not preserve the forced JSON ID format")
    submission.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(str(submission / "tokenizer"))
    lengths = []
    for split in data.SPLITS:
        by_label = {(r["question_id"], r["pool"]): r for r in supervision[split]}
        frozen = []
        for item in inputs[split]:
            case, _ = core.render_case(tokenizer, item, by_label[(item["question_id"], item["pool"])], spec, config["seed"], 0, chat_kwargs)
            budget = core.check_budget(case, None, spec, config["max_seq_length"])
            frozen.append(case)
            lengths.append({"split": split, "question_id": item["question_id"], "pool": item["pool"],
                            "input_tokens": len(case["input_ids"]), "input_plus_reserved_output": budget})
        data.jsonl(submission / f"{split}_generation_inputs.jsonl", frozen)
        data.jsonl(submission / f"{split}_inputs.jsonl", inputs[split])
        data.jsonl(submission / f"{split}_supervision.jsonl", supervision[split])
        if split == "train":
            counts = []
            for epoch in range(1, config["epochs"] + 1):
                training = []
                for item in inputs[split]:
                    case, row = core.render_case(tokenizer, item, by_label[(item["question_id"], item["pool"])], spec, config["seed"], epoch, chat_kwargs)
                    core.check_budget(case, row, spec, config["max_seq_length"])
                    if row is not None:
                        training.append(row)
                data.jsonl(submission / f"train_epoch{epoch}.jsonl", training)
                counts.append({"epoch": epoch, "views": len(training), "questions": len({r['question_id'] for r in training}),
                               "maximum_training_tokens": max(len(r['input_ids']) for r in training)})
    data.write(submission / "encoding_spec.json", spec)
    data.write(submission / "configuration.json", config)
    data.jsonl(submission / "length_audit.jsonl", lengths)
    data.write(submission / "data_summary.json", {"question_counts": data.COUNTS, "training_epochs": counts,
        "validation_views": len(inputs['validation']), "maximum_input_plus_output": max(r['input_plus_reserved_output'] for r in lengths),
        "all_evidence_preserved": True, "candidate_surfaces_preserved": True, "tokenizer_sha256": hashes(submission / 'tokenizer')})
    for filename in ("heldout_ids.json", "source_provenance.json", "manifest.json"):
        shutil.copy2(audit / filename, submission / ("audit_manifest.json" if filename == "manifest.json" else filename))
    from scripts.train_sft_ensemble_reranker import snapshot_code
    snapshot_code(repository, submission / "code")
    for name in ('src/__init__.py', 'src/utility/__init__.py',
                 'src/utility/training_execution.py', 'src/utility/ministral_generation.py'):
        relative = Path('gadi_sft_8b_starter') / name
        target = submission / 'code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    for name in ("baseline_answer_selector.pbs", "train_answer_selector.pbs", "report_answer_selector.pbs"):
        shutil.copy2(repository / "gadi_sft_8b_starter/jobs" / name, submission / name)
    verify_backbone(backbone, base_hashes)
    data.write(submission / "prepared.json", {"schema": config['schema'], "configuration": config,
        "audit_source": str(audit), "audit_manifest_sha256": data.digest(audit / 'manifest.json'),
        "backbone_snapshot": str(backbone), "backbone_files": base_hashes,
        "backbone_model_name": source['model_name'], "backbone_revision": source['base_revision'],
        "model_loader": source['model_loader'], "chat_template_kwargs": chat_kwargs,
        "execution_mode": source.get('execution_mode', 'default'),
        "initialization": "fresh selector LoRA; cached instruction backbone only; no generator SFT adapter",
        "files_sha256": hashes(submission), "training": "supervised only", "DPO_training": False, "test_evaluation": False})
    validate(submission)
    print(f"Prepared tie-aware {model_key} selector: {counts[0]['views']} fitting views / {counts[0]['questions']} questions; "
          f"all 144 validation questions, both pools. Maximum reserved context {max(r['input_plus_reserved_output'] for r in lengths)}.")


def validate(submission, backbone=False):
    state = data.read(submission / "prepared.json")
    validate_config(state["configuration"])
    data.verify_files(submission, state["files_sha256"])
    if state["configuration"] != data.read(submission / "configuration.json"):
        raise ValueError("Frozen selector configuration differs")
    if backbone:
        verify_backbone(Path(state["backbone_snapshot"]), state["backbone_files"])
    return state


def pools_for_baseline(submission):
    import json
    examples, pools = {}, {}
    for split in data.SPLITS:
        examples[split], pools[split] = {}, {p: [] for p in POOLS}
        inputs = {(r['question_id'], r['pool']): r for r in data.records(submission / f"{split}_inputs.jsonl")}
        for row in data.records(submission / f"{split}_supervision.jsonl"):
            context = json.loads(inputs[(row['question_id'], row['pool'])]['messages'][1]['content'])
            examples[split][row['question_id']] = {"question": context['question'], "snippets": context['snippets']}
            pools[split][row['pool']].extend({**{k: v for k, v in c.items() if k not in ('id', 'official_accepted', 'occurrences', 'draw_support')},
                                            'label': int(c['official_accepted'])} for c in row['candidates'])
    return examples, pools


def baseline(submission, output):
    state = validate(submission)
    from scripts import train_sft_ensemble_reranker as rr
    import joblib
    examples, pools = pools_for_baseline(submission)
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'prepared_sha256': data.digest(submission / 'prepared.json')}
    data.write(output / 'manifest.json', status)
    try:
        best, trials = rr.fit_selector(pools, examples, state['configuration']['logistic_C_grid'], ('max5',))
        score, C, components, metrics, questions, rankings = best
        joblib.dump({'components': components, 'selected_C': C}, output / 'logistic.joblib')
        data.write(output / 'summary.json', {'selected_C': C, 'mean_validation_mrr_at5': score,
            'conditions': {p: metrics[p + '_logistic'] for p in POOLS}, 'trials': trials,
            'scope': 'fitting-only logistic; C selected on all144; identical prospective verbatim pools'})
        data.write(output / 'per_question.json', {p: questions[p + '_logistic'] for p in POOLS})
        data.jsonl(output / 'rankings.jsonl', rankings)
        status.update(status='complete', artifacts_sha256=hashes(output))
        # The status file changes in the finally block and is not self-hashed.
        status['artifacts_sha256'].pop('manifest.json', None)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        data.write(output / 'manifest.json', status)


def verify_result(directory, prepared_hash, smoke=None):
    state = data.read(directory / 'manifest.json')
    if state.get('status') != 'complete' or state.get('prepared_sha256') != prepared_hash:
        raise ValueError('Need completed result from this frozen pilot: ' + str(directory))
    if smoke is not None and state.get('smoke') != smoke:
        raise ValueError('Wrong smoke/full result')
    data.verify_files(directory, state['artifacts_sha256'])
    return state


def evaluate_model(model, cases, supervision, spec, device, torch):
    model.eval()
    predicted = []
    for i, case in enumerate(cases, 1):
        started = time.perf_counter()
        codes, identities = core.generate_ids(model, case, spec, device, torch)
        predicted.append({'question_id': case['question_id'], 'pool': case['pool'],
                          'selected_codes': codes, 'selected_source_ids': identities,
                          'generation_seconds': time.perf_counter() - started})
        if i % 32 == 0:
            print(f"Validated selector {i}/{len(cases)} views", flush=True)
    metrics, scores = core.score_predictions(predicted, supervision)
    return metrics, scores, predicted


def load_model(state):
    # Set before importing Unsloth/Torch; actual training workers only.
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    from run_extractive_expansion_8b import configure_job_local_compiler_cache
    configure_job_local_compiler_cache()
    # Data tooling already imports the repository's src package. Load the small
    # portable GPU helpers by exact path instead of replacing that shared package.
    execution = gpu_helper('training_execution')
    mode = state.get('execution_mode', 'default')
    execution.prepare_execution(mode)
    import unsloth
    import torch
    execution.activate_execution(mode, torch)
    if not torch.cuda.is_available():
        raise RuntimeError('The selector requires a GPU worker')
    config = state['configuration']
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    loader_name = state.get('model_loader', MODELS[config['model_key']][0])
    if loader_name != MODELS[config['model_key']][0]:
        raise ValueError('Wrong selector loader')
    loader = unsloth.FastModel if loader_name == 'fast_model' else unsloth.FastLanguageModel
    options = dict(model_name=state['backbone_snapshot'], max_seq_length=config['max_seq_length'],
                   dtype=torch.float16, load_in_4bit=True, local_files_only=True)
    if loader_name == 'fast_model':
        options['return_logits'] = True
    model, _ = loader.from_pretrained(**options)
    if getattr(model, 'peft_config', None):
        raise ValueError('Expected instruction backbone, not an existing SFT/DPO adapter')
    peft = dict(r=config['lora_r'], lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'], bias='none', random_state=config['seed'],
        use_gradient_checkpointing='unsloth', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    if loader_name == 'fast_model':
        peft.update(finetune_vision_layers=False, finetune_language_layers=True,
                    finetune_attention_modules=True, finetune_mlp_modules=True)
    model = loader.get_peft_model(model, **peft)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and 'lora_' not in name:
            raise ValueError('Unexpected trainable backbone parameter: ' + name)
        if parameter.requires_grad and any(s in name.lower() for s in ('vision', 'visual', 'projector')):
            raise ValueError('Only language LoRA parameters may be trained: ' + name)
        if parameter.requires_grad:
            # GradScaler needs FP32 trainable gradients; base weights remain 4-bit.
            parameter.data = parameter.data.float()
    return model, loader, torch


def gpu_helper(name):
    if name not in ('training_execution', 'ministral_generation'):
        raise ValueError('Unexpected GPU helper')
    path = ROOT / 'gadi_sft_8b_starter/src/utility' / (name + '.py')
    spec = importlib.util.spec_from_file_location('selector_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optimizer_group(model, group, optimizer, scaler, trainable, device, torch,
                    max_grad_norm, max_overflow_retries=12, record_attempt=None):
    """Replay the same accumulation group after AMP overflow, never dropping views.

    GradScaler skips an update and lowers its scale on nonfinite gradients.
    A retry does not advance the data cursor or learning-rate scheduler. Persistent
    invalid gradients fail closed after a bounded number of attempts.
    """
    if not group or not 0 <= max_overflow_retries <= 12:
        raise ValueError('Need a nonempty optimizer group and bounded overflow retries')
    for attempt in range(1, max_overflow_retries + 2):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for row in group:
            tokens = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
            with torch.autocast(device.type if hasattr(device, 'type') else str(device), dtype=torch.float16):
                logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
                loss = core.marginal_id_loss(logits, row['actions'])
            total_loss += float(loss.detach())
            scaler.scale(loss / len(group)).backward()
            del logits, loss, tokens
        scaler.unscale_(optimizer)
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads:
            raise ValueError('Selector produced no trainable gradients')
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads]))
        finite = math.isfinite(float(norm))
        if finite:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        elif all(bool(torch.isfinite(g).all()) for g in grads):
            raise ValueError('Nonfinite aggregate gradient norm despite finite gradients; refusing update')
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        after = scaler.get_scale()
        stepped = after >= before
        event = {'attempt': attempt, 'loss': total_loss / len(group),
                 'gradient_norm': float(norm) if finite else None, 'gradients_finite': finite,
                 'loss_scale_before': before, 'loss_scale_after': after, 'optimizer_stepped': stepped,
                 'question_ids': [r['question_id'] for r in group]}
        if record_attempt is not None:
            record_attempt(event)
        if stepped:
            if not finite:
                raise ValueError('AMP reported an update with nonfinite gradients')
            return event
        if finite:
            raise ValueError('AMP skipped an update despite finite gradients')
        if attempt <= max_overflow_retries:
            print(f'AMP overflow: scale {before:g} -> {after:g}; retrying the same {len(group)} views', flush=True)
    optimizer.zero_grad(set_to_none=True)
    raise ValueError(f'Persistent nonfinite gradients after {max_overflow_retries + 1} attempts; '
                     'inspect optimizer_attempts.jsonl; no data views were skipped')


def train(submission, output, smoke=False, smoke_result=None, baseline_result=None):
    state = validate(submission, backbone=True)
    prepared_hash = data.digest(submission / 'prepared.json')
    if not smoke:
        if smoke_result is None or baseline_result is None:
            raise ValueError('Full training requires this submission\'s successful smoke and CPU baseline')
        verify_result(smoke_result, prepared_hash, smoke=True)
        verify_result(baseline_result, prepared_hash)
    config, spec = state['configuration'], data.read(submission / 'encoding_spec.json')
    cases = data.records(submission / 'validation_generation_inputs.jsonl')
    labels = data.records(submission / 'validation_supervision.jsonl')
    keep = None
    if smoke:
        # Longest fitting and validation contexts catch memory/length problems early.
        long = sorted(cases, key=lambda r: len(r['input_ids']), reverse=True)
        keep = {r['question_id'] for r in long[:2]}
        keep.update(sorted({r['question_id'] for r in cases})[:2])
        cases = [r for r in cases if r['question_id'] in keep]
        labels = [r for r in labels if r['question_id'] in keep]
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'prepared_sha256': prepared_hash, 'smoke': smoke,
              'model_key': config['model_key'], 'DPO_training': False, 'test_evaluation': False}
    data.write(output / 'manifest.json', status)
    try:
        model, loader, torch = load_model(state)
        device = next(model.parameters()).device
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError('No trainable selector LoRA parameters')
        optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        epochs = 1 if smoke else config['epochs']
        accumulation = 2 if smoke else config['gradient_accumulation_steps']
        first = data.records(submission / 'train_epoch1.jsonl')
        steps_per_epoch = 2 if smoke else math.ceil(len(first) / accumulation)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, 0 if smoke else math.ceil(steps_per_epoch * epochs * config['warmup_ratio']), steps_per_epoch * epochs)
        scaler = torch.amp.GradScaler('cuda', init_scale=config.get('gradient_scaler_initial_scale', 128.0))
        scores, global_steps, losses = [], 0, []

        def checkpoint(epoch):
            directory = output / f'epoch-{epoch}'
            directory.mkdir()
            adapter = directory / 'adapter'
            model.save_pretrained(str(adapter), safe_serialization=True, save_embedding_layers=False)
            shutil.copytree(submission / 'tokenizer', adapter, dirs_exist_ok=True)
            if not (adapter / 'adapter_model.safetensors').exists():
                raise ValueError('Selector adapter was not saved')
            loader.for_inference(model)
            if config['model_key'] == 'ministral3':
                gpu_helper('ministral_generation').configure_ministral_generation(model)
            metrics, perq, predictions = evaluate_model(model, cases, labels, spec, device, torch)
            data.write(directory / 'validation_summary.json', metrics)
            data.write(directory / 'validation_per_question.json', perq)
            data.jsonl(directory / 'validation_predictions.jsonl', predictions)
            scores.append({'epoch': epoch, **metrics, 'adapter': str(adapter), 'adapter_files': hashes(adapter)})
            data.write(output / 'checkpoint_scores.json', scores)
            print(f"Epoch {epoch}: validation mean MRR@5={metrics['mean_validation_mrr_at5']:.6f}", flush=True)
            loader.for_training(model, use_gradient_checkpointing='unsloth')
            for name, parameter in model.named_parameters():
                parameter.requires_grad_('lora_' in name and not any(s in name.lower() for s in ('vision', 'visual', 'projector')))
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
            model.config.use_cache = False
            model.train()
            torch.cuda.empty_cache()

        checkpoint(0)
        for epoch in range(1, epochs + 1):
            rows = data.records(submission / f'train_epoch{epoch}.jsonl')
            if smoke:
                rows = sorted(rows, key=lambda r: len(r['input_ids']), reverse=True)[:4]
            data.rng(config['seed'], 'training_order', epoch).shuffle(rows)
            epoch_start_steps = global_steps
            for start in range(0, len(rows), accumulation):
                group = rows[start:start + accumulation]
                def record_attempt(event):
                    with (output / 'optimizer_attempts.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps({'epoch': epoch, 'start_view': start, **event}, allow_nan=False) + '\n')
                event = optimizer_group(model, group, optimizer, scaler, trainable, device, torch,
                    config['max_grad_norm'], config.get('max_overflow_retries', 12), record_attempt)
                scheduler.step(); global_steps += 1
                losses.append({'epoch': epoch, 'start_view': start, **event, 'global_steps': global_steps})
                data.jsonl(output / 'training_log.jsonl', losses)
                if len(losses) % 10 == 0 or smoke:
                    print(f"Epoch {epoch}, views {min(start+len(group),len(rows))}/{len(rows)}; "
                          f"loss={event['loss']:.6f}; optimizer steps={global_steps}; scale={scaler.get_scale():g}", flush=True)
            if global_steps - epoch_start_steps != math.ceil(len(rows) / accumulation):
                raise ValueError('Optimizer update count differs from the complete epoch schedule')
            data.jsonl(output / 'training_log.jsonl', losses)
            checkpoint(epoch)
        selected = core.choose_epoch(scores)
        data.write(output / 'selection.json', {'status': 'frozen', 'selected_epoch': selected,
            'checkpoint_scores': scores, 'selection_policy': config['checkpoint_selection'],
            'validation_question_count': len({r['question_id'] for r in cases}),
            'scope': 'Internal validation only; epoch0 included; all question pools retained', 'smoke': smoke})
        validate(submission, backbone=True)
        status.update(status='complete', selected_epoch=selected, global_steps=global_steps,
            runtime={n: str(getattr(__import__(n), '__version__', 'unknown')) for n in ('torch', 'transformers', 'unsloth')},
            device=torch.cuda.get_device_name(0), artifacts_sha256=hashes(output))
        status['artifacts_sha256'].pop('manifest.json', None)
        print('Selected selector epoch:', selected, '(0 = untrained instruction selector)', flush=True)
    except BaseException as exc:
        if 'global_steps' in locals():
            status.update(global_steps=global_steps, loss_scale=scaler.get_scale())
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        data.write(output / 'manifest.json', status)


def report(submission, training, baseline_result, output):
    validate(submission)
    pin = data.digest(submission / 'prepared.json')
    verify_result(training, pin, smoke=False); verify_result(baseline_result, pin)
    selected = data.read(training / 'selection.json')
    epoch = selected['selected_epoch']
    scores = data.read(training / f'epoch-{epoch}/validation_per_question.json')
    base_scores = data.read(baseline_result / 'per_question.json')
    summary = data.read(training / f'epoch-{epoch}/validation_summary.json')
    baseline_summary = data.read(baseline_result / 'summary.json')
    if any(len(scores[p]) != data.COUNTS['validation'] or len(base_scores[p]) != data.COUNTS['validation'] for p in POOLS):
        raise ValueError('Report must retain the complete validation panel in each pool')
    if any(summary['conditions'][p]['coverage_full_pool'] != baseline_summary['conditions'][p]['coverage_full_pool'] for p in POOLS):
        raise ValueError('Selector or baseline changed fixed full-pool membership')
    from scripts.analyze_matched_qwen_expansion import paired_contrast
    contrasts = {p: paired_contrast(base_scores[p], scores[p], metrics=('mrr_at5','coverage_at5','coverage_full_pool')) for p in POOLS}
    predictions = {(r['question_id'],r['pool']):r for r in data.records(training / f'epoch-{epoch}/validation_predictions.jsonl')}
    inputs = {(r['question_id'],r['pool']):r for r in data.records(submission / 'validation_inputs.jsonl')}
    import json
    cases = []
    for row in data.records(submission / 'validation_supervision.jsonl'):
        identity = (row['question_id'],row['pool'])
        by_id = {r['id']:r for r in row['candidates']}
        chosen = predictions[identity]['selected_source_ids']
        accepted = [r for r in row['candidates'] if r['official_accepted']]
        hit = next((i for i,c in enumerate(chosen,1) if by_id[c]['official_accepted']),None)
        if hit == 1:
            continue
        cases.append({'question_id':row['question_id'],'split':'validation','pool':row['pool'],
            'context':json.loads(inputs[identity]['messages'][1]['content']),
            'top_five':[by_id[c] for c in chosen], 'accepted_candidates':accepted, 'first_accepted_rank':hit,
            'diagnosis':'pool_absence' if not accepted else 'accepted_outside_top5' if not hit else 'accepted_below_rank1',
            'semantic_review':'unresolved; official nonacceptance is not factual error',
            'DPO_training_eligible':False, 'reason':'Validation examples are diagnostic only; never training preferences'})
    output.mkdir(parents=True,exist_ok=False)
    result={'question_count':data.COUNTS['validation'], 'model_key': data.read(submission/'configuration.json')['model_key'],
        'selected_epoch':epoch,'checkpoint_scores':selected['checkpoint_scores'],
        'conditions':{'logistic':baseline_summary['conditions'],'selector':summary['conditions']},
        'paired_contrasts':contrasts,'test_evaluation':False,'DPO_pairs_created':0,
        'interpretation':'Validation diagnostics used for model selection; not an independent test improvement'}
    data.write(output/'comparison_summary.json',result)
    data.jsonl(output/'validation_mistakes.jsonl',cases)
    lines=['# Supervised answer-selector pilot: validation only','',f'Selected epoch: {epoch} (0 = instruction selector before supervised updates).',
        '', '| Pool | Logistic MRR@5 | Selector MRR@5 | Difference | 95% paired interval |','|---|---:|---:|---:|---|']
    for p in POOLS:
        c=contrasts[p]['mrr_at5']
        lines.append(f"| {p} | {baseline_summary['conditions'][p]['mrr_at5']:.6f} | {summary['conditions'][p]['mrr_at5']:.6f} | {c['sft_minus_base']:+.6f} | {c['paired_bootstrap_95ci']} |")
    lines += ['', 'These are all144 validation questions, including uncovered pools. The intervals are descriptive because validation selects the checkpoint.',
        'Same prospective verbatim pools; independently refitted fitting-only logistic baseline. Four stochastic draws/model, not ten.',
        'Invalid/duplicate output IDs are prevented by constrained decoding; this does not guarantee correct biomedical answers.',
        'Supervision optimizes official acceptance. Alternative supported wording can receive official label 0.',
        'Mistake cases are validation-only diagnostics. No DPO pairs or training are created; fitting-only preference generation is a separate later stage.', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    data.write(output/'manifest.json',{'status':'complete','prepared_sha256':pin,'DPO_training':False,
        'test_evaluation':False,'artifacts_sha256':hashes(output)})
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('prepare','baseline','train','report'))
    parser.add_argument('--repository',type=Path,default=ROOT)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--submission',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--training',type=Path)
    parser.add_argument('--smoke-result',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--model', choices=tuple(MODELS), default='llama31')
    args=parser.parse_args()
    sub=args.submission.resolve()
    if args.mode=='prepare':
        if args.audit is None:parser.error('--audit required')
        prepare(args.repository.resolve(),args.audit.resolve(),sub,args.model)
    elif args.output is None:parser.error('--output required')
    elif args.mode=='baseline':baseline(sub,args.output.resolve())
    elif args.mode=='train':train(sub,args.output.resolve(),args.smoke,args.smoke_result,args.baseline)
    else:
        if args.training is None or args.baseline is None:parser.error('--training and --baseline required')
        report(sub,args.training.resolve(),args.baseline.resolve(),args.output.resolve())


if __name__=='__main__':main()
