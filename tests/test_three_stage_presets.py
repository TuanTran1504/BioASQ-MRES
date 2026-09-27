from pathlib import Path

from cse_dpo.train_factoid_three_stage_dpo import (
    Config,
    FULL_GOLD_SUPPORTED_STAGED_ROOT,
    SPLIT_SFT80_DPO20_INITIAL_05B,
    SPLIT_SFT80_DPO20_STAGE1,
    SPLIT_SFT80_DPO20_STAGE2,
    SPLIT_SFT80_DPO20_STAGED_ROOT,
    load_stage_rows,
)


def test_split_model_and_dataset_presets_resolve() -> None:
    cfg = Config(
        model_preset="qwen25_05b_sft80_dpo20_r32",
        dataset_preset="sft80_dpo20_r32_step175_dpo226_gpt",
        smoke_test=False,
    )

    assert Path(cfg.initial_adapter) == SPLIT_SFT80_DPO20_INITIAL_05B
    assert Path(cfg.staged_root) == SPLIT_SFT80_DPO20_STAGED_ROOT
    assert SPLIT_SFT80_DPO20_INITIAL_05B.exists()
    assert SPLIT_SFT80_DPO20_STAGE1.exists()
    assert SPLIT_SFT80_DPO20_STAGE2.exists()

    stages = load_stage_rows(cfg)
    assert len(stages["concept_learning"]) == 94
    assert len(stages["format_alignment"]) == 265


def test_full_gold_supported_dataset_preset_resolves() -> None:
    cfg = Config(
        model_preset="qwen25_05b_sft80_dpo20_r32",
        dataset_preset="gold_supported_full_merged",
        smoke_test=False,
    )

    assert Path(cfg.initial_adapter) == SPLIT_SFT80_DPO20_INITIAL_05B
    assert Path(cfg.staged_root) == FULL_GOLD_SUPPORTED_STAGED_ROOT

    stages = load_stage_rows(cfg)
    assert len(stages["concept_learning"]) == 293
    assert len(stages["format_alignment"]) == 1319
