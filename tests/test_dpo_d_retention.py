import math

import torch

from cse_dpo.build_stage2_tie_aware_dataset import classify_pair
from cse_dpo.train_factoid_three_stage_dpo import (
    Config,
    _preference_losses,
    _retention_metrics,
    _row_is_tie,
    load_stage_rows,
)


def _dpo_d_loss(delta: float, tie: bool) -> float:
    cfg = Config(objective="dpo_d", beta=1.0, dpo_d_nu=1.0)
    policy_c = torch.tensor([delta])
    policy_r = torch.tensor([0.0])
    refs = torch.zeros((1, 2))
    return float(
        _preference_losses(
            policy_c,
            policy_r,
            refs,
            cfg,
            torch.tensor([float(tie)]),
        )[0]
    )


def test_dpo_d_matches_davidson_probabilities_at_zero_margin() -> None:
    assert math.isclose(_dpo_d_loss(0.0, tie=False), math.log(4.0), rel_tol=1e-6)
    assert math.isclose(_dpo_d_loss(0.0, tie=True), math.log(2.0), rel_tol=1e-6)


def test_dpo_d_tie_is_symmetric_and_win_rewards_positive_margin() -> None:
    assert math.isclose(_dpo_d_loss(-2.0, tie=True), _dpo_d_loss(2.0, tie=True), rel_tol=1e-6)
    assert _dpo_d_loss(2.0, tie=False) < _dpo_d_loss(0.0, tie=False)


def test_explicit_tie_labels_are_read_strictly() -> None:
    cfg = Config(objective="dpo_d")
    assert _row_is_tie({"preference_label": "tie"}, cfg)
    assert not _row_is_tie({"preference_label": "win"}, cfg)


def test_retention_metrics_count_gains_and_losses() -> None:
    baseline = [
        {"question_id": "a", "mrr": 1.0},
        {"question_id": "b", "mrr": 1.0},
        {"question_id": "c", "mrr": 0.0},
    ]
    current = [
        {"question_id": "a", "mrr": 1.0},
        {"question_id": "b", "mrr": 0.0},
        {"question_id": "c", "mrr": 1.0},
    ]
    result = _retention_metrics(baseline, current, loss_penalty=2.0, min_rate=0.75)
    assert result["retained_correct_count"] == 1
    assert result["lost_correct_count"] == 1
    assert result["newly_correct_count"] == 1
    assert result["retention_rate"] == 0.5
    assert result["retention_score"] == -1.0
    assert not result["retention_constraint_met"]


def test_tie_dataset_rule_preserves_punctuation() -> None:
    containment = {"chosen": "[BE]first trimester of pregnancy[EE]", "rejected": "[BE]first trimester[EE]"}
    punctuation_change = {"chosen": "[BE]S-adenosylmethionine[EE]", "rejected": "[BE]S adenosylmethionine[EE]"}
    assert classify_pair(containment) == ("win", "surface_containment")
    assert classify_pair(punctuation_change) == (
        "tie",
        "zero_literal_token_overlap_semantic_equivalence",
    )


def test_tie_aware_dataset_preset_loads_expected_labels() -> None:
    cfg = Config(dataset_preset="legacy_first400_dpo_d_ties", smoke_test=False)
    rows = load_stage_rows(cfg)["format_alignment"]
    labels = [row["preference_label"] for row in rows]
    assert len(rows) == 830
    assert labels.count("win") == 538
    assert labels.count("tie") == 292
