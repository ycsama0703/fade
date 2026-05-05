"""Quick sanity tests for feature extractors and label generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.factor_generator import FactorTreeGenerator, FactorExpression
from src.data.label_generation import (
    compute_half_life_label,
    compute_decay_slope_label,
)
from src.features.structural import extract_structural_features
from src.features.early_ic import extract_early_ic_features


def test_factor_generator_produces_valid_expressions():
    gen = FactorTreeGenerator(seed=0, max_depth=3)
    pool = gen.generate_pool(n=20)
    assert len(pool) == 20
    for expr in pool:
        assert isinstance(expr, FactorExpression)
        assert expr.depth() <= 3
        s = expr.to_qlib_str()
        assert len(s) > 0


def test_structural_features_keys():
    gen = FactorTreeGenerator(seed=1, max_depth=4)
    expr = gen.generate()
    feats = extract_structural_features(expr)
    expected_keys = {
        "tree_depth", "node_count", "ts_op_ratio", "cs_op_ratio",
        "max_lookback", "op_diversity", "factor_type",
    }
    assert expected_keys.issubset(feats.keys())


def test_half_life_no_decay_is_censored():
    # Constant high IC: should be right-censored
    s = pd.Series([0.05] * 24)
    duration, event = compute_half_life_label(s)
    assert event == 0
    assert duration == 24


def test_half_life_clear_decay_is_observed():
    # Drop from 0.05 to 0.01 after month 6 — should detect decay
    ic = [0.05] * 6 + [0.01] * 18
    s = pd.Series(ic)
    duration, event = compute_half_life_label(s)
    assert event == 1
    assert 6 <= duration <= 12


def test_decay_slope_negative_for_declining_series():
    s = pd.Series(np.linspace(0.05, 0.0, 24))
    slope = compute_decay_slope_label(s)
    assert slope < 0


def test_early_ic_features_handle_short_series():
    s = pd.Series([0.04, 0.03])  # too short
    feats = extract_early_ic_features(s, K=6)
    # Should return mostly NaN, not crash
    assert "ic_mean_6m" in feats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
