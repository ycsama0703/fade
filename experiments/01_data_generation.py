"""
Stage 1: Generate factor pool, compute IC time series, and produce decay labels.

Usage:
    python experiments/01_data_generation.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import yaml

from src.data import (
    FactorTreeGenerator,
    compute_ic_series,
)
from src.data.label_generation import generate_labels_for_panel


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic factor pool
    syn_cfg = cfg["factor_pool"]["synthetic"]
    generator = FactorTreeGenerator(
        operators=syn_cfg["operators_ts"]
        + syn_cfg["operators_cs"]
        + syn_cfg["operators_arith"],
        base_fields=[f"${f}" for f in syn_cfg["base_fields"]],
        lookback_windows=syn_cfg["lookback_windows"],
        max_depth=syn_cfg["max_tree_depth"],
        seed=cfg["seed"],
    )
    pool = generator.generate_pool(syn_cfg["n_candidates"])
    expressions = {f"syn_{i:05d}": expr for i, expr in enumerate(pool)}
    print(f"Generated {len(expressions)} synthetic factor expressions.")

    # Persist expressions for later feature extraction
    with open(data_dir / "factor_expressions.pkl", "wb") as f:
        pickle.dump(expressions, f)

    # 2. Compute IC time series
    expr_strings = {fid: e.to_qlib_str() for fid, e in expressions.items()}
    print("Computing IC series via Qlib (this may take a while)...")
    ic_panel = compute_ic_series(
        factor_expressions=expr_strings,
        market=cfg["data"]["market"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        forward_horizon_days=cfg["backtest"]["forward_return_horizon_days"],
        frequency=cfg["backtest"]["ic_frequency"],
    )
    ic_panel.to_parquet(data_dir / "ic_series.parquet")
    print(f"Saved IC panel: {ic_panel.shape}")

    # 3. Filter weak factors
    initial_window = cfg["factor_pool"]["filter"]["initial_window_months"]
    min_abs_ic = cfg["factor_pool"]["filter"]["min_initial_abs_ic"]

    early_means = (
        ic_panel.sort_values("date")
        .groupby("factor_id")["ic"]
        .apply(lambda s: s.iloc[:initial_window].mean())
    )
    keep_ids = set(early_means[early_means.abs() >= min_abs_ic].index)
    ic_panel = ic_panel[ic_panel["factor_id"].isin(keep_ids)]
    expressions = {fid: e for fid, e in expressions.items() if fid in keep_ids}
    print(f"Kept {len(expressions)} factors after filtering.")

    ic_panel.to_parquet(data_dir / "ic_series_filtered.parquet")
    with open(data_dir / "factor_expressions_filtered.pkl", "wb") as f:
        pickle.dump(expressions, f)

    # 4. Generate labels
    labels = generate_labels_for_panel(ic_panel)
    labels.to_parquet(data_dir / "labels.parquet")
    print(f"Saved labels: {labels.shape}")
    print(labels.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
