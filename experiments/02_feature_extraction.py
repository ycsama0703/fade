"""
Stage 2: Extract structural / early-IC / market-environment features.

Usage:
    python experiments/02_feature_extraction.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import yaml

from src.features.structural import extract_structural_features_batch
from src.features.early_ic import extract_early_ic_features_batch
from src.features.market_env import extract_market_env_features


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])

    # Load inputs
    with open(data_dir / "factor_expressions_filtered.pkl", "rb") as f:
        expressions = pickle.load(f)
    ic_panel = pd.read_parquet(data_dir / "ic_series_filtered.parquet")

    # Group A: structural
    print("Extracting structural features...")
    feat_A = extract_structural_features_batch(expressions)

    # Group B: early IC
    print("Extracting early IC features for K = 3, 6, 12 ...")
    feat_B = extract_early_ic_features_batch(
        ic_panel, K_values=cfg["features"]["early_ic_windows"]
    )

    # Group C: market environment
    # NOTE: market_returns must be loaded separately (e.g., from Qlib).
    # Here we leave a placeholder; populate before running.
    print("Extracting market environment features...")
    feat_C_rows = []
    for fid in expressions.keys():
        feats = {"factor_id": fid}
        feats.update(
            extract_market_env_features(
                discovery_date=cfg["data"]["start_date"],
                market_returns=pd.Series(dtype=float),  # placeholder
                factor_library_ic=None,
                new_factor_ic=None,
            )
        )
        feat_C_rows.append(feats)
    feat_C = pd.DataFrame(feat_C_rows).set_index("factor_id")

    # Concatenate
    X = pd.concat([feat_A, feat_B, feat_C], axis=1)
    X.to_parquet(data_dir / "feature_matrix.parquet")
    print(f"Saved feature matrix: {X.shape}")
    print("Columns:")
    for c in X.columns:
        print(f"  {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
