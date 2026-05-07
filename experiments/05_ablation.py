"""
Stage 5: Ablation study — feature group contribution analysis.

Tests Cox (M1) and XGBoost (B4) with each feature group alone and in
all combinations:
    A   structural (expression string features + factor_type)
    B   early IC dynamics (K = 3 / 6 / 12 months)
    C   market environment at discovery time
    A+B, A+C, B+C, A+B+C (full)

Usage:
    python experiments/05_ablation.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CoxSurvivalModel, XGBoostRegressor
from src.evaluation.metrics import summarize_predictions, make_results_table


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

GROUP_A = [
    "tree_depth", "node_count", "ts_op_ratio", "cs_op_ratio",
    "max_lookback", "op_diversity", "n_distinct_fields", "uses_volume",
]
GROUP_A_OHE_PREFIX = "factor_type_"   # one-hot columns also belong to A

GROUP_B_PREFIX = "ic_"               # all early-IC columns

GROUP_C = [
    "market_vol_60d", "factor_lib_size", "novelty_score",
]
GROUP_C_OHE_PREFIX = "market_regime_"


def select_groups(X: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    cols = []
    if "A" in groups:
        cols += GROUP_A
        cols += [c for c in X.columns if c.startswith(GROUP_A_OHE_PREFIX)]
    if "B" in groups:
        cols += [c for c in X.columns if c.startswith(GROUP_B_PREFIX)]
    if "C" in groups:
        cols += [c for c in GROUP_C if c in X.columns]
        cols += [c for c in X.columns if c.startswith(GROUP_C_OHE_PREFIX)]
    # keep only columns that exist and deduplicate
    cols = [c for c in dict.fromkeys(cols) if c in X.columns]
    return X[cols]


def drop_constant(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop zero-variance columns; return pruned DataFrame and dropped list."""
    zero_var = X.columns[X.var() < 1e-10].tolist()
    return X.drop(columns=zero_var), zero_var


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_cox(X_tr, dur_tr, evt_tr, penalizer: float) -> CoxSurvivalModel:
    X_tr_p, dropped = drop_constant(X_tr)
    model = CoxSurvivalModel(penalizer=penalizer).fit(X_tr_p, dur_tr, evt_tr)
    model._ablation_dropped = dropped
    return model


def predict_cox(model: CoxSurvivalModel, X: pd.DataFrame, fallback: float) -> np.ndarray:
    X_p = X.drop(columns=model._ablation_dropped, errors="ignore")
    preds = model.predict_median_survival(X_p)
    return np.where(np.isfinite(preds), preds, fallback)


def fit_xgb(X_tr, dur_tr, evt_tr, cfg) -> XGBoostRegressor:
    xgb_cfg = cfg["models"]["xgboost"]
    return XGBoostRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        random_state=cfg["seed"],
    ).fit(X_tr, dur_tr, evt_tr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir  = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr = pd.read_parquet(data_dir / "X_train.parquet")
    X_te = pd.read_parquet(data_dir / "X_test.parquet")
    y_tr = pd.read_parquet(data_dir / "y_train.parquet")
    y_te = pd.read_parquet(data_dir / "y_test.parquet")

    dur_tr = y_tr["L1_duration"]
    evt_tr = y_tr["L1_event"].astype(int)
    dur_te = y_te["L1_duration"].values
    evt_te = y_te["L1_event"].astype(int).values
    fallback = float(np.median(dur_te[evt_te == 1]))

    penalizer = cfg["models"]["cox"]["penalizer"]

    # All non-empty group subsets: A, B, C, A+B, A+C, B+C, A+B+C
    all_groups = ["A", "B", "C"]
    subsets = []
    for r in range(1, len(all_groups) + 1):
        for combo in combinations(all_groups, r):
            subsets.append(list(combo))

    cox_results = {}
    xgb_results = {}

    for groups in subsets:
        label = "+".join(groups)
        print(f"\n--- Groups: {label} ---")

        X_tr_sub = select_groups(X_tr, groups)
        X_te_sub = select_groups(X_te, groups)

        if X_tr_sub.shape[1] == 0:
            print("  No features — skipping.")
            continue

        n_feat = X_tr_sub.shape[1]

        # Cox
        try:
            cox = fit_cox(X_tr_sub, dur_tr, evt_tr, penalizer)
            preds = predict_cox(cox, X_te_sub, fallback)
            cox_results[f"Cox [{label}] (n={n_feat})"] = summarize_predictions(dur_te, preds, evt_te)
            ci = cox_results[f"Cox [{label}] (n={n_feat})"]["C_index"]
            print(f"  Cox  C-index={ci:.3f}  n_feat={n_feat}")
        except Exception as e:
            print(f"  Cox  FAILED: {e}")

        # XGBoost
        try:
            xgb = fit_xgb(X_tr_sub, dur_tr, evt_tr, cfg)
            preds = xgb.predict(X_te_sub)
            xgb_results[f"XGB [{label}] (n={n_feat})"] = summarize_predictions(dur_te, preds, evt_te)
            ci = xgb_results[f"XGB [{label}] (n={n_feat})"]["C_index"]
            print(f"  XGB  C-index={ci:.3f}  n_feat={n_feat}")
        except Exception as e:
            print(f"  XGB  FAILED: {e}")

    print("\n\n=== Ablation Results: Cox ===")
    cox_table = make_results_table(cox_results)
    print(cox_table[["C_index", "Spearman_rho", "MAE_observed"]].round(3).to_string())
    cox_table.to_csv(out_dir / "ablation_cox.csv")

    print("\n=== Ablation Results: XGBoost ===")
    xgb_table = make_results_table(xgb_results)
    print(xgb_table[["C_index", "Spearman_rho", "MAE_observed"]].round(3).to_string())
    xgb_table.to_csv(out_dir / "ablation_xgboost.csv")

    print(f"\nSaved to {out_dir}/ablation_cox.csv  and  ablation_xgboost.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
