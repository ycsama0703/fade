"""
Stage 4: Evaluate trained models on the test set, produce metrics tables.

Usage:
    python experiments/04_evaluation.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import summarize_predictions, make_results_table


def recover_factor_type(X: pd.DataFrame) -> pd.Series:
    type_cols = [c for c in X.columns if c.startswith("factor_type_")]
    if not type_cols:
        return pd.Series("other", index=X.index)
    ohe = X[type_cols]
    labels = ohe.idxmax(axis=1).str.replace("factor_type_", "", regex=False)
    labels[ohe.sum(axis=1) == 0] = "other"
    return labels


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    out_dir   = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X_te = pd.read_parquet(data_dir / "X_test.parquet")
    y_te = pd.read_parquet(data_dir / "y_test.parquet")
    durations = y_te["L1_duration"].values
    events    = y_te["L1_event"].astype(int).values

    ic_panel = pd.read_parquet(data_dir / "ic_series_filtered.parquet")
    ic_panel["date"] = pd.to_datetime(ic_panel["date"])

    with open(model_dir / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)

    median_observed = float(np.median(durations[events == 1]))

    results = {}

    # B1 — global mean
    b1 = models["B1_global_mean"]
    preds = b1.predict(X_te)
    results["B1_global_mean"] = summarize_predictions(durations, preds, events)

    # B2 — per-type mean (needs factor_type column)
    b2 = models["B2_type_mean"]
    X_te_typed = X_te.copy()
    X_te_typed["factor_type"] = recover_factor_type(X_te).values
    preds = b2.predict(X_te_typed)
    results["B2_type_mean"] = summarize_predictions(durations, preds, events)

    # B3 — hyperbolic decay fit (needs IC panel)
    b3 = models["B3_hyperbolic"]
    ic_te = ic_panel[ic_panel["factor_id"].isin(X_te.index)]
    b3_preds_series = b3.predict_panel(ic_te)
    # Align to test index; fill missing/inf with median
    b3_preds = b3_preds_series.reindex(X_te.index)
    b3_preds = np.where(np.isfinite(b3_preds), b3_preds, median_observed)
    results["B3_hyperbolic"] = summarize_predictions(durations, b3_preds, events)

    # B4 — XGBoost
    b4 = models["B4_xgboost"]
    preds = b4.predict(X_te)
    results["B4_xgboost"] = summarize_predictions(durations, preds, events)

    # M1 — Cox (predict_median_survival reindexes to trained feature columns automatically)
    cox = models["M1_cox"]
    preds = cox.predict_median_survival(X_te)
    preds = np.where(np.isfinite(preds), preds, median_observed)
    results["M1_cox"] = summarize_predictions(durations, preds, events)

    # M2 — Random Survival Forest
    if "M2_rsf" in models:
        preds = models["M2_rsf"].predict_median_survival(X_te)
        preds = np.where(np.isfinite(preds), preds, median_observed)
        results["M2_rsf"] = summarize_predictions(durations, preds, events)

    # M3 — Gradient Boosting Survival
    if "M3_gbs" in models:
        preds = models["M3_gbs"].predict_median_survival(X_te)
        preds = np.where(np.isfinite(preds), preds, median_observed)
        results["M3_gbs"] = summarize_predictions(durations, preds, events)

    # D1 — DeepSurv
    if "D1_deepsurv" in models:
        preds = models["D1_deepsurv"].predict_median_survival(X_te)
        results["D1_deepsurv"] = summarize_predictions(durations, preds, events)

    # D2 — LSTMSurv
    if "D2_lstmsurv" in models:
        registry_df = pd.read_parquet(data_dir / "X_test.parquet")  # just to get index
        registry_csv = pd.read_csv(Path(cfg["paths"]["data_dir"]) / "factor_registry.csv")
        disc_dates = dict(zip(registry_csv["factor_id"], registry_csv["discovery_date"]))
        preds = models["D2_lstmsurv"].predict_median_survival(
            X_te.index.tolist(), ic_panel, disc_dates
        )
        results["D2_lstmsurv"] = summarize_predictions(durations, preds, events)

    # Print and save results table
    table = make_results_table(results)
    table.to_csv(out_dir / "main_results.csv")
    print("\n=== Test Set Results ===")
    print(table.round(3).to_string())

    # Cox hazard ratios
    hr = cox.hazard_ratios.sort_values(ascending=False)
    hr.to_csv(out_dir / "cox_hazard_ratios.csv")
    print("\n=== Top 10 Cox Hazard Ratios (>1 = faster decay) ===")
    print(hr.head(10).round(3).to_string())

    # XGBoost feature importance
    imp = b4.feature_importances
    imp.to_csv(out_dir / "xgboost_feature_importance.csv")
    print("\n=== Top 10 XGBoost Feature Importances ===")
    print(imp.head(10).round(4).to_string())

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
