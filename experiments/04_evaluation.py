"""
Stage 4: Evaluate trained models on the test set, produce metrics tables and plots.

Usage:
    python experiments/04_evaluation.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import summarize_predictions, make_results_table


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X_te = pd.read_parquet(data_dir / "X_test.parquet")
    y_te = pd.read_parquet(data_dir / "y_test.parquet")
    durations = y_te["L1_duration"].values
    events = y_te["L1_event"].astype(int).values

    with open(model_dir / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)

    results = {}
    for name, model in models.items():
        if hasattr(model, "predict_median_survival"):
            preds = model.predict_median_survival(X_te)
        else:
            preds = model.predict(X_te)
        # Replace inf / nan with median observed duration as a fallback
        preds = np.where(
            np.isfinite(preds), preds, np.median(durations[events == 1])
        )
        results[name] = summarize_predictions(durations, preds, events)

    table = make_results_table(results)
    table.to_csv(out_dir / "main_results.csv")
    print("Main results:\n", table.round(3))

    # Cox-specific: print top hazard ratios
    cox = models.get("M1_cox")
    if cox is not None:
        hr = cox.hazard_ratios.sort_values(ascending=False)
        hr.to_csv(out_dir / "cox_hazard_ratios.csv")
        print("\nTop 10 hazard ratios (Cox):\n", hr.head(10))

    # XGBoost feature importance
    xgb = models.get("B4_xgboost")
    if xgb is not None:
        imp = xgb.feature_importances
        imp.to_csv(out_dir / "xgboost_feature_importance.csv")
        print("\nTop 10 XGBoost feature importances:\n", imp.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
