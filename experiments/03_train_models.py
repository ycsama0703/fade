"""
Stage 3: Train all models on the training split, save checkpoints.

Usage:
    python experiments/03_train_models.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import yaml

from src.models import (
    GlobalMeanBaseline,
    TypeMeanBaseline,
    HyperbolicDecayFit,
    CoxSurvivalModel,
    XGBoostRegressor,
)


def split_by_discovery_date(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stub: split by factor discovery date.

    For synthetic factors, all share the same date — replace with academic-factor
    publication dates or a synthetic timeline before running for real.
    """
    # Placeholder: random split for now.
    rng = pd.np.random.default_rng(cfg["seed"]) if hasattr(pd, "np") else None
    n = len(X)
    perm = list(range(n))
    import random
    random.Random(cfg["seed"]).shuffle(perm)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return (
        X.iloc[train_idx], labels.iloc[train_idx],
        X.iloc[val_idx],   labels.iloc[val_idx],
        X.iloc[test_idx],  labels.iloc[test_idx],
    )


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(data_dir / "feature_matrix.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet").set_index("factor_id")

    # Align indices
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    labels = labels.loc[common]

    # Split
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_by_discovery_date(X, labels, cfg)

    # Save splits for evaluation stage
    for name, df in [
        ("X_train", X_tr), ("y_train", y_tr),
        ("X_val", X_va),   ("y_val", y_va),
        ("X_test", X_te),  ("y_test", y_te),
    ]:
        df.to_parquet(data_dir / f"{name}.parquet")

    durations_tr = y_tr["L1_duration"]
    events_tr = y_tr["L1_event"].astype(int)

    # Train models
    models = {}

    print("Training B1: global mean baseline...")
    m = GlobalMeanBaseline().fit(X_tr, durations_tr, events_tr)
    models["B1_global_mean"] = m

    print("Training B2: per-type mean baseline...")
    m = TypeMeanBaseline().fit(X_tr, durations_tr, events_tr)
    models["B2_type_mean"] = m

    print("Skipping B3 (hyperbolic fit) — runs at predict-time on IC series.")

    print("Training B4: XGBoost regression...")
    xgb_cfg = cfg["models"]["xgboost"]
    m = XGBoostRegressor(**xgb_cfg).fit(X_tr, durations_tr, events_tr)
    models["B4_xgboost"] = m

    print("Training M1: Cox proportional hazards...")
    cox_cfg = cfg["models"]["cox"]
    m = CoxSurvivalModel(**cox_cfg).fit(X_tr, durations_tr, events_tr)
    models["M1_cox"] = m

    # Save all
    with open(model_dir / "trained_models.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"Saved models to {model_dir / 'trained_models.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
