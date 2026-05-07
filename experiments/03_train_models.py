"""
Stage 3: Train all models and save checkpoints.

Models
------
B1  GlobalMeanBaseline      — predict training-set mean half-life for all
B2  TypeMeanBaseline        — predict per-factor-type mean half-life
B3  HyperbolicDecayFit      — fit K/(1+λt) curve to early IC series
B4  XGBoostRegressor        — gradient-boosted regression on feature matrix
M1  CoxSurvivalModel        — Cox proportional hazards (main model)

Usage:
    python experiments/03_train_models.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

# Prevent OpenMP conflict between PyTorch and XGBoost on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    GlobalMeanBaseline,
    TypeMeanBaseline,
    HyperbolicDecayFit,
    CoxSurvivalModel,
    XGBoostRegressor,
    RandomSurvivalForest,
    GradientBoostingSurvival,
    DeepSurv,
    LSTMSurv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recover_factor_type(X: pd.DataFrame) -> pd.Series:
    """Reconstruct factor_type string from one-hot columns."""
    type_cols = [c for c in X.columns if c.startswith("factor_type_")]
    if not type_cols:
        return pd.Series("other", index=X.index)
    # argmax over one-hot → original label
    ohe = X[type_cols]
    labels = ohe.idxmax(axis=1).str.replace("factor_type_", "", regex=False)
    # rows with all-zero one-hot (no type matched) → "other"
    all_zero = (ohe.sum(axis=1) == 0)
    labels[all_zero] = "other"
    return labels


def random_split(X, labels, cfg):
    """Random 60/20/20 split (fallback when all factors share the same discovery date)."""
    seed = cfg["seed"]
    idx = X.index.tolist()
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.40, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=seed)
    return (
        X.loc[idx_tr], labels.loc[idx_tr],
        X.loc[idx_va], labels.loc[idx_va],
        X.loc[idx_te], labels.loc[idx_te],
    )


def temporal_split(X, labels, registry_path: Path):
    """
    Temporal split by discovery date: train < 2015, val 2015–2016, test ≥ 2017.
    Falls back to random split if discovery_date column is absent.
    """
    reg = pd.read_csv(registry_path).set_index("factor_id")
    if "discovery_date" not in reg.columns:
        return None

    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    dates = reg["discovery_date"].reindex(X.index)

    idx_tr = X.index[dates < "2015-01-01"].tolist()
    idx_va = X.index[(dates >= "2015-01-01") & (dates < "2017-01-01")].tolist()
    idx_te = X.index[dates >= "2017-01-01"].tolist()

    if min(len(idx_tr), len(idx_va), len(idx_te)) == 0:
        return None  # degenerate — caller should fall back to random

    print(f"Temporal split: train={len(idx_tr)} | val={len(idx_va)} | test={len(idx_te)}")
    return (
        X.loc[idx_tr], labels.loc[idx_tr],
        X.loc[idx_va], labels.loc[idx_va],
        X.loc[idx_te], labels.loc[idx_te],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X      = pd.read_parquet(data_dir / "feature_matrix.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet").set_index("factor_id")
    common = X.index.intersection(labels.index)
    X, labels = X.loc[common], labels.loc[common]

    print(f"Dataset: {len(X)} factors × {X.shape[1]} features")

    # Split — prefer temporal (by discovery date), fall back to random
    split = temporal_split(X, labels, data_dir / "factor_registry.csv")
    if split is None:
        print("No discovery dates found — using random 60/20/20 split.")
        split = random_split(X, labels, cfg)
    X_tr, y_tr, X_va, y_va, X_te, y_te = split
    print(f"Split: train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")

    # Save splits for stage 4
    for name, df in [
        ("X_train", X_tr), ("y_train", y_tr),
        ("X_val",   X_va), ("y_val",   y_va),
        ("X_test",  X_te), ("y_test",  y_te),
    ]:
        df.to_parquet(data_dir / f"{name}.parquet")

    dur_tr  = y_tr["L1_duration"]
    evt_tr  = y_tr["L1_event"].astype(int)
    type_tr = recover_factor_type(X_tr)

    models = {}

    # B1 — global mean
    print("\nTraining B1: global mean baseline...")
    models["B1_global_mean"] = GlobalMeanBaseline().fit(X_tr, dur_tr, evt_tr)
    pred = models["B1_global_mean"].predict(X_tr)
    print(f"  Mean predicted half-life: {pred[0]:.1f} months")

    # B2 — per-type mean
    print("Training B2: per-type mean baseline...")
    X_tr_typed = X_tr.copy()
    X_tr_typed["factor_type"] = type_tr.values
    models["B2_type_mean"] = TypeMeanBaseline().fit(X_tr_typed, dur_tr, evt_tr)
    print(f"  Type means: {models['B2_type_mean'].type_means_}")

    # B3 — hyperbolic decay fit (stateless, fitted at predict time in Stage 4)
    models["B3_hyperbolic"] = HyperbolicDecayFit()
    print("B3: HyperbolicDecayFit registered (predictions computed in Stage 4)")

    # B4 — XGBoost regression
    print("Training B4: XGBoost regression...")
    xgb_cfg = cfg["models"]["xgboost"]
    models["B4_xgboost"] = XGBoostRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        random_state=cfg["seed"],
    ).fit(X_tr, dur_tr, evt_tr)
    preds_tr = models["B4_xgboost"].predict(X_tr)
    print(f"  Train predictions: mean={preds_tr.mean():.1f}m, std={preds_tr.std():.1f}m")

    # M1 — Cox proportional hazards
    print("Training M1: Cox proportional hazards...")
    cox_cfg = cfg["models"]["cox"]
    # Drop zero-variance columns — they make the Hessian singular
    zero_var = X_tr.columns[X_tr.var() < 1e-10].tolist()
    if zero_var:
        print(f"  Dropping {len(zero_var)} constant columns: {zero_var}")
    X_tr_cox = X_tr.drop(columns=zero_var)
    X_va_cox = X_va.drop(columns=zero_var)
    X_te_cox = X_te.drop(columns=zero_var)
    models["M1_cox"] = CoxSurvivalModel(**cox_cfg).fit(X_tr_cox, dur_tr, evt_tr)
    models["M1_cox"].dropped_cols_ = zero_var  # remember for predict time
    c_idx = models["M1_cox"].concordance_index(X_tr_cox, dur_tr, evt_tr)
    print(f"  Train C-index: {c_idx:.4f}")
    print("  Top hazard ratios (>1 = faster decay):")
    hr = models["M1_cox"].hazard_ratios.sort_values(ascending=False)
    print(hr.head(8).round(3).to_string())

    # Save cox-pruned splits so stage 4 can evaluate M1 consistently
    for name, df in [("X_train_cox", X_tr_cox), ("X_val_cox", X_va_cox), ("X_test_cox", X_te_cox)]:
        df.to_parquet(data_dir / f"{name}.parquet")

    # M2 — Random Survival Forest
    print("Training M2: Random Survival Forest...")
    models["M2_rsf"] = RandomSurvivalForest(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=cfg["seed"],
        n_jobs=1,
    ).fit(X_tr, dur_tr, evt_tr)
    c_idx = models["M2_rsf"].concordance_index(X_tr, dur_tr, evt_tr)
    print(f"  Train C-index: {c_idx:.4f}")

    # M3 — Gradient Boosting Survival
    print("Training M3: Gradient Boosting Survival...")
    models["M3_gbs"] = GradientBoostingSurvival(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=cfg["seed"],
    ).fit(X_tr, dur_tr, evt_tr)
    c_idx = models["M3_gbs"].concordance_index(X_tr, dur_tr, evt_tr)
    print(f"  Train C-index: {c_idx:.4f}")
    print("  Top GBS feature importances:")
    print(models["M3_gbs"].feature_importances.head(6).round(4).to_string())

    # D1 — DeepSurv
    print("Training D1: DeepSurv (MLP + Cox loss)...")
    models["D1_deepsurv"] = DeepSurv(
        hidden_dims=[64, 32],
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-3,
        epochs=200,
        patience=25,
        random_state=cfg["seed"],
    ).fit(X_tr, dur_tr, evt_tr, X_val=X_va, dur_val=y_va["L1_duration"], evt_val=y_va["L1_event"].astype(int))
    c_idx = models["D1_deepsurv"].concordance_index(X_tr, dur_tr, evt_tr)
    print(f"  Train C-index: {c_idx:.4f}")

    # D2 — LSTMSurv
    print("Training D2: LSTMSurv (LSTM on IC series)...")
    ic_panel = pd.read_parquet(data_dir / "ic_series_filtered.parquet")
    ic_panel["date"] = pd.to_datetime(ic_panel["date"])
    registry_df = pd.read_csv(data_dir / "factor_registry.csv")
    disc_dates = dict(zip(registry_df["factor_id"], registry_df["discovery_date"]))

    models["D2_lstmsurv"] = LSTMSurv(
        seq_len=12,
        hidden_dim=32,
        num_layers=2,
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-3,
        epochs=300,
        patience=30,
        random_state=cfg["seed"],
    ).fit(
        factor_ids_tr=X_tr.index.tolist(),
        ic_panel=ic_panel,
        discovery_dates=disc_dates,
        durations=dur_tr,
        events=evt_tr,
        factor_ids_val=X_va.index.tolist(),
        dur_val=y_va["L1_duration"],
        evt_val=y_va["L1_event"].astype(int),
    )
    c_idx = models["D2_lstmsurv"].concordance_index(
        X_tr.index.tolist(), ic_panel, disc_dates, dur_tr, evt_tr
    )
    print(f"  Train C-index: {c_idx:.4f}")

    # Save
    with open(model_dir / "trained_models.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"\nSaved {len(models)} models → {model_dir / 'trained_models.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
