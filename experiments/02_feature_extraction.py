"""
Stage 2: Extract structural / early-IC / market-environment features.

Usage:
    python experiments/02_feature_extraction.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.early_ic import extract_early_ic_features_batch


# ---------------------------------------------------------------------------
# Group A: structural features from Qlib expression strings
# (Alpha158 has no AST objects — we parse strings directly)
# ---------------------------------------------------------------------------

# Known operator sets in Qlib expression syntax
TS_OPS = {
    "Mean", "Std", "Max", "Min", "Rank", "Corr", "Cov",
    "Delta", "Ref", "Sum", "Skew", "Kurt", "WMA", "EMA",
}
CS_OPS = {"rank", "zscore", "scale", "Greater", "Less", "Abs", "Log", "Sign"}
ARITH_OPS = {"Add", "Sub", "Mul", "Div", "Neg", "Power", "If"}


def _extract_operators(expr: str) -> list[str]:
    """Return all operator names found in an expression string."""
    return re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*\(", expr)


def _max_lookback(expr: str) -> int:
    """Extract the largest integer window from an expression (e.g. Mean($x, 20) → 20)."""
    nums = re.findall(r",\s*(\d+)\s*[,\)]", expr)
    return max((int(n) for n in nums), default=0)


def _tree_depth(expr: str) -> int:
    """Estimate expression tree depth by max nesting of parentheses."""
    depth = max_depth = 0
    for c in expr:
        if c == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ")":
            depth -= 1
    return max_depth


def _classify_factor_type(expr: str) -> str:
    """Heuristic factor-type label from expression string."""
    e = expr.lower()
    if "volume" in e and ("mean" in e or "std" in e):
        return "volume"
    if "ref" in e and re.search(r",\s*(1|2|3|4|5)\s*[,\)]", e):
        return "reversal"
    if "ref" in e and re.search(r",\s*(10|15|20|30)\s*[,\)]", e):
        return "momentum"
    if "std" in e or "skew" in e or "kurt" in e:
        return "volatility"
    if "corr" in e or "cov" in e:
        return "correlation"
    return "other"


def extract_structural_from_string(expr: str) -> dict:
    ops = _extract_operators(expr)
    n_ops = max(len(ops), 1)
    ts_count = sum(o in TS_OPS for o in ops)
    cs_count = sum(o in CS_OPS for o in ops)
    return {
        "tree_depth":      _tree_depth(expr),
        "node_count":      len(ops) + expr.count("$"),  # ops + leaf fields
        "ts_op_ratio":     ts_count / n_ops,
        "cs_op_ratio":     cs_count / n_ops,
        "max_lookback":    _max_lookback(expr),
        "op_diversity":    len(set(ops)) / n_ops,
        "n_distinct_fields": len(set(re.findall(r"\$\w+", expr))),
        "uses_volume":     int("volume" in expr.lower()),
        "factor_type":     _classify_factor_type(expr),
    }


def extract_structural_features_from_registry(
    registry_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in registry_df.iterrows():
        feats = extract_structural_from_string(row["expression"])
        feats["factor_id"] = row["factor_id"]
        rows.append(feats)
    return pd.DataFrame(rows).set_index("factor_id")


# ---------------------------------------------------------------------------
# Group C: market environment at discovery time
# ---------------------------------------------------------------------------

def load_market_returns(
    provider_uri: str,
    market: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Pull market-level daily returns from Qlib (equal-weighted CSI500)."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    instruments = D.instruments(market=market)
    panel = D.features(
        instruments,
        ["$close / Ref($close, 1) - 1"],
        start_time=start_date,
        end_time=end_date,
    )
    panel.columns = ["ret"]
    # Equal-weight cross-section mean as market return proxy
    mkt = panel.groupby(level="datetime")["ret"].mean()
    mkt.index = pd.to_datetime(mkt.index)
    return mkt


def extract_market_env_for_factor(
    discovery_date: str,
    mkt_returns: pd.Series,
    factor_ic: pd.Series,
    all_ic_panel: pd.DataFrame,
    vol_window: int = 60,
) -> dict:
    disc = pd.Timestamp(discovery_date)

    # 60-day realised volatility: prefer lookback, fall back to lookahead
    # (for factors discovered at data-start there is no prior history)
    pre  = mkt_returns.loc[:disc].iloc[-vol_window:]
    post = mkt_returns.loc[disc:].iloc[:vol_window]
    window = pre if len(pre) >= 20 else post
    mkt_vol = float(window.std() * np.sqrt(252)) if len(window) >= 20 else np.nan

    # Market regime: trailing 120-day return; fall back to forward window
    trail = mkt_returns.loc[:disc].iloc[-120:]
    if len(trail) < 20:
        trail = mkt_returns.loc[disc:].iloc[:120]
    cum = float((1 + trail).prod() - 1) if len(trail) >= 20 else 0.0
    if cum > 0.10:
        regime = "bull"
    elif cum < -0.10:
        regime = "bear"
    else:
        regime = "sideways"

    # Factor library size at discovery (number of already-known factors)
    known = all_ic_panel[all_ic_panel["date"] <= disc]["factor_id"].nunique()

    # Novelty: 1 - max absolute correlation with other factors' early IC
    early = factor_ic.dropna().iloc[:6]
    if len(early) >= 4:
        corrs = []
        for fid2, grp in all_ic_panel.groupby("factor_id"):
            other = grp.sort_values("date")["ic"].iloc[:6].dropna()
            joined = pd.concat([early, other], axis=1, join="inner").dropna()
            if len(joined) >= 4:
                c = abs(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
                if not np.isnan(c):
                    corrs.append(c)
        novelty = 1.0 - max(corrs) if corrs else 1.0
    else:
        novelty = np.nan

    return {
        "market_vol_60d":   mkt_vol,
        "market_regime":    regime,
        "factor_lib_size":  known,
        "novelty_score":    novelty,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])

    registry_df = pd.read_csv(data_dir / "factor_registry.csv")
    ic_panel    = pd.read_parquet(data_dir / "ic_series_filtered.parquet")
    ic_panel["date"] = pd.to_datetime(ic_panel["date"])

    # ------------------------------------------------------------------
    # Group A: structural features
    # ------------------------------------------------------------------
    print("Extracting Group A: structural features...")
    feat_A = extract_structural_features_from_registry(registry_df)
    print(f"  Group A shape: {feat_A.shape}")

    # ------------------------------------------------------------------
    # Group B: early IC dynamics (sliced from discovery date)
    # ------------------------------------------------------------------
    print("Extracting Group B: early IC dynamics (K = 3, 6, 12) ...")
    disc_dates = dict(zip(registry_df["factor_id"], registry_df["discovery_date"])) \
        if "discovery_date" in registry_df.columns else None
    feat_B = extract_early_ic_features_batch(
        ic_panel,
        K_values=cfg["features"]["early_ic_windows"],
        discovery_dates=disc_dates,
    )
    print(f"  Group B shape: {feat_B.shape}")

    # ------------------------------------------------------------------
    # Group C: market environment at discovery time
    # ------------------------------------------------------------------
    print("Extracting Group C: market environment...")
    print("  Loading market returns from Qlib...")
    mkt_returns = load_market_returns(
        provider_uri=cfg["data"]["qlib_provider_uri"],
        market=cfg["data"]["market"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
    )

    # For Alpha158, all share the same nominal discovery date (data start).
    # Novelty computation across 141 factors is expensive — use a progress bar.
    from tqdm import tqdm

    feat_C_rows = []
    for _, row in tqdm(registry_df.iterrows(), total=len(registry_df),
                       desc="  Market env"):
        fid = row["factor_id"]
        disc_date = row.get("discovery_date", cfg["data"]["start_date"])
        factor_ic = (
            ic_panel[ic_panel["factor_id"] == fid]
            .sort_values("date")
            .set_index("date")["ic"]
        )
        feats = extract_market_env_for_factor(
            discovery_date=disc_date,
            mkt_returns=mkt_returns,
            factor_ic=factor_ic,
            all_ic_panel=ic_panel,
        )
        feats["factor_id"] = fid
        feat_C_rows.append(feats)

    feat_C = pd.DataFrame(feat_C_rows).set_index("factor_id")
    # One-hot encode market_regime
    feat_C = pd.get_dummies(feat_C, columns=["market_regime"], drop_first=False)
    print(f"  Group C shape: {feat_C.shape}")

    # ------------------------------------------------------------------
    # Concatenate and save
    # ------------------------------------------------------------------
    X = pd.concat([feat_A, feat_B, feat_C], axis=1)

    # One-hot encode any remaining string columns (factor_type)
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=False)

    X.to_parquet(data_dir / "feature_matrix.parquet")
    print(f"\nSaved feature_matrix.parquet: {X.shape}")
    print(f"  Features: {list(X.columns)}")

    # Quick NaN report
    nan_frac = X.isna().mean().sort_values(ascending=False)
    if nan_frac.max() > 0:
        print("\nNaN fraction per feature (top 10):")
        print(nan_frac[nan_frac > 0].head(10).round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
