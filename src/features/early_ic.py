"""
Group B features: early IC dynamics in the first K months after discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_early_ic_features(ic_series: pd.Series, K: int = 6) -> dict:
    """
    Compute IC dynamics summary using only the first K observations.
    """
    s = ic_series.dropna().iloc[:K]
    if len(s) < max(3, K // 2):
        return {f"ic_mean_{K}m": np.nan}

    ic_mean = s.mean()
    ic_std = s.std()
    icir = ic_mean / ic_std if ic_std > 1e-9 else np.nan

    # Linear trend within the window
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1) if len(s) >= 2 else (np.nan, np.nan)

    # Lag-1 autocorrelation
    autocorr = s.autocorr(lag=1) if len(s) >= 3 else np.nan

    # Drawdown of cumulative IC
    cum = s.cumsum()
    drawdown = (cum.cummax() - cum).max()

    # Sign consistency
    monotone_pos = (s > 0).mean()

    return {
        f"ic_mean_{K}m": ic_mean,
        f"ic_std_{K}m": ic_std,
        f"icir_{K}m": icir,
        f"ic_trend_{K}m": slope,
        f"ic_autocorr_{K}m": autocorr,
        f"ic_drawdown_{K}m": drawdown,
        f"ic_pos_ratio_{K}m": monotone_pos,
    }


def extract_early_ic_features_batch(
    ic_panel: pd.DataFrame,
    K_values: list[int] = (3, 6, 12),
) -> pd.DataFrame:
    """
    ic_panel : DataFrame with columns [factor_id, date, ic]
    """
    rows = []
    for fid, group in ic_panel.groupby("factor_id"):
        s = group.sort_values("date")["ic"]
        feats: dict = {"factor_id": fid}
        for K in K_values:
            feats.update(extract_early_ic_features(s, K=K))
        rows.append(feats)
    return pd.DataFrame(rows).set_index("factor_id")
