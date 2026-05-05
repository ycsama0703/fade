"""
Decay label generation from IC time series.

Three label flavors:
    L1: half-life (months until rolling IC drops below 0.5 * initial IC)
    L2: binary survival at horizon N
    L3: continuous IC slope (linear regression)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_half_life_label(
    ic_series: pd.Series,
    initial_window_months: int = 6,
    threshold_ratio: float = 0.5,
    persist_months: int = 3,
) -> tuple[float, int]:
    """
    Compute half-life of an IC series.

    Returns
    -------
    duration : float
        Number of months from t0 to confirmed decay event.
        For right-censored cases, equals the full observation length.
    event : int
        1 if decay was observed (IC dropped below threshold for `persist_months`),
        0 if right-censored.
    """
    s = ic_series.dropna()
    if len(s) < initial_window_months + persist_months:
        return float(len(s)), 0

    ic_0 = s.iloc[:initial_window_months].mean()
    if abs(ic_0) < 1e-6:
        return float(len(s)), 0

    threshold = threshold_ratio * ic_0
    rolling = s.rolling(window=persist_months, min_periods=persist_months).mean()

    # Sign-aware: if ic_0 > 0, decay = rolling falls below threshold.
    if ic_0 > 0:
        decayed = rolling < threshold
    else:
        decayed = rolling > threshold

    decayed_idx = np.where(decayed.values)[0]
    if len(decayed_idx) == 0:
        return float(len(s)), 0

    return float(decayed_idx[0]), 1


def compute_binary_label(
    ic_series: pd.Series,
    horizon_months: int = 24,
    initial_window_months: int = 6,
    threshold_ratio: float = 0.5,
    persist_months: int = 3,
) -> int:
    """Did the factor decay within `horizon_months`? (1 = decayed, 0 = survived)"""
    duration, event = compute_half_life_label(
        ic_series, initial_window_months, threshold_ratio, persist_months
    )
    if event == 1 and duration <= horizon_months:
        return 1
    return 0


def compute_decay_slope_label(ic_series: pd.Series) -> float:
    """OLS slope of IC ~ t. More negative = faster decay."""
    s = ic_series.dropna()
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1)
    return float(slope)


def generate_labels_for_panel(ic_panel: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generate all three labels for each factor.

    ic_panel : DataFrame with columns [factor_id, date, ic]
    """
    rows = []
    for fid, group in ic_panel.groupby("factor_id"):
        ic = group.sort_values("date")["ic"]
        duration, event = compute_half_life_label(ic)
        rows.append({
            "factor_id": fid,
            "L1_duration": duration,
            "L1_event": event,
            "L2_decayed_24m": compute_binary_label(ic, horizon_months=24),
            "L2_decayed_36m": compute_binary_label(ic, horizon_months=36),
            "L3_slope": compute_decay_slope_label(ic),
        })
    return pd.DataFrame(rows)
