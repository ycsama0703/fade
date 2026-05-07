"""
Decay label generation from IC time series.

Three label flavors:
    L1: half-life  — months until rolling IC sign-flips and persists
    L2: binary survival at horizon N
    L3: continuous IC slope (linear regression)

Decay definition (L1):
    A factor is declared "decayed" when its `persist_months`-rolling mean IC
    crosses zero (relative to the sign of IC_0) and stays there.
    This is more robust than a relative threshold (50% of IC_0) for near-zero IC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_half_life_label(
    ic_series: pd.Series,
    initial_window_months: int = 6,
    persist_months: int = 3,
    min_abs_ic0: float = 0.005,
) -> tuple[float, int]:
    """
    Compute IC half-life via sign-flip criterion.

    A factor is "decayed" when its rolling `persist_months`-mean IC crosses
    zero (sign is opposite to IC_0) for the first time.

    Parameters
    ----------
    ic_series : pd.Series
        Monthly IC values in chronological order.
    initial_window_months : int
        Months used to estimate IC_0 (the initial IC level).
    persist_months : int
        Rolling window length; decay requires the rolling mean to flip sign.
    min_abs_ic0 : float
        If |IC_0| < min_abs_ic0 the factor is considered noise and
        treated as right-censored (event = 0).

    Returns
    -------
    duration : float
        Months from start to first confirmed decay. Equals series length
        for right-censored cases.
    event : int
        1 = decay observed, 0 = right-censored.
    """
    s = ic_series.dropna()
    n = len(s)
    min_len = initial_window_months + persist_months
    if n < min_len:
        return float(n), 0

    ic_0 = s.iloc[:initial_window_months].mean()

    # Treat near-zero initial IC as noise → censored
    if abs(ic_0) < min_abs_ic0:
        return float(n), 0

    rolling = s.rolling(window=persist_months, min_periods=persist_months).mean()

    # Decay = rolling IC crosses zero (in the opposite direction to ic_0)
    if ic_0 > 0:
        decayed = rolling < 0   # positive factor turns negative
    else:
        decayed = rolling > 0   # negative factor turns positive (also decay)

    decayed_idx = np.where(decayed.values)[0]
    if len(decayed_idx) == 0:
        return float(n), 0

    return float(decayed_idx[0]), 1


def compute_binary_label(
    ic_series: pd.Series,
    horizon_months: int = 24,
    **kwargs,
) -> int:
    """Did the factor decay within `horizon_months`? (1 = decayed, 0 = survived)"""
    duration, event = compute_half_life_label(ic_series, **kwargs)
    return int(event == 1 and duration <= horizon_months)


def compute_decay_slope_label(ic_series: pd.Series) -> float:
    """OLS slope of IC ~ t. More negative = faster decay."""
    s = ic_series.dropna()
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1)
    return float(slope)


def generate_labels_for_panel(
    ic_panel: pd.DataFrame,
    discovery_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Generate all three labels for each factor in the panel.

    Parameters
    ----------
    ic_panel : DataFrame with columns [factor_id, date, ic]
    discovery_dates : optional dict {factor_id: "YYYY-MM-DD"}
        If provided, IC series is sliced to start from the discovery date,
        so features and labels are computed in-sample from that point.
    """
    rows = []
    for fid, group in ic_panel.groupby("factor_id"):
        grp = group.sort_values("date")
        if discovery_dates and fid in discovery_dates:
            disc = pd.Timestamp(discovery_dates[fid])
            grp = grp[grp["date"] >= disc]
        ic = grp["ic"]
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
