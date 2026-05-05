"""
Group C features: market environment at the factor's discovery time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _market_volatility(market_returns: pd.Series, end_date, window_days: int = 60) -> float:
    s = market_returns.loc[:end_date].iloc[-window_days:]
    return s.std() * np.sqrt(252) if len(s) >= 20 else np.nan


def _market_regime(market_returns: pd.Series, end_date, window_days: int = 120) -> str:
    """Crude three-state regime: bull / bear / sideways based on trailing return."""
    s = market_returns.loc[:end_date].iloc[-window_days:]
    if len(s) < 20:
        return "unknown"
    cum = (1 + s).prod() - 1
    if cum > 0.10:
        return "bull"
    if cum < -0.10:
        return "bear"
    return "sideways"


def _novelty_score(
    new_factor_ic: pd.Series,
    library_ic_panel: pd.DataFrame,
) -> float:
    """1 - max correlation between the new factor's IC series and library factors."""
    if library_ic_panel.empty or len(new_factor_ic.dropna()) < 6:
        return 1.0
    corrs = []
    for fid, group in library_ic_panel.groupby("factor_id"):
        other = group.set_index("date")["ic"]
        joined = pd.concat([new_factor_ic, other], axis=1, join="inner").dropna()
        if len(joined) >= 6:
            c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            corrs.append(abs(c) if not np.isnan(c) else 0.0)
    return 1.0 - max(corrs) if corrs else 1.0


def extract_market_env_features(
    discovery_date,
    market_returns: pd.Series,
    factor_library_ic: pd.DataFrame | None = None,
    new_factor_ic: pd.Series | None = None,
    same_type_recent_decay: float | None = None,
) -> dict:
    return {
        "market_vol_60d": _market_volatility(market_returns, discovery_date, 60),
        "market_regime": _market_regime(market_returns, discovery_date),
        "factor_lib_size": (
            factor_library_ic["factor_id"].nunique()
            if factor_library_ic is not None and not factor_library_ic.empty
            else 0
        ),
        "novelty_score": (
            _novelty_score(new_factor_ic, factor_library_ic)
            if factor_library_ic is not None and new_factor_ic is not None
            else np.nan
        ),
        "same_type_recent_decay": same_type_recent_decay,
    }
