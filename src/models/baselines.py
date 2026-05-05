"""Naive and theoretical baselines for half-life prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class GlobalMeanBaseline:
    """Predicts the global training-set mean half-life for every factor."""

    def __init__(self):
        self.mean_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, event: pd.Series | None = None):
        self.mean_ = float(y[event == 1].mean()) if event is not None else float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.mean_ is not None
        return np.full(len(X), self.mean_)


class TypeMeanBaseline:
    """Predicts the per-factor-type training mean half-life."""

    def __init__(self, type_column: str = "factor_type"):
        self.type_column = type_column
        self.type_means_: dict[str, float] = {}
        self.global_mean_: float = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series, event: pd.Series | None = None):
        df = X.copy()
        df["__y__"] = y
        if event is not None:
            df = df[event == 1]
        self.type_means_ = df.groupby(self.type_column)["__y__"].mean().to_dict()
        self.global_mean_ = float(df["__y__"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.type_means_.get(t, self.global_mean_)
            for t in X[self.type_column]
        ])


class HyperbolicDecayFit:
    """
    Per-factor hyperbolic decay model: alpha(t) = K / (1 + lambda * t).

    Fits (K, lambda) on the early IC observations of each factor, then derives
    predicted half-life as the time t_half where alpha(t_half) = 0.5 * K, i.e.
    t_half = 1 / lambda.

    This replicates the theoretical framework from arXiv:2512.11913 (retracted).
    """

    @staticmethod
    def _hyper(t, K, lam):
        return K / (1.0 + lam * t)

    def fit_predict_for_series(self, ic_series: pd.Series) -> float:
        s = ic_series.dropna()
        if len(s) < 4:
            return np.nan
        t = np.arange(len(s))
        try:
            popt, _ = curve_fit(
                self._hyper, t, s.values,
                p0=[s.iloc[0], 0.05],
                maxfev=5000,
            )
            _, lam = popt
            if lam <= 0:
                return np.inf  # not decaying within the model
            return float(1.0 / lam)
        except Exception:
            return np.nan

    def predict_panel(self, ic_panel: pd.DataFrame) -> pd.Series:
        out = {}
        for fid, group in ic_panel.groupby("factor_id"):
            s = group.sort_values("date")["ic"]
            out[fid] = self.fit_predict_for_series(s)
        return pd.Series(out, name="predicted_half_life")
