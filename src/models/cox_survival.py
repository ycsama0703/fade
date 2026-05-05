"""
Cox Proportional Hazards model for factor decay prediction.

Wraps `lifelines.CoxPHFitter` with a clean fit/predict interface and adds
a half-life point predictor (median survival time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CoxSurvivalModel:
    def __init__(self, penalizer: float = 0.01, l1_ratio: float = 0.0):
        from lifelines import CoxPHFitter
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.feature_columns_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        durations: pd.Series,
        events: pd.Series,
    ):
        df = X.copy()
        # One-hot encode any object columns (e.g. factor_type, market_regime)
        df = pd.get_dummies(df, drop_first=True)
        df["__duration__"] = durations.values
        df["__event__"] = events.values
        self.feature_columns_ = [
            c for c in df.columns if c not in ("__duration__", "__event__")
        ]
        self.model.fit(df, duration_col="__duration__", event_col="__event__")
        return self

    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.get_dummies(X.copy(), drop_first=True)
        df = df.reindex(columns=self.feature_columns_, fill_value=0)
        median = self.model.predict_median(df)
        return median.values

    def predict_survival_function(self, X: pd.DataFrame, times=None) -> pd.DataFrame:
        df = pd.get_dummies(X.copy(), drop_first=True)
        df = df.reindex(columns=self.feature_columns_, fill_value=0)
        return self.model.predict_survival_function(df, times=times)

    def concordance_index(self, X, durations, events) -> float:
        df = pd.get_dummies(X.copy(), drop_first=True)
        df = df.reindex(columns=self.feature_columns_, fill_value=0)
        df["__duration__"] = durations.values
        df["__event__"] = events.values
        return self.model.score(df, scoring_method="concordance_index")

    @property
    def hazard_ratios(self) -> pd.Series:
        return np.exp(self.model.params_)
