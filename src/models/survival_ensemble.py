"""
Ensemble survival models wrapping scikit-survival.

M2  RandomSurvivalForest  — non-parametric, handles non-linear interactions
M3  GradientBoostingSurvival — boosting with censoring-aware loss
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_structured(durations: pd.Series, events: pd.Series) -> np.ndarray:
    """Convert to scikit-survival structured array (event: bool, duration: float)."""
    return np.array(
        [(bool(e), float(d)) for e, d in zip(events, durations)],
        dtype=[("event", bool), ("duration", float)],
    )


def _prep_X(X: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    df = pd.get_dummies(X.copy(), drop_first=True).astype(float)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df.values.astype(float)


class RandomSurvivalForest:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        from sksurv.ensemble import RandomSurvivalForest as _RSF
        self.model = _RSF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, durations: pd.Series, events: pd.Series):
        df = pd.get_dummies(X.copy(), drop_first=True).astype(float)
        self.feature_columns_ = list(df.columns)
        y = _to_structured(durations, events)
        self.model.fit(df.values, y)
        return self

    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        """Return median survival time for each sample."""
        Xp = _prep_X(X, self.feature_columns_)
        surv_funcs = self.model.predict_survival_function(Xp)
        medians = []
        for fn in surv_funcs:
            # Find first time where S(t) <= 0.5
            idx = np.searchsorted(-fn.y, -0.5)
            if idx < len(fn.x):
                medians.append(float(fn.x[idx]))
            else:
                medians.append(float(fn.x[-1]))  # never crosses 0.5 → last time
        return np.array(medians)

    def concordance_index(self, X, durations, events) -> float:
        from sksurv.metrics import concordance_index_censored
        Xp = _prep_X(X, self.feature_columns_)
        risk = self.model.predict(Xp)  # higher = higher risk
        y = _to_structured(durations, events)
        ci, _, _, _, _ = concordance_index_censored(y["event"], y["duration"], risk)
        return float(ci)


class GradientBoostingSurvival:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis as _GBS
        self.model = _GBS(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
        )
        self.feature_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, durations: pd.Series, events: pd.Series):
        df = pd.get_dummies(X.copy(), drop_first=True).astype(float)
        self.feature_columns_ = list(df.columns)
        y = _to_structured(durations, events)
        self.model.fit(df.values, y)
        return self

    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _prep_X(X, self.feature_columns_)
        surv_funcs = self.model.predict_survival_function(Xp)
        medians = []
        for fn in surv_funcs:
            idx = np.searchsorted(-fn.y, -0.5)
            if idx < len(fn.x):
                medians.append(float(fn.x[idx]))
            else:
                medians.append(float(fn.x[-1]))
        return np.array(medians)

    def concordance_index(self, X, durations, events) -> float:
        from sksurv.metrics import concordance_index_censored
        Xp = _prep_X(X, self.feature_columns_)
        risk = self.model.predict(Xp)
        y = _to_structured(durations, events)
        ci, _, _, _, _ = concordance_index_censored(y["event"], y["duration"], risk)
        return float(ci)

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns_,
        ).sort_values(ascending=False)
