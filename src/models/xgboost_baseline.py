"""XGBoost regression baseline for half-life prediction (no censoring handling)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class XGBoostRegressor:
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            tree_method="hist",
        )
        self.feature_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, event: pd.Series | None = None):
        # Without survival framework, drop censored samples or treat them as observed.
        df = pd.get_dummies(X.copy(), drop_first=True)
        if event is not None:
            mask = event == 1
            df = df[mask.values]
            y = y[mask.values]
        self.feature_columns_ = list(df.columns)
        self.model.fit(df, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.get_dummies(X.copy(), drop_first=True)
        df = df.reindex(columns=self.feature_columns_, fill_value=0)
        return self.model.predict(df)

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns_,
        ).sort_values(ascending=False)
