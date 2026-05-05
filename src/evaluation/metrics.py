"""Evaluation metrics for half-life prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def mean_absolute_error_observed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event: np.ndarray,
) -> float:
    """MAE computed only over uncensored (observed) samples."""
    mask = event == 1
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])))


def spearman_rank_corr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event: np.ndarray | None = None,
) -> float:
    """Spearman correlation, optionally restricted to observed cases."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if event is not None:
        mask = event == 1
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) < 3:
        return np.nan
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def concordance_index(
    durations: np.ndarray,
    predicted_scores: np.ndarray,
    events: np.ndarray,
) -> float:
    """Harrell's C-index. Higher = better."""
    from lifelines.utils import concordance_index as _ci
    # In lifelines, predictions should rank lower for higher risk; for half-life
    # we predict survival time, so a higher predicted half-life = lower risk.
    return float(_ci(durations, predicted_scores, events))


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event: np.ndarray,
) -> dict:
    return {
        "MAE_observed":   mean_absolute_error_observed(y_true, y_pred, event),
        "Spearman_rho":   spearman_rank_corr(y_true, y_pred, event),
        "C_index":        concordance_index(y_true, y_pred, event),
        "n_observed":     int((event == 1).sum()),
        "n_censored":     int((event == 0).sum()),
    }


def make_results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Pivot a dict of {model_name: metrics_dict} into a tidy DataFrame."""
    return pd.DataFrame(results).T
