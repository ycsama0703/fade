"""
IC (Information Coefficient) time series computation via Qlib backtest engine.

Returns a long-format DataFrame: factor_id | t (month) | ic
"""

from __future__ import annotations

import pandas as pd
from typing import Iterable


def compute_ic_series(
    factor_expressions: dict[str, str],
    market: str = "csi500",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    forward_horizon_days: int = 20,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Compute IC time series for each factor expression.

    Parameters
    ----------
    factor_expressions : dict
        Mapping of factor_id -> Qlib expression string.
    market : str
        Qlib stockpool identifier.
    start_date, end_date : str
    forward_horizon_days : int
        Forward return horizon used as the IC target.
    frequency : {'daily', 'monthly'}
        Granularity of the output IC series.

    Returns
    -------
    DataFrame with columns: factor_id, date, ic
    """
    try:
        import qlib
        from qlib.data import D
        from qlib.constant import REG_CN
    except ImportError as e:
        raise ImportError(
            "qlib is required. Install with `pip install pyqlib` "
            "and run `python -m qlib.run.get_data qlib_data`."
        ) from e

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    instruments = D.instruments(market=market)

    # Forward return label expression: future {h}-day return.
    # Qlib sign convention: positive offset = look BACK, negative = look FORWARD.
    # So Ref($close, -h) / $close - 1 gives the h-day forward return.
    label_expr = f"Ref($close, -{forward_horizon_days}) / $close - 1"

    fields = list(factor_expressions.values()) + [label_expr]
    field_names = list(factor_expressions.keys()) + ["__label__"]

    panel = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date,
    )
    panel.columns = field_names

    # Compute cross-sectional Spearman IC for each date.
    ic_records: list[dict] = []
    for date, group in panel.groupby(level="datetime"):
        labels = group["__label__"]
        if labels.notna().sum() < 30:
            continue
        for fid in factor_expressions.keys():
            scores = group[fid]
            mask = scores.notna() & labels.notna()
            if mask.sum() < 30:
                continue
            ic = scores[mask].rank().corr(labels[mask].rank())
            ic_records.append({"factor_id": fid, "date": date, "ic": ic})

    df = pd.DataFrame.from_records(ic_records)

    if frequency == "monthly":
        df = (
            df.set_index("date")
              .groupby("factor_id")
              .resample("ME")["ic"].mean()
              .reset_index()
        )
    return df


def discovery_dates(
    factor_ids: Iterable[str],
    default_date: str = "2010-01-01",
) -> dict[str, str]:
    """Stub: return discovery dates for each factor.

    For synthetic factors, all share the same nominal discovery date.
    For academic factors, look up from a curated table.
    """
    return {fid: default_date for fid in factor_ids}
