"""
Load Qlib Alpha158 factor expressions without running a full backtest.

Requires qlib.init() to have been called before use (Alpha158 instantiation
needs the Qlib data backend even though we don't load market data here).
"""

from __future__ import annotations

import qlib
from qlib.constant import REG_CN


def load_alpha158_expressions(
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
) -> dict[str, str]:
    """
    Extract the 158 factor (name, expression) pairs from Qlib's Alpha158 handler.

    Parameters
    ----------
    provider_uri : str
        Path to Qlib data directory (needed to instantiate the handler).

    Returns
    -------
    dict
        Mapping of factor_id (e.g. "alpha158_KMID") -> Qlib expression string.
    """
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.contrib.data.handler import Alpha158

    # Instantiate with a minimal date range just to access the config.
    h = Alpha158(
        instruments="csi500",
        start_time="2020-01-01",
        end_time="2020-01-31",
        fit_start_time="2020-01-01",
        fit_end_time="2020-01-31",
    )
    fields, names = h.get_feature_config()

    if len(fields) != len(names):
        raise ValueError(
            f"Alpha158 field/name count mismatch: {len(fields)} vs {len(names)}"
        )

    return {f"alpha158_{name}": expr for name, expr in zip(names, fields)}
