"""
Group A features: structural properties of the factor expression tree.

Operates on FactorExpression AST objects.
"""

from __future__ import annotations

import pandas as pd
from ..data.factor_generator import FactorExpression, OPERATORS


TS_OPS = {name for name, spec in OPERATORS.items() if spec.output_type == "ts" and spec.needs_window}
CS_OPS = {name for name, spec in OPERATORS.items() if spec.output_type == "cs"}


def _walk_ops(expr: FactorExpression) -> list[str]:
    out = []
    if expr.op is not None:
        out.append(expr.op)
    for c in expr.children:
        out.extend(_walk_ops(c))
    return out


def _max_lookback(expr: FactorExpression) -> int:
    own = expr.window or 0
    child_max = max((_max_lookback(c) for c in expr.children), default=0)
    return max(own, child_max)


def _classify_factor_type(expr: FactorExpression) -> str:
    """Heuristic factor-type tagging based on operator and field composition."""
    ops = set(_walk_ops(expr))
    fields = expr.collect_fields()
    has_volume = any("volume" in f for f in fields)
    max_lb = _max_lookback(expr)

    if "delta" in ops and max_lb <= 5:
        return "reversal"
    if "delta" in ops and max_lb >= 20:
        return "momentum"
    if "ts_std" in ops:
        return "volatility"
    if has_volume:
        return "volume"
    if "ts_corr" in ops:
        return "correlation"
    return "other"


def extract_structural_features(expr: FactorExpression) -> dict:
    ops = _walk_ops(expr)
    n_nodes = expr.node_count()
    if n_nodes == 0:
        return {}

    return {
        "tree_depth": expr.depth(),
        "node_count": n_nodes,
        "ts_op_ratio": sum(o in TS_OPS for o in ops) / max(len(ops), 1),
        "cs_op_ratio": sum(o in CS_OPS for o in ops) / max(len(ops), 1),
        "max_lookback": _max_lookback(expr),
        "op_diversity": len(set(ops)) / max(len(ops), 1),
        "n_distinct_fields": len(expr.collect_fields()),
        "uses_volume": int(any("volume" in f for f in expr.collect_fields())),
        "uses_price_only": int(all(
            "volume" not in f for f in expr.collect_fields()
        )),
        "factor_type": _classify_factor_type(expr),
    }


def extract_structural_features_batch(
    expressions: dict[str, FactorExpression],
) -> pd.DataFrame:
    rows = []
    for fid, expr in expressions.items():
        feats = extract_structural_features(expr)
        feats["factor_id"] = fid
        rows.append(feats)
    return pd.DataFrame(rows).set_index("factor_id")
