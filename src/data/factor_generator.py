"""
Synthetic factor expression generator.

Generates random expression trees over a typed operator library, with
depth and lookback constraints. Output expressions follow Qlib's
expression syntax so they can be plugged into Qlib's backtest engine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional

NodeType = Literal["ts", "cs", "scalar", "field"]


@dataclass
class OperatorSpec:
    name: str
    arity: int
    output_type: NodeType
    input_types: list[NodeType]
    needs_window: bool = False


# Minimal operator library — extend as needed.
OPERATORS: dict[str, OperatorSpec] = {
    "ts_mean": OperatorSpec("ts_mean", 1, "ts", ["ts"], needs_window=True),
    "ts_std":  OperatorSpec("ts_std",  1, "ts", ["ts"], needs_window=True),
    "ts_max":  OperatorSpec("ts_max",  1, "ts", ["ts"], needs_window=True),
    "ts_min":  OperatorSpec("ts_min",  1, "ts", ["ts"], needs_window=True),
    "ts_rank": OperatorSpec("ts_rank", 1, "ts", ["ts"], needs_window=True),
    "delta":   OperatorSpec("delta",   1, "ts", ["ts"], needs_window=True),
    "rank":    OperatorSpec("rank",    1, "cs", ["ts"]),
    "zscore":  OperatorSpec("zscore",  1, "cs", ["ts"]),
    "add":     OperatorSpec("add",     2, "ts", ["ts", "ts"]),
    "sub":     OperatorSpec("sub",     2, "ts", ["ts", "ts"]),
    "mul":     OperatorSpec("mul",     2, "ts", ["ts", "ts"]),
    "div":     OperatorSpec("div",     2, "ts", ["ts", "ts"]),
    "log":     OperatorSpec("log",     1, "ts", ["ts"]),
    "abs":     OperatorSpec("abs",     1, "ts", ["ts"]),
}

BASE_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]


@dataclass
class FactorExpression:
    """An expression tree node."""
    op: Optional[str]                  # None for leaf
    children: list["FactorExpression"] = field(default_factory=list)
    field_name: Optional[str] = None   # for leaf
    window: Optional[int] = None       # for time-series ops

    def to_qlib_str(self) -> str:
        """Render to Qlib expression syntax, e.g. Mean($close, 20)."""
        if self.op is None:
            return self.field_name or ""
        args = [c.to_qlib_str() for c in self.children]
        if self.window is not None:
            args.append(str(self.window))
        # Qlib uses CamelCase, e.g. Mean, Std, Rank — adjust mapping if needed.
        return f"{self.op}({', '.join(args)})"

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def collect_fields(self) -> set[str]:
        if self.op is None and self.field_name:
            return {self.field_name}
        out: set[str] = set()
        for c in self.children:
            out |= c.collect_fields()
        return out


class FactorTreeGenerator:
    """Random typed expression tree generator with depth and lookback control."""

    def __init__(
        self,
        operators: list[str] | None = None,
        base_fields: list[str] | None = None,
        lookback_windows: list[int] | None = None,
        max_depth: int = 4,
        seed: int | None = None,
    ):
        self.operators = operators or list(OPERATORS.keys())
        self.base_fields = base_fields or BASE_FIELDS
        self.lookback_windows = lookback_windows or [5, 10, 20, 60]
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def generate(self, target_depth: int | None = None) -> FactorExpression:
        depth = target_depth or self.rng.randint(2, self.max_depth)
        return self._build(depth, output_type="ts")

    def _build(self, depth_remaining: int, output_type: NodeType) -> FactorExpression:
        if depth_remaining <= 1:
            return self._leaf()

        candidates = [
            name for name in self.operators
            if OPERATORS[name].output_type in (output_type, "ts")
        ]
        if not candidates:
            return self._leaf()

        op_name = self.rng.choice(candidates)
        spec = OPERATORS[op_name]
        children = [
            self._build(depth_remaining - 1, t)
            for t in spec.input_types
        ]
        window = self.rng.choice(self.lookback_windows) if spec.needs_window else None
        return FactorExpression(op=op_name, children=children, window=window)

    def _leaf(self) -> FactorExpression:
        return FactorExpression(op=None, field_name=self.rng.choice(self.base_fields))

    def generate_pool(self, n: int, dedupe: bool = True) -> list[FactorExpression]:
        seen: set[str] = set()
        pool: list[FactorExpression] = []
        attempts = 0
        max_attempts = n * 10
        while len(pool) < n and attempts < max_attempts:
            expr = self.generate()
            key = expr.to_qlib_str()
            if not dedupe or key not in seen:
                seen.add(key)
                pool.append(expr)
            attempts += 1
        return pool
