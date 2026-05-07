"""
Stage 1: Build factor pool, compute IC time series, and generate decay labels.

Supports two factor sources (controlled by configs/default.yaml):
  - Alpha158: 158 pre-defined Qlib factors
  - Synthetic: randomly generated expression trees

Usage:
    python experiments/01_data_generation.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    FactorTreeGenerator,
    compute_ic_series,
    load_alpha158_expressions,
)
from src.data.label_generation import generate_labels_for_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_factor_pool(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build merged expression dict from enabled sources.

    Returns
    -------
    exprs    : {factor_id: qlib_expression}
    registry : {factor_id: source}   source in {"alpha158", "synthetic"}
    """
    exprs: dict[str, str] = {}
    registry: dict[str, str] = {}

    if cfg["factor_pool"].get("use_alpha158", False):
        uri = cfg["data"]["qlib_provider_uri"]
        alpha158 = load_alpha158_expressions(provider_uri=uri)
        exprs.update(alpha158)
        registry.update({fid: "alpha158" for fid in alpha158})
        print(f"[pool] Loaded {len(alpha158)} Alpha158 expressions.")

    if cfg["factor_pool"].get("use_synthetic", False):
        syn = cfg["factor_pool"]["synthetic"]
        gen = FactorTreeGenerator(
            operators=(
                syn["operators_ts"] + syn["operators_cs"] + syn["operators_arith"]
            ),
            base_fields=[f"${f}" for f in syn["base_fields"]],
            lookback_windows=syn["lookback_windows"],
            max_depth=syn["max_tree_depth"],
            seed=cfg["seed"],
        )
        pool = gen.generate_pool(syn["n_candidates"])
        syn_exprs = {f"syn_{i:05d}": e.to_qlib_str() for i, e in enumerate(pool)}
        exprs.update(syn_exprs)
        registry.update({fid: "synthetic" for fid in syn_exprs})
        print(f"[pool] Generated {len(syn_exprs)} synthetic expressions.")

    if not exprs:
        raise ValueError(
            "No factor sources enabled. Set use_alpha158 or use_synthetic in config."
        )
    print(f"[pool] Total: {len(exprs)} factors.")
    return exprs, registry


def assign_discovery_dates(
    registry: dict[str, str],
    cfg: dict,
    seed: int = 42,
) -> dict[str, str]:
    """Assign a random discovery date to each synthetic factor."""
    rng = random.Random(seed)
    syn_cfg = cfg["factor_pool"]["synthetic"]
    d_start = pd.Timestamp(syn_cfg.get("discovery_date_start", cfg["data"]["start_date"]))
    d_end   = pd.Timestamp(syn_cfg.get("discovery_date_end",   cfg["data"]["end_date"]))
    span_days = (d_end - d_start).days

    dates: dict[str, str] = {}
    for fid, source in registry.items():
        if source == "alpha158":
            dates[fid] = cfg["data"]["start_date"]
        else:
            offset = rng.randint(0, span_days)
            dates[fid] = (d_start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
    return dates


def compute_ic_in_batches(
    exprs: dict[str, str],
    cfg: dict,
    batch_size: int = 158,
) -> pd.DataFrame:
    """Wrap compute_ic_series() with batching and tqdm progress."""
    keys = list(exprs.keys())
    batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]
    frames: list[pd.DataFrame] = []

    for batch_keys in tqdm(batches, desc="IC batches"):
        batch = {k: exprs[k] for k in batch_keys}
        try:
            ic = compute_ic_series(
                factor_expressions=batch,
                market=cfg["data"]["market"],
                start_date=cfg["data"]["start_date"],
                end_date=cfg["data"]["end_date"],
                forward_horizon_days=cfg["backtest"]["forward_return_horizon_days"],
                frequency=cfg["backtest"]["ic_frequency"],
            )
            frames.append(ic)
        except Exception as e:
            print(f"\n[warn] Batch failed ({list(batch.keys())[0]}…): {e}")

    if not frames:
        raise RuntimeError("All IC batches failed — check Qlib setup.")
    return pd.concat(frames, ignore_index=True)


def filter_weak_factors(
    ic_panel: pd.DataFrame,
    exprs: dict[str, str],
    registry: dict[str, str],
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    initial_window = cfg["factor_pool"]["filter"]["initial_window_months"]
    min_abs_ic    = cfg["factor_pool"]["filter"]["min_initial_abs_ic"]

    early_means = (
        ic_panel.sort_values("date")
        .groupby("factor_id")["ic"]
        .apply(lambda s: s.iloc[:initial_window].mean())
    )
    keep = set(early_means[early_means.abs() >= min_abs_ic].index)
    n_before = len(exprs)

    ic_panel = ic_panel[ic_panel["factor_id"].isin(keep)].copy()
    exprs    = {k: v for k, v in exprs.items()    if k in keep}
    registry = {k: v for k, v in registry.items() if k in keep}
    print(f"[filter] Kept {len(exprs)}/{n_before} factors (|IC₀| ≥ {min_abs_ic}).")
    return ic_panel, exprs, registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build factor pool
    exprs, registry = build_factor_pool(cfg)

    # Use a single batch for Alpha158 to minimise redundant data loads.
    alpha158_n = sum(1 for s in registry.values() if s == "alpha158")
    batch_size = max(alpha158_n, 50)

    # 2. Compute IC time series
    print(f"\n[IC] Computing IC for {len(exprs)} factors on {cfg['data']['market']} "
          f"({cfg['data']['start_date']} → {cfg['data']['end_date']})...")
    ic_panel = compute_ic_in_batches(exprs, cfg, batch_size=batch_size)
    print(f"[IC] Raw panel shape: {ic_panel.shape}")

    # 3. Filter weak factors
    ic_panel, exprs, registry = filter_weak_factors(ic_panel, exprs, registry, cfg)

    # 4. Save artefacts
    ic_panel.to_parquet(data_dir / "ic_series_filtered.parquet", index=False)
    print(f"[save] ic_series_filtered.parquet  {ic_panel.shape}")

    disc_dates = assign_discovery_dates(registry, cfg, seed=cfg["seed"])
    registry_df = pd.DataFrame([
        {"factor_id": fid, "source": src,
         "expression": exprs[fid], "discovery_date": disc_dates[fid]}
        for fid, src in registry.items()
    ])
    registry_df.to_csv(data_dir / "factor_registry.csv", index=False)
    print(f"[save] factor_registry.csv  ({len(registry_df)} rows)")

    # Pickle placeholder for synthetic AST objects (Alpha158 has no AST).
    with open(data_dir / "factor_expressions_filtered.pkl", "wb") as f:
        pickle.dump({fid: None for fid in registry}, f)

    # 5. Generate labels (slice IC from discovery date for each factor)
    print("\n[labels] Computing decay labels...")
    labels = generate_labels_for_panel(ic_panel, discovery_dates=disc_dates)
    labels.to_parquet(data_dir / "labels.parquet")
    print(f"[save] labels.parquet  {labels.shape}")

    # 6. Summary
    print("\n=== Label summary ===")
    print(labels[["L1_duration", "L1_event", "L3_slope"]].describe().round(3))
    event_rate = labels["L1_event"].mean()
    print(f"\nDecay event rate (L1_event=1): {event_rate:.1%}")
    if not 0.2 <= event_rate <= 0.9:
        print("[warn] Event rate outside 20–90%. Consider adjusting label thresholds.")

    print("\n=== IC summary by source ===")
    ic_with_src = ic_panel.merge(registry_df[["factor_id", "source"]], on="factor_id")
    print(ic_with_src.groupby("source")["ic"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
