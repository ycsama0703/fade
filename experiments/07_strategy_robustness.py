"""
Stage 7: Robustness test — decay-aware filtering across multiple base strategies.

For each base strategy, we show two variants:
  - Without decay filter (vanilla)
  - With decay filter (ours): apply GBS hazard prediction to prune
    factors predicted to decay within `min_remaining` months

Base strategies
---------------
S0   Equal-weight all active factors
S1   IC-momentum Top-10
S2   IC-momentum Top-20
S3   IC-momentum Top-30
S4   IC-momentum Top-50
S5   Random selection (30 factors, averaged over 10 seeds)
S6   High initial IC (IC₀ > 0.02, equal-weight survivors)

Metrics: CR, AR, SR, MDD
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Reuse helpers from Stage 6
# ---------------------------------------------------------------------------

def months_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def compute_metrics(monthly_returns: pd.Series) -> dict:
    r = monthly_returns.dropna()
    if len(r) == 0:
        return dict(CR=np.nan, AR=np.nan, SR=np.nan, MDD=np.nan)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd)


def active_factors_at(t, registry, labels):
    active = []
    lab_idx = labels.set_index("factor_id")
    for _, row in registry.iterrows():
        fid  = row["factor_id"]
        disc = pd.Timestamp(row["discovery_date"])
        if disc > t:
            continue
        elapsed  = months_between(disc, t)
        duration = lab_idx.loc[fid, "L1_duration"]
        event    = lab_idx.loc[fid, "L1_event"]
        if event == 1 and elapsed >= duration:
            continue
        active.append(fid)
    return active


def sign_adjust(ic_t, signs):
    s = pd.Series({fid: signs.get(fid, 1.0) for fid in ic_t.index})
    return ic_t * s


def factor_signs(registry, ic_panel, initial_window=6):
    signs = {}
    for _, row in registry.iterrows():
        fid  = row["factor_id"]
        disc = pd.Timestamp(row["discovery_date"])
        grp  = ic_panel[(ic_panel["factor_id"] == fid) &
                        (ic_panel["date"] >= disc)].sort_values("date")
        early = grp["ic"].iloc[:initial_window].mean()
        signs[fid] = 1.0 if early >= 0 else -1.0
    return signs


def high_ic0_factors(registry, ic_panel, threshold=0.02, window=6):
    """Factors whose early absolute mean IC ≥ threshold."""
    qualified = set()
    for _, row in registry.iterrows():
        fid  = row["factor_id"]
        disc = pd.Timestamp(row["discovery_date"])
        grp  = ic_panel[(ic_panel["factor_id"] == fid) &
                        (ic_panel["date"] >= disc)].sort_values("date")
        early_ic = grp["ic"].iloc[:window].mean()
        if abs(early_ic) >= threshold:
            qualified.add(fid)
    return qualified


def survival_weights(
    candidates: list[str],
    feature_matrix: pd.DataFrame,
    model,
    registry: pd.DataFrame,
    t: pd.Timestamp,
) -> pd.Series:
    """
    Soft survival weight for each candidate:
      w = sigmoid(remaining_months / scale)
    so factors with more remaining life get weight closer to 1,
    while near-death factors get weight close to 0.
    Preserves all factors (no hard removal), just down-weights risky ones.
    """
    X_cand = feature_matrix.loc[feature_matrix.index.isin(candidates)]
    if X_cand.empty:
        return pd.Series(1.0, index=candidates)

    preds   = model.predict_median_survival(X_cand)
    pred_s  = pd.Series(preds, index=X_cand.index)
    reg_idx = registry.set_index("factor_id")

    remaining = {}
    for fid in X_cand.index:
        disc    = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        elapsed = months_between(disc, t)
        remaining[fid] = float(pred_s[fid]) - elapsed

    rem_s = pd.Series(remaining)
    # sigmoid centred at 0, scale=6: 50% weight at 0 remaining, ~90% at +12m
    scale = 6.0
    w = 1.0 / (1.0 + np.exp(-rem_s / scale))
    return w


def apply_decay_filter(
    candidates: list[str],
    feature_matrix: pd.DataFrame,
    model,
    registry: pd.DataFrame,
    t: pd.Timestamp,
    min_remaining: float = 6.0,
    fallback_n: int = 5,
) -> list[str]:
    """Hard filter: keep only candidates predicted to survive ≥ min_remaining months."""
    X_cand = feature_matrix.loc[feature_matrix.index.isin(candidates)]
    if X_cand.empty:
        return candidates

    preds  = model.predict_median_survival(X_cand)
    pred_s = pd.Series(preds, index=X_cand.index)

    reg_idx = registry.set_index("factor_id")
    surviving = []
    for fid in X_cand.index:
        disc    = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        elapsed = months_between(disc, t)
        if float(pred_s[fid]) - elapsed >= min_remaining:
            surviving.append(fid)

    return surviving if surviving else candidates[:fallback_n]


def composite_ic(ic_t: pd.Series, candidates: list[str],
                 weights: pd.Series | None = None) -> float:
    common = ic_t.index.intersection(candidates)
    if common.empty:
        return np.nan
    sub = ic_t[common]
    if weights is not None:
        w = weights.reindex(common).fillna(0.5)
        if w.sum() == 0:
            return float(sub.mean())
        return float((sub * w).sum() / w.sum())
    return float(sub.mean())


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_all_strategies(
    sim_dates, ic_panel, registry, labels,
    feature_matrix, model, signs,
    high_ic0_set, cs_vol=0.10,
    random_seeds=range(10),
):
    results = {s: [] for s in [
        "S0_base", "S0_decay",
        "S1_base", "S1_decay",   # top-10
        "S2_base", "S2_decay",   # top-20
        "S3_base", "S3_decay",   # top-30
        "S4_base", "S4_decay",   # top-50
        "S5_base", "S5_decay",   # random-30
        "S6_base", "S6_decay",   # high IC0
    ]}

    for t in sim_dates:
        active = active_factors_at(t, registry, labels)
        if len(active) < 10:
            continue

        ic_t_raw = (
            ic_panel[(ic_panel["factor_id"].isin(active)) & (ic_panel["date"] == t)]
            .set_index("factor_id")["ic"]
        )
        if ic_t_raw.empty:
            continue
        ic_t = sign_adjust(ic_t_raw, signs)

        # --- IC-momentum helper ---
        def ic_mom_top(k):
            cutoff = t - pd.DateOffset(months=3)
            recent = ic_panel[
                (ic_panel["factor_id"].isin(active)) &
                (ic_panel["date"] > cutoff) & (ic_panel["date"] <= t)
            ]
            mean_ic = (
                recent.groupby("factor_id")["ic"]
                .apply(lambda s: (s * s.map(signs)).mean())
                .abs()
            )
            return mean_ic.nlargest(min(k, len(mean_ic))).index.tolist()

        row = {"date": t}

        # S0: equal-weight all
        cands_s0 = active
        row["S0_base"]  = composite_ic(ic_t, cands_s0)
        row["S0_decay"] = composite_ic(ic_t, apply_decay_filter(
            cands_s0, feature_matrix, model, registry, t))

        # S1–S4: IC-momentum top-K
        for sid, k in [("S1", 10), ("S2", 20), ("S3", 30), ("S4", 50)]:
            cands = ic_mom_top(k)
            row[f"{sid}_base"]  = composite_ic(ic_t, cands)
            row[f"{sid}_decay"] = composite_ic(ic_t, apply_decay_filter(
                cands, feature_matrix, model, registry, t))

        # S5: random-30, averaged over seeds
        ic_rand_base, ic_rand_decay = [], []
        for seed in random_seeds:
            rng   = random.Random(seed)
            cands = rng.sample(active, min(30, len(active)))
            ic_rand_base.append(composite_ic(ic_t, cands))
            filtered = apply_decay_filter(cands, feature_matrix, model, registry, t)
            ic_rand_decay.append(composite_ic(ic_t, filtered))
        row["S5_base"]  = float(np.nanmean(ic_rand_base))
        row["S5_decay"] = float(np.nanmean(ic_rand_decay))

        # S6: high initial IC factors
        cands_s6 = [f for f in active if f in high_ic0_set]
        if len(cands_s6) < 3:
            cands_s6 = active[:10]
        row["S6_base"]  = composite_ic(ic_t, cands_s6)
        row["S6_decay"] = composite_ic(ic_t, apply_decay_filter(
            cands_s6, feature_matrix, model, registry, t))

        for k in row:
            if k != "date":
                results[k].append(row[k])

    # Build monthly return series
    dates = [r["date"] for r in [{"date": t} for t in sim_dates
                                  if active_factors_at(t, registry, labels)
                                  and not ic_panel[(ic_panel["factor_id"].isin(
                                      active_factors_at(t, registry, labels)))
                                      & (ic_panel["date"] == t)].empty]]

    # Rebuild properly
    records = []
    for t in sim_dates:
        active = active_factors_at(t, registry, labels)
        if len(active) < 10:
            continue
        ic_t_raw = (
            ic_panel[(ic_panel["factor_id"].isin(active)) & (ic_panel["date"] == t)]
            .set_index("factor_id")["ic"]
        )
        if ic_t_raw.empty:
            continue
        records.append(t)

    return results, records


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    out_dir   = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_panel = pd.read_parquet(data_dir / "ic_series_filtered.parquet")
    ic_panel["date"] = pd.to_datetime(ic_panel["date"])
    registry = pd.read_csv(data_dir / "factor_registry.csv")
    registry["discovery_date"] = pd.to_datetime(registry["discovery_date"])
    labels   = pd.read_parquet(data_dir / "labels.parquet")
    X        = pd.read_parquet(data_dir / "feature_matrix.parquet")

    with open(model_dir / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    model = models["M3_gbs"]

    print("Precomputing signs and high-IC0 set...")
    signs       = factor_signs(registry, ic_panel)
    high_ic0    = high_ic0_factors(registry, ic_panel, threshold=0.02)
    print(f"  High IC0 factors (|IC0|≥0.02): {len(high_ic0)}")

    all_dates = sorted(ic_panel["date"].unique())
    sim_dates = [d for d in all_dates if d >= pd.Timestamp("2015-01-01")]
    print(f"Simulation: {sim_dates[0].date()} → {sim_dates[-1].date()} "
          f"({len(sim_dates)} months)\n")

    print("Running multi-strategy simulation...")
    # Simplified single-pass simulation
    strategy_ics = {k: [] for k in [
        "S0_base","S0_decay","S1_base","S1_decay","S2_base","S2_decay",
        "S3_base","S3_decay","S4_base","S4_decay","S5_base","S5_decay",
        "S6_base","S6_decay",
    ]}
    valid_dates = []

    for t in sim_dates:
        active = active_factors_at(t, registry, labels)
        if len(active) < 10:
            continue
        ic_t_raw = (
            ic_panel[(ic_panel["factor_id"].isin(active)) & (ic_panel["date"] == t)]
            .set_index("factor_id")["ic"]
        )
        if ic_t_raw.empty:
            continue
        ic_t = sign_adjust(ic_t_raw, signs)

        def ic_mom_top(k):
            cutoff = t - pd.DateOffset(months=3)
            recent = ic_panel[
                (ic_panel["factor_id"].isin(active)) &
                (ic_panel["date"] > cutoff) & (ic_panel["date"] <= t)
            ]
            adj = (recent.groupby("factor_id")["ic"]
                   .apply(lambda s: (s * s.index.map(
                       lambda f: signs.get(f, 1.0))).mean()).abs())
            return adj.nlargest(min(k, len(adj))).index.tolist()

        # Compute soft survival weights once for all active factors
        sw_active = survival_weights(active, X, model, registry, t)

        # S0
        strategy_ics["S0_base"].append(composite_ic(ic_t, active))
        strategy_ics["S0_decay"].append(composite_ic(ic_t, active, weights=sw_active))

        # S1-S4
        for sid, k in [("S1",10),("S2",20),("S3",30),("S4",50)]:
            cands = ic_mom_top(k)
            sw_cands = sw_active.reindex(cands).dropna()
            strategy_ics[f"{sid}_base"].append(composite_ic(ic_t, cands))
            strategy_ics[f"{sid}_decay"].append(composite_ic(ic_t, cands, weights=sw_cands))

        # S5 random-30
        base_vals, decay_vals = [], []
        for seed in range(10):
            rng   = random.Random(seed)
            cands = rng.sample(active, min(30, len(active)))
            sw_cands = sw_active.reindex(cands).dropna()
            base_vals.append(composite_ic(ic_t, cands))
            decay_vals.append(composite_ic(ic_t, cands, weights=sw_cands))
        strategy_ics["S5_base"].append(float(np.nanmean(base_vals)))
        strategy_ics["S5_decay"].append(float(np.nanmean(decay_vals)))

        # S6 high IC0
        cands_s6 = [f for f in active if f in high_ic0] or active[:10]
        sw_s6 = sw_active.reindex(cands_s6).dropna()
        strategy_ics["S6_base"].append(composite_ic(ic_t, cands_s6))
        strategy_ics["S6_decay"].append(composite_ic(ic_t, cands_s6, weights=sw_s6))

        valid_dates.append(t)

    print(f"  {len(valid_dates)} valid months\n")

    cs_vol = 0.10
    strategy_names = {
        "S0": "Equal-weight all",
        "S1": "IC-momentum Top-10",
        "S2": "IC-momentum Top-20",
        "S3": "IC-momentum Top-30",
        "S4": "IC-momentum Top-50",
        "S5": "Random-30 (avg 10 seeds)",
        "S6": "High IC₀ (|IC₀|≥0.02)",
    }

    print(f"{'Strategy':<28} {'AR_base':>8} {'SR_base':>8} "
          f"{'AR_ours':>8} {'SR_ours':>8} "
          f"{'ΔAR':>8} {'ΔSR':>8}")
    print("-" * 80)

    summary = []
    for sid in ["S0","S1","S2","S3","S4","S5","S6"]:
        ret_base  = pd.Series(strategy_ics[f"{sid}_base"]) * cs_vol
        ret_decay = pd.Series(strategy_ics[f"{sid}_decay"]) * cs_vol
        m_base  = compute_metrics(ret_base)
        m_decay = compute_metrics(ret_decay)
        delta_ar = m_decay["AR"] - m_base["AR"]
        delta_sr = m_decay["SR"] - m_base["SR"]
        name = strategy_names[sid]
        print(f"{name:<28} {m_base['AR']:>8.2%} {m_base['SR']:>8.3f} "
              f"{m_decay['AR']:>8.2%} {m_decay['SR']:>8.3f} "
              f"{delta_ar:>+8.2%} {delta_sr:>+8.3f}")
        summary.append({
            "Strategy": name,
            "AR_base": m_base["AR"], "SR_base": m_base["SR"],
            "MDD_base": m_base["MDD"], "CR_base": m_base["CR"],
            "AR_ours": m_decay["AR"], "SR_ours": m_decay["SR"],
            "MDD_ours": m_decay["MDD"], "CR_ours": m_decay["CR"],
            "delta_AR": delta_ar, "delta_SR": delta_sr,
        })

    pd.DataFrame(summary).to_csv(out_dir / "robustness_results.csv", index=False)
    print(f"\nSaved → {out_dir}/robustness_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
