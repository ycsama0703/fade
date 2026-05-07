"""
Stage 6: Decay-aware factor timing — portfolio simulation.

Core mechanism
--------------
After a factor's IC sign flips (L1 decay event), its sign-adjusted IC becomes
NEGATIVE — it actively hurts the portfolio. The value of our half-life
prediction is that it tells us when to EXIT a factor BEFORE this flip happens.

Experiment design
-----------------
Each strategy independently decides when to hold/exit each factor.
No oracle filtering is applied as a common preprocessing step — the oracle
information is only used by the "Oracle" strategy itself.

Strategies compared
-------------------
  Hold-Forever   Include factor from discovery until end of simulation
                 (suffers full negative-IC penalty after decay)
  Fixed-6        Exit at disc + 6 months
  Fixed-12       Exit at disc + 12 months
  Fixed-24       Exit at disc + 24 months
  IC-trailing    Exit when 3-month trailing signed-IC turns negative for 2+
                 consecutive months (real-time but reactive rule)
  Predicted      Exit at disc + GBS-predicted half-life  [our method]
  Oracle         Exit at disc + true L1 half-life        [upper bound]

Monthly return = equal_weight_composite(signed_IC) × cs_vol (0.10).
Sign of each factor is fixed from its first 6 months of IC (no lookahead).
After L1 decay (sign flip), the sign-adjusted IC contribution is negative.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def months_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def compute_metrics(monthly_returns: pd.Series) -> dict:
    r = monthly_returns.dropna()
    if len(r) == 0:
        return dict(CR=np.nan, AR=np.nan, SR=np.nan, MDD=np.nan, N=0)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r))


def factor_signs(registry: pd.DataFrame, ic_panel: pd.DataFrame,
                 initial_window: int = 6) -> dict[str, float]:
    """Sign fixed from early IC — no lookahead."""
    signs = {}
    for _, row in registry.iterrows():
        fid  = row["factor_id"]
        disc = pd.Timestamp(row["discovery_date"])
        grp  = (ic_panel[(ic_panel["factor_id"] == fid) & (ic_panel["date"] >= disc)]
                .sort_values("date"))
        early = grp["ic"].iloc[:initial_window].mean()
        signs[fid] = 1.0 if early >= 0 else -1.0
    return signs


# ---------------------------------------------------------------------------
# Exit-date computation per strategy (static, evaluated at discovery time)
# ---------------------------------------------------------------------------

def exits_hold_forever(registry: pd.DataFrame) -> dict[str, pd.Timestamp]:
    return {row["factor_id"]: pd.Timestamp("2099-01-01")
            for _, row in registry.iterrows()}


def exits_fixed(registry: pd.DataFrame, k: int) -> dict[str, pd.Timestamp]:
    return {
        row["factor_id"]: pd.Timestamp(row["discovery_date"]) + pd.DateOffset(months=k)
        for _, row in registry.iterrows()
    }


def exits_predicted(registry: pd.DataFrame, X: pd.DataFrame,
                    model, offset: int = 2) -> dict[str, pd.Timestamp]:
    """
    Exit at predicted HL minus `offset` months.
    offset = persist_months - 1 = 2: exits just before the rolling-mean
    confirmation window starts, i.e. before the IC has turned negative.
    """
    common = X.index.intersection(registry["factor_id"])
    preds  = model.predict_median_survival(X.loc[common])
    pred_s = pd.Series(preds, index=common).clip(lower=1.0)

    reg_idx = registry.set_index("factor_id")
    exits = {}
    for fid in registry["factor_id"]:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        hl   = float(pred_s.get(fid, 24.0))
        hold = max(round(hl) - offset, 2)
        exits[fid] = disc + pd.DateOffset(months=hold)
    return exits


def exits_oracle(registry: pd.DataFrame, labels: pd.DataFrame,
                 offset: int = 2) -> dict[str, pd.Timestamp]:
    """
    Exit at true HL minus `offset` months (upper bound).
    offset = 2 = persist_months - 1: exits right when IC first turns negative,
    before the rolling-mean confirmation. This is the best achievable timing.
    """
    lab_idx = labels.set_index("factor_id") if "factor_id" in labels.columns else labels
    reg_idx = registry.set_index("factor_id")
    exits = {}
    for fid in registry["factor_id"]:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        dur  = float(lab_idx.loc[fid, "L1_duration"])
        evt  = int(lab_idx.loc[fid, "L1_event"])
        if evt == 1:
            hold = max(round(dur) - offset, 2)
            exits[fid] = disc + pd.DateOffset(months=hold)
        else:
            exits[fid] = pd.Timestamp("2099-01-01")
    return exits


def exits_ic_trailing(
    registry: pd.DataFrame,
    ic_panel: pd.DataFrame,
    signs: dict[str, float],
    lookback: int = 3,
    persist: int = 2,
) -> dict[str, pd.Timestamp]:
    """
    Real-time exit: triggered when trailing `lookback`-month signed IC is
    negative AND stays negative for `persist` more months.
    Scanned forward in time per factor (no lookahead).
    """
    reg_idx = registry.set_index("factor_id")
    exits = {}
    for fid in registry["factor_id"]:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        sign = signs.get(fid, 1.0)
        grp  = (ic_panel[(ic_panel["factor_id"] == fid) & (ic_panel["date"] > disc)]
                .sort_values("date"))
        dates = grp["date"].tolist()
        ics   = (grp["ic"] * sign).tolist()

        exit_date = pd.Timestamp("2099-01-01")
        for i in range(lookback, len(ics)):
            trailing = np.mean(ics[max(0, i - lookback):i])
            if trailing < 0:
                future = ics[i: i + persist]
                if len(future) == persist and all(v < 0 for v in future):
                    exit_date = dates[i]
                    break
        exits[fid] = exit_date
    return exits


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate(
    exits: dict[str, pd.Timestamp],
    sim_dates: list[pd.Timestamp],
    ic_panel: pd.DataFrame,
    registry: pd.DataFrame,
    signs: dict[str, float],
    cs_vol: float = 0.10,
) -> tuple[pd.Series, float]:
    """
    Monthly returns for a strategy defined by its exit-date dict.

    A factor is INCLUDED at month t if:  disc_i ≤ t  AND  t < exit_i
    Its sign-adjusted IC contribution can be negative (post-decay penalty).
    """
    reg_idx = registry.set_index("factor_id")
    all_fids = registry["factor_id"].tolist()
    # Preindex IC panel for speed
    ic_idx = ic_panel.set_index(["factor_id", "date"])["ic"]

    records: dict[pd.Timestamp, float] = {}
    n_actives = []

    for t in sim_dates:
        included = []
        for fid in all_fids:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t or t >= exits[fid]:
                continue
            included.append(fid)

        if len(included) < 3:
            continue

        # Gather sign-adjusted IC (includes negative contributions post-decay)
        ic_vals = []
        for fid in included:
            try:
                raw_ic = ic_idx.loc[(fid, t)]
                ic_vals.append(raw_ic * signs.get(fid, 1.0))
            except KeyError:
                pass

        if len(ic_vals) < 3:
            continue

        records[t] = float(np.mean(ic_vals)) * cs_vol
        n_actives.append(len(included))

    return pd.Series(records), float(np.mean(n_actives)) if n_actives else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    labels = pd.read_parquet(data_dir / "labels.parquet")
    X      = pd.read_parquet(data_dir / "feature_matrix.parquet")

    with open(model_dir / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    model = models["M3_gbs"]

    all_dates = sorted(ic_panel["date"].unique())
    sim_dates = [d for d in all_dates if d >= pd.Timestamp("2015-01-01")]
    print(f"Simulation: {sim_dates[0].date()} → {sim_dates[-1].date()} "
          f"({len(sim_dates)} months)\n")

    print("Computing factor signs...")
    signs = factor_signs(registry, ic_panel)
    print()

    # ------------------------------------------------------------------
    # Compute exit dates for each strategy (once, at start)
    # ------------------------------------------------------------------
    print("Precomputing exit dates...")
    all_exits = {
        "Hold-Forever": exits_hold_forever(registry),
        "Fixed-6":      exits_fixed(registry, 6),
        "Fixed-12":     exits_fixed(registry, 12),
        "Fixed-24":     exits_fixed(registry, 24),
        "IC-trailing":  exits_ic_trailing(registry, ic_panel, signs),
        "Predicted":    exits_predicted(registry, X, model),
        "Oracle":       exits_oracle(registry, labels),
    }

    # Summarise timing stats vs oracle
    oracle_exits = all_exits["Oracle"]
    pred_exits   = all_exits["Predicted"]
    reg_idx      = registry.set_index("factor_id")
    oracle_hl_s  = pd.Series({
        fid: months_between(pd.Timestamp(reg_idx.loc[fid,"discovery_date"]),
                            oracle_exits[fid])
        for fid in registry["factor_id"]
        if oracle_exits[fid] < pd.Timestamp("2090-01-01")  # event=1 only
    })
    pred_hl_s = pd.Series({
        fid: months_between(pd.Timestamp(reg_idx.loc[fid,"discovery_date"]),
                            pred_exits[fid])
        for fid in registry["factor_id"]
    })
    mae = float((pred_hl_s.reindex(oracle_hl_s.index) - oracle_hl_s).abs().mean())
    print(f"  Oracle HL (event=1): mean={oracle_hl_s.mean():.1f}m  "
          f"median={oracle_hl_s.median():.1f}m")
    print(f"  Predicted HL:        mean={pred_hl_s.mean():.1f}m  "
          f"median={pred_hl_s.median():.1f}m  MAE={mae:.1f}m")
    print()

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------
    strategy_order = ["Hold-Forever","Fixed-6","Fixed-12","Fixed-24",
                      "IC-trailing","Predicted","Oracle"]
    print("Running simulations...")
    returns_all = {}
    avg_active  = {}
    for name in strategy_order:
        ret, avg_n = simulate(all_exits[name], sim_dates, ic_panel,
                              registry, signs, cs_vol=0.10)
        returns_all[name] = ret
        avg_active[name]  = avg_n
        print(f"  [{name:<14}]  {len(ret):>3} months  avg_active={avg_n:.0f}")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n")
    header = (f"{'Strategy':<16} {'CR':>8} {'AR':>8} {'SR':>8} "
              f"{'MDD':>8}  {'AvgN':>6}")
    print(header)
    print("-" * len(header))

    rows = []
    for name in strategy_order:
        m     = compute_metrics(returns_all[name])
        avg_n = avg_active[name]
        flag  = " ← ours" if name == "Predicted" else \
                " ← upper bound" if name == "Oracle" else ""
        print(f"{name:<16} {m['CR']:>8.2%} {m['AR']:>8.2%} "
              f"{m['SR']:>8.3f} {m['MDD']:>8.2%}  {avg_n:>6.0f}{flag}")
        rows.append({"Strategy": name, **m, "AvgActive": avg_n})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "simulation_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Improvement vs Hold-Forever and vs Fixed-24
    # ------------------------------------------------------------------
    for baseline in ["Hold-Forever", "Fixed-24"]:
        m_base = compute_metrics(returns_all[baseline])
        print(f"\n--- Improvement vs {baseline} ---")
        print(f"{'Strategy':<16} {'ΔAR':>8} {'ΔSR':>8} {'ΔMDD':>8}")
        for name in strategy_order:
            if name == baseline:
                continue
            m = compute_metrics(returns_all[name])
            flag = " ← ours" if name == "Predicted" else \
                   " ← ub" if name == "Oracle" else ""
            print(f"{name:<16} {m['AR']-m_base['AR']:>+8.2%} "
                  f"{m['SR']-m_base['SR']:>+8.3f} "
                  f"{m['MDD']-m_base['MDD']:>+8.2%}{flag}")

    # Cumulative return curves
    cum = pd.DataFrame({
        name: (1 + returns_all[name].reindex(sim_dates, fill_value=np.nan)).cumprod() - 1
        for name in strategy_order
    })
    cum.to_csv(out_dir / "simulation_cumret.csv")
    print(f"\nSaved → {out_dir}/simulation_metrics.csv  &  simulation_cumret.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
