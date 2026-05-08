"""
Exploration v2: full-pool quality selection with replenishment.

Two key changes vs explore_quality_selection.py:
- Full factor pool (408) — train+val+test — for statistical power.
  pred_HL is mildly leaky for train factors but used only as a relative
  selection score, not as a label. Treat as concept-validation experiment.
- Compares fixed-K selection rules in a regime where active pool grows
  organically and decays naturally.

Also tests "remaining-life" formulations:
  - pred_remaining = pred_HL - elapsed_at_t  (dynamic, recomputed each month)
  - quality_dyn = z(|IC0|) + z(log1p(max(pred_remaining, 1)))
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
MOD  = ROOT / "models" / "checkpoints"
OUT  = ROOT / "outputs"


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return dict(CR=np.nan, AR=np.nan, SR=np.nan, MDD=np.nan, N=0)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r))


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    labels = pd.read_parquet(DATA / "labels.parquet").set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")
    with open(MOD / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    m = models["M3_gbs"]

    pred_hl = pd.Series(m.predict_median_survival(X), index=X.index, name="pred_hl")

    all_ids = X.index.tolist()
    print(f"Pool size: {len(all_ids)}")

    # Pre-compute |IC0| & sign from first 6m
    def first_window(fid, w=6):
        disc = reg_idx.loc[fid, "discovery_date"]
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:w]
        return g if len(g) >= w else None

    ic0_abs = {}
    signs = {}
    for f in all_ids:
        g = first_window(f)
        if g is None:
            continue
        ic0_abs[f] = abs(g.mean())
        signs[f]   = 1.0 if g.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs, name="ic0_abs")

    # Static z-scores
    log_hl = np.log1p(pred_hl)
    z_ic0  = (ic0 - ic0.mean()) / ic0.std()
    z_lhl  = (log_hl - log_hl.mean()) / log_hl.std()
    composite_static = (z_ic0 + z_lhl).rename("composite_static")

    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp("2015-01-01")]
    print(f"Sim months: {len(sim_dates)}  ({sim_dates[0].date()} → {sim_dates[-1].date()})")

    def active_at(t):
        out = []
        for fid in all_ids:
            if fid not in signs: continue
            disc = reg_idx.loc[fid, "discovery_date"]
            if disc > t: continue
            elapsed = months_between(disc, t)
            if elapsed < 6: continue
            dur = labels.loc[fid, "L1_duration"]
            evt = int(labels.loc[fid, "L1_event"])
            if evt == 1 and elapsed >= dur:
                continue
            out.append(fid)
        return out

    def composite_ic(t, subset):
        sub = ic[(ic.factor_id.isin(subset)) & (ic.date == t)]
        if sub.empty: return np.nan
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return float(v.mean())

    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    # Dynamic remaining-life score per (t, factor)
    def dyn_quality(cands, t):
        rem = {}
        for f in cands:
            disc = reg_idx.loc[f, "discovery_date"]
            elapsed = months_between(disc, t)
            rem[f] = max(float(pred_hl[f]) - elapsed, 1.0)
        rem_s   = pd.Series(rem)
        z_rem   = (np.log1p(rem_s) - np.log1p(rem_s).mean()) / np.log1p(rem_s).std()
        z_ic0_c = z_ic0.reindex(cands)
        return (z_ic0_c + z_rem).dropna()

    rules = {
        "All":             lambda c, k, t: c,
        "Top|IC0|":        lambda c, k, t: topk(ic0,                c, k),
        "TopPredHL":       lambda c, k, t: topk(pred_hl,            c, k),
        "Composite_static":lambda c, k, t: topk(composite_static,   c, k),
        "Composite_prod":  lambda c, k, t: topk(ic0 * np.log1p(pred_hl), c, k),
        "Composite_dyn":   lambda c, k, t: dyn_quality(c, t).nlargest(min(k, len(c))).index.tolist(),
    }
    # Filter-then-rank variants
    rules["IC0→PredHL"]   = lambda c, k, t: topk(pred_hl, topk(ic0, c, 2*k), k)
    rules["PredHL→IC0"]   = lambda c, k, t: topk(ic0,    topk(pred_hl, c, 3*k), k)

    cs_vol = 0.10
    summary = []
    cumret = {}
    for K in [10, 20, 30, 50]:
        print(f"\n=== K={K} ===")
        for name, fn in rules.items():
            rets = {}
            for t in sim_dates:
                cands = active_at(t)
                if len(cands) < K: continue
                sel = fn(cands, K, t)
                if len(sel) < 3: continue
                v = composite_ic(t, sel)
                if not np.isnan(v):
                    rets[t] = v * cs_vol
            r = pd.Series(rets).sort_index()
            mt = metrics(r)
            print(f"  {name:<20}  CR={mt['CR']:>+8.2%}  AR={mt['AR']:>+7.2%}  "
                  f"SR={mt['SR']:>6.3f}  MDD={mt['MDD']:>+7.2%}  N={mt['N']}")
            summary.append({"K": K, "Rule": name, **mt})
            cumret[f"K{K}_{name}"] = (1 + r).cumprod()

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "explore_quality_full.csv", index=False)
    pd.DataFrame(cumret).to_csv(OUT / "explore_quality_full_cumret.csv")
    print(f"\nSaved → {OUT}/explore_quality_full.csv")

    # ---- Headline summary: at each K, show ΔAR / ΔSR / ΔMDD vs Top|IC0| baseline ----
    print("\n=== Headline: improvement of best survival-augmented rule vs Top|IC0| ===")
    base = df[df.Rule == "Top|IC0|"].set_index("K")
    for K in [10, 20, 30, 50]:
        sub = df[df.K == K].copy()
        b = base.loc[K]
        # Score: prioritize SR; tiebreak AR
        sub["dAR"] = sub.AR - b.AR
        sub["dSR"] = sub.SR - b.SR
        sub["dMDD"] = sub.MDD - b.MDD
        # exclude "All" and "Top|IC0|" from "augmented" candidates
        cand = sub[~sub.Rule.isin(["All","Top|IC0|"])].sort_values("SR", ascending=False)
        if cand.empty: continue
        best = cand.iloc[0]
        print(f"  K={K:<3}  baseline Top|IC0|  AR={b.AR:+.2%} SR={b.SR:.3f}  →  "
              f"best={best.Rule:<18} AR={best.AR:+.2%} (Δ{best.dAR:+.2%})  "
              f"SR={best.SR:.3f} (Δ{best.dSR:+.3f})  MDD Δ{best.dMDD:+.2%}")


if __name__ == "__main__":
    main()
