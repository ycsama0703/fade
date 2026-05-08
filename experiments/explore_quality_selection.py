"""
Exploration: cross-sectional factor selection using survival score.

Hypothesis: predicted half-life (pred_HL) is an orthogonal "durability"
dimension to |IC0| (signal strength). Combining them as a Quality score
should beat selecting on either alone, even on synthetic factors where
pure timing-based exit doesn't help.

Out-of-sample: only test-set factors (disc >= 2017-01) are used.
Sim window: 2017-06 to end of IC data.
"""
from __future__ import annotations
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
MOD  = ROOT / "models" / "checkpoints"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


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
    # Load data
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    labels = pd.read_parquet(DATA / "labels.parquet").set_index("factor_id")
    X_test = pd.read_parquet(DATA / "X_test.parquet")
    with open(MOD / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    m = models["M3_gbs"]

    test_ids = X_test.index.tolist()
    reg_t = reg[reg.factor_id.isin(test_ids)].set_index("factor_id")
    print(f"Test factors: {len(test_ids)}  disc range "
          f"{reg_t.discovery_date.min().date()} → {reg_t.discovery_date.max().date()}")

    # Predict HL out-of-sample
    pred_hl = pd.Series(m.predict_median_survival(X_test),
                        index=X_test.index, name="pred_hl")

    # |IC0|: first 6m after discovery (forward-safe)
    def ic0_abs(fid, w=6):
        disc = reg_t.loc[fid, "discovery_date"]
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:w]
        return abs(g.mean()) if len(g) >= w else np.nan

    ic0 = pd.Series({f: ic0_abs(f) for f in test_ids}, name="ic0_abs")

    # Sign: from first 6m IC
    def sign(fid, w=6):
        disc = reg_t.loc[fid, "discovery_date"]
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:w]
        return 1.0 if g.mean() >= 0 else -1.0
    signs = {f: sign(f) for f in test_ids}

    # Score table
    scores = pd.concat([ic0, pred_hl], axis=1).dropna()
    # Add z-scores
    scores["z_ic0"]  = (scores.ic0_abs - scores.ic0_abs.mean()) / scores.ic0_abs.std()
    # Use log to dampen heavy tail of pred_hl
    log_hl = np.log1p(scores.pred_hl)
    scores["z_loghl"] = (log_hl - log_hl.mean()) / log_hl.std()
    scores["composite_sum"]  = scores.z_ic0 + scores.z_loghl
    scores["composite_prod"] = scores.ic0_abs * np.log1p(scores.pred_hl)
    print("\n--- Score correlations ---")
    print(scores[["ic0_abs","pred_hl","composite_sum","composite_prod"]].corr(method="spearman").round(3))

    # Sim dates: monthly snapshots from 2017-06 to end
    all_dates = sorted(ic.date.unique())
    sim_dates = [d for d in all_dates if d >= pd.Timestamp("2017-06-01")]
    print(f"\nSim months: {len(sim_dates)}  ({sim_dates[0].date()} → {sim_dates[-1].date()})")

    # Active set: discovered AND not L1-decayed
    def active_at(t):
        out = []
        for fid in test_ids:
            disc = reg_t.loc[fid, "discovery_date"]
            if disc > t:
                continue
            elapsed = months_between(disc, t)
            dur = labels.loc[fid, "L1_duration"]
            evt = int(labels.loc[fid, "L1_event"])
            if evt == 1 and elapsed >= dur:
                continue
            # Need 6m of post-disc IC for ic0 to be valid
            if elapsed < 6:
                continue
            out.append(fid)
        return out

    # Sign-adjusted IC at t for given subset
    def composite(t, subset):
        sub = ic[(ic.factor_id.isin(subset)) & (ic.date == t)]
        if sub.empty: return np.nan
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return float(v.mean())

    # Selection rules — all return a list of factor_ids
    def topk_by(score: pd.Series, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    rules = {
        "All":              lambda c, k: c,
        "Top|IC0|":         lambda c, k: topk_by(scores.ic0_abs,        c, k),
        "TopPredHL":        lambda c, k: topk_by(scores.pred_hl,        c, k),
        "TopComposite_sum": lambda c, k: topk_by(scores.composite_sum,  c, k),
        "TopComposite_prod":lambda c, k: topk_by(scores.composite_prod, c, k),
    }
    # Filter-then-rank: top-2K by IC0, then top-K by pred_HL
    def filter_then_rank(c, k):
        first  = topk_by(scores.ic0_abs, c, 2*k)
        return  topk_by(scores.pred_hl, first, k)
    rules["IC0→PredHL"] = filter_then_rank

    # Run sim per K
    cs_vol = 0.10
    summary = []
    cumret_dict = {}
    for K in [10, 20, 30, 50]:
        for name, fn in rules.items():
            rets = {}
            for t in sim_dates:
                cands = active_at(t)
                if len(cands) < K:
                    continue
                sel = fn(cands, K)
                if len(sel) < 3:
                    continue
                ic_t = composite(t, sel)
                if not np.isnan(ic_t):
                    rets[t] = ic_t * cs_vol
            r = pd.Series(rets).sort_index()
            mt = metrics(r)
            label = f"K={K} {name}"
            cumret_dict[label] = (1 + r).cumprod()
            summary.append({"K": K, "Rule": name, **mt})
            print(f"  {label:<28}  CR={mt['CR']:>+7.2%}  AR={mt['AR']:>+7.2%}  "
                  f"SR={mt['SR']:>6.3f}  MDD={mt['MDD']:>+7.2%}  N={mt['N']}")
        print()

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "explore_quality_selection.csv", index=False)
    pd.DataFrame(cumret_dict).to_csv(OUT / "explore_quality_cumret.csv")
    print(f"Saved → {OUT}/explore_quality_selection.csv")

    # ---- Also: nearer-term predictive power of pred_HL conditional on IC0 ----
    # For each test factor, compute mean sign-adjusted IC over months 7-18
    # post-discovery (the "near-mid term"), and see if pred_HL predicts it
    # after controlling for IC0.
    print("\n--- Near-term IC (months 7-18 post-disc) vs pred_HL conditional on |IC0| ---")
    near_ic = {}
    for fid in test_ids:
        disc = reg_t.loc[fid, "discovery_date"]
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic
        if len(g) < 18: continue
        v = g.iloc[6:18] * signs[fid]
        near_ic[fid] = float(v.mean())
    near = pd.Series(near_ic, name="near_ic")
    cmp = pd.concat([scores, near], axis=1).dropna()
    print(f"  N = {len(cmp)}")
    print("  Spearman corr(near_ic, ...):")
    for col in ["ic0_abs","pred_hl","composite_sum","composite_prod"]:
        print(f"    {col:<18} {cmp[[col,'near_ic']].corr(method='spearman').iloc[0,1]:+.3f}")
    # Partial: regress near_ic on z_ic0, see if z_loghl adds R^2
    from sklearn.linear_model import LinearRegression
    yv = cmp.near_ic.values
    for cols in [["z_ic0"], ["z_loghl"], ["z_ic0","z_loghl"]]:
        Xv = cmp[cols].values
        r2 = LinearRegression().fit(Xv, yv).score(Xv, yv)
        print(f"  R^2 on near_ic: features={cols}  R²={r2:.4f}")


if __name__ == "__main__":
    main()
