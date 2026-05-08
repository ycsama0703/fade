"""
Paper anchor experiment for Alpha158.

Two contributions tested back-to-back on the SAME factor pool:

  Anchor A — Cross-sectional decay-aware selection
    For each rebalance month, score every active factor by various rules
    and take top-K. We compare:
      Baseline: Top|IC0|
      Aug:      Composite = z(|IC0|) + z(log pred_HL_remaining)
                Composite_prod = |IC0| * log(pred_HL_remaining)
                Filter-then-rank: top-2K |IC0| -> top-K pred_HL

  Anchor B — Decay-aware exit (timing)
    For each factor, exit at predicted half-life (-2 month offset) before
    IC sign flip drag in. Compared to Hold-Forever, Fixed-K, IC-trailing,
    Oracle.

To produce honest predictions on all 158 factors, we use 5-fold CV: each
factor's pred_HL comes from a model that did NOT see its label.

Outputs:
  outputs/anchor_A_selection.csv
  outputs/anchor_B_timing.csv
  outputs/anchor_summary.csv
"""

from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
MOD  = ROOT / "models" / "checkpoints"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ---------- helpers ----------
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


def fit_and_predict_oos(X: pd.DataFrame, y: pd.DataFrame, n_splits=5, seed=42):
    """5-fold CV: returns pd.Series of OOS predicted median-survival per factor."""
    from src.models.survival_ensemble import GradientBoostingSurvival
    # Align X and y on common index
    common = X.index.intersection(y.index)
    X = X.loc[common].copy()
    y = y.loc[common].copy()
    # Drop rows with NaN labels globally
    mask = y["L1_duration"].notna() & y["L1_event"].notna()
    X, y = X.loc[mask], y.loc[mask]
    print(f"  [CV] aligned: X={X.shape}  y={y.shape}")

    preds = pd.Series(index=X.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte      = X.iloc[te]
        m = GradientBoostingSurvival(random_state=seed)
        # GradientBoostingSurvival.fit() does its own get_dummies + astype(float)
        m.fit(Xtr, ytr["L1_duration"], ytr["L1_event"].astype(int))
        preds.iloc[te] = m.predict_median_survival(Xte)
        print(f"  fold {fold+1}/{n_splits}: train={len(Xtr)} test={len(Xte)}")
    return preds.rename("pred_hl")


# ---------- main ----------
def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    labels = pd.read_parquet(DATA / "labels.parquet").set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    print(f"Pool: {len(reg)} factors  |  IC dates: "
          f"{ic.date.min().date()} → {ic.date.max().date()}")
    print(f"L1 event rate: {labels.L1_event.mean():.1%}")

    # ---- Out-of-fold pred_HL ----
    print("\n[CV] Computing OOS pred_HL via 5-fold CV ...")
    pred_hl = fit_and_predict_oos(X, labels, n_splits=5, seed=42)
    pred_hl = pred_hl.clip(lower=1.0).fillna(pred_hl.median())
    print(f"  pred_HL: median={pred_hl.median():.1f}  mean={pred_hl.mean():.1f}  "
          f"q90={pred_hl.quantile(0.9):.1f}")

    # ---- |IC0| & sign from first 6m post-disc ----
    print("\n[features] Computing |IC0| and sign from first 6m post-discovery ...")
    ic0_abs, signs = {}, {}
    for fid in reg.factor_id:
        disc = reg_idx.loc[fid, "discovery_date"]
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6:
            continue
        ic0_abs[fid] = abs(g.mean())
        signs[fid]   = 1.0 if g.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)
    print(f"  |IC0|: median={ic0.median():.4f}  q90={ic0.quantile(0.9):.4f}")

    # Save scores
    scores = pd.concat([ic0.rename("ic0_abs"), pred_hl], axis=1).dropna()
    scores.to_parquet(DATA / "_anchor_scores.parquet")
    sp = scores.corr(method="spearman").iloc[0, 1]
    print(f"  Spearman corr(|IC0|, pred_HL) = {sp:+.3f}")

    # =====================================================================
    # ANCHOR B — Decay-aware timing exit
    # =====================================================================
    print("\n========== ANCHOR B: Decay-aware timing ==========")
    sim_dates = [d for d in sorted(ic.date.unique())
                 if d >= pd.Timestamp("2012-01-01")]
    print(f"Sim months: {len(sim_dates)}  ({sim_dates[0].date()} → {sim_dates[-1].date()})")

    def composite_ic(t, subset, signs_d):
        sub = ic[(ic.factor_id.isin(subset)) & (ic.date == t)]
        if sub.empty: return np.nan
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs_d).reindex(v.index)
        return float(v.mean())

    def exits_for_strategy(name):
        """Return dict[fid] = exit_date for each strategy."""
        out = {}
        for fid in reg.factor_id:
            if fid not in signs:
                continue
            disc = reg_idx.loc[fid, "discovery_date"]
            if name == "Hold-Forever":
                out[fid] = pd.Timestamp("2099-01-01")
            elif name.startswith("Fixed-"):
                k = int(name.split("-")[1])
                out[fid] = disc + pd.DateOffset(months=k)
            elif name == "Predicted":
                hl = float(pred_hl.get(fid, 24.0))
                hold = max(round(hl) - 2, 2)
                out[fid] = disc + pd.DateOffset(months=hold)
            elif name == "Oracle":
                dur = labels.loc[fid, "L1_duration"]
                evt = int(labels.loc[fid, "L1_event"])
                if evt == 1:
                    out[fid] = disc + pd.DateOffset(months=max(round(dur)-2, 2))
                else:
                    out[fid] = pd.Timestamp("2099-01-01")
        return out

    timing_strats = ["Hold-Forever", "Fixed-12", "Fixed-24", "Fixed-36",
                     "Predicted", "Oracle"]
    timing_rows = []
    cs_vol = 0.10
    for s in timing_strats:
        ex = exits_for_strategy(s)
        rets = {}
        for t in sim_dates:
            included = []
            for fid in signs:
                disc = reg_idx.loc[fid, "discovery_date"]
                if disc > t or t >= ex.get(fid, pd.Timestamp("1900-01-01")):
                    continue
                included.append(fid)
            if len(included) < 5:
                continue
            v = composite_ic(t, included, signs)
            if not np.isnan(v):
                rets[t] = v * cs_vol
        r = pd.Series(rets).sort_index()
        m = metrics(r)
        timing_rows.append({"Strategy": s, **m})
        print(f"  {s:<14}  CR={m['CR']:>+8.2%}  AR={m['AR']:>+7.2%}  "
              f"SR={m['SR']:>6.3f}  MDD={m['MDD']:>+7.2%}  N={m['N']}")
    pd.DataFrame(timing_rows).to_csv(OUT / "anchor_B_timing.csv", index=False)

    # =====================================================================
    # ANCHOR A — Cross-sectional selection
    # =====================================================================
    print("\n========== ANCHOR A: Cross-sectional decay-aware selection ==========")
    log_hl = np.log1p(pred_hl)
    z_ic0  = (ic0 - ic0.mean()) / ic0.std()
    z_lhl  = (log_hl - log_hl.mean()) / log_hl.std()
    composite_static = (z_ic0 + z_lhl)
    composite_prod_s = ic0 * np.log1p(pred_hl)

    def active_at(t):
        # "Active" = has at least 6m post-discovery history at t.
        # We do NOT use L1_duration here (that's forward info) — let
        # the selection score do the work of avoiding decayed factors.
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out.append(fid)
        return out

    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    def composite_dyn_topk(cands, k, t):
        rem = {}
        for f in cands:
            disc = pd.Timestamp(reg_idx.loc[f, "discovery_date"])
            elapsed = months_between(disc, t)
            rem[f] = max(float(pred_hl.get(f, 12)) - elapsed, 1.0)
        rem_s = pd.Series(rem)
        z_rem = (np.log1p(rem_s) - np.log1p(rem_s).mean()) / np.log1p(rem_s).std()
        z_ic  = z_ic0.reindex(cands)
        score = (z_ic + z_rem).dropna()
        return score.nlargest(min(k, len(score))).index.tolist()

    rules = {
        "All":              lambda c, k, t: c,
        "Top|IC0|":         lambda c, k, t: topk(ic0,                c, k),
        "TopPredHL":        lambda c, k, t: topk(pred_hl,            c, k),
        "Composite_static": lambda c, k, t: topk(composite_static,   c, k),
        "Composite_prod":   lambda c, k, t: topk(composite_prod_s,   c, k),
        "Composite_dyn":    composite_dyn_topk,
        "IC0→PredHL":       lambda c, k, t: topk(pred_hl, topk(ic0, c, 2*k), k),
    }

    sel_rows = []
    for K in [10, 20, 30, 50]:
        print(f"\n--- K = {K} ---")
        for name, fn in rules.items():
            rets = {}
            for t in sim_dates:
                cands = active_at(t)
                if len(cands) < K: continue
                sel = fn(cands, K, t)
                if len(sel) < 3: continue
                v = composite_ic(t, sel, signs)
                if not np.isnan(v):
                    rets[t] = v * cs_vol
            r = pd.Series(rets).sort_index()
            m = metrics(r)
            print(f"  {name:<18}  CR={m['CR']:>+8.2%}  AR={m['AR']:>+7.2%}  "
                  f"SR={m['SR']:>6.3f}  MDD={m['MDD']:>+7.2%}  N={m['N']}")
            sel_rows.append({"K": K, "Rule": name, **m})
    df = pd.DataFrame(sel_rows)
    df.to_csv(OUT / "anchor_A_selection.csv", index=False)

    # ---- Summary table: best aug rule vs Top|IC0| at each K ----
    print("\n========== Headline ==========")
    rows = []
    base = df[df.Rule == "Top|IC0|"].set_index("K")
    for K in [10, 20, 30, 50]:
        sub = df[df.K == K].copy()
        b = base.loc[K]
        cand = sub[~sub.Rule.isin(["All", "Top|IC0|"])].sort_values("SR", ascending=False)
        if cand.empty: continue
        best = cand.iloc[0]
        print(f"  K={K:<3}  Top|IC0|: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%}  →  "
              f"{best.Rule:<18}: AR={best.AR:+.2%} (Δ{best.AR-b.AR:+.2%})  "
              f"SR={best.SR:.3f} (Δ{best.SR-b.SR:+.3f})  MDD={best.MDD:+.2%} (Δ{best.MDD-b.MDD:+.2%})")
        rows.append({"K": K, "best_rule": best.Rule,
                     "AR_base": b.AR, "AR_aug": best.AR, "dAR": best.AR - b.AR,
                     "SR_base": b.SR, "SR_aug": best.SR, "dSR": best.SR - b.SR,
                     "MDD_base": b.MDD, "MDD_aug": best.MDD, "dMDD": best.MDD - b.MDD})
    pd.DataFrame(rows).to_csv(OUT / "anchor_summary.csv", index=False)
    print(f"\nSaved → outputs/anchor_A_selection.csv  anchor_B_timing.csv  anchor_summary.csv")


if __name__ == "__main__":
    main()
