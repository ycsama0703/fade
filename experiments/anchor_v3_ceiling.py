"""
Anchor v3 — close the loop on the v2 winners.

Adds:
  1. Oracle-selection ceiling: at each t, pick K factors with highest
     ACTUAL next-month sign-adjusted IC. Realistic upper bound for any
     ex-ante selection rule. Fraction of gap closed = (rule - base) / (oracle - base).
  2. IC-momentum head-to-head: replicate Stage-7 style "trailing 3m IC" top-K
     to confirm our gains aren't just "any selection beats none".
  3. ICIR-Top: select by Information Coefficient Information Ratio (ic_mean_6m / ic_std_6m).
     This checks whether half the win is just "ICIR > raw |IC0|" rather than pred_HL.
  4. Per-year SR breakdown for the K=5 Comp_IC0+RollHL Pareto winner.
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

from src.models.survival_ensemble import GradientBoostingSurvival
from src.features.early_ic import extract_early_ic_features_batch


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return dict(CR=np.nan, AR=np.nan, SR=np.nan, MDD=np.nan, N=0)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r))


def cv_pred_hl(X, y, seed=42, n_splits=5):
    common = X.index.intersection(y.index)
    X = X.loc[common].copy(); y = y.loc[common].copy()
    mask = y["L1_duration"].notna() & y["L1_event"].notna()
    X, y = X.loc[mask], y.loc[mask]
    preds = pd.Series(index=X.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = GradientBoostingSurvival(random_state=seed)
        m.fit(X.iloc[tr], y["L1_duration"].iloc[tr], y["L1_event"].iloc[tr].astype(int))
        preds.iloc[te] = m.predict_median_survival(X.iloc[te])
    return preds.clip(lower=1.0).fillna(preds.median())


def build_rolling_pred_hl(ic_panel, X_full, y, sim_dates, seed=42):
    m = GradientBoostingSurvival(random_state=seed)
    m.fit(X_full, y["L1_duration"], y["L1_event"].astype(int))
    group_b_cols = [c for c in X_full.columns
                    if c.startswith("ic_") or c.startswith("icir_")]
    other_cols = [c for c in X_full.columns if c not in group_b_cols]
    out = pd.DataFrame(index=sim_dates, columns=X_full.index, dtype=float)
    for t in sim_dates:
        cutoff_lo = t - pd.DateOffset(months=12)
        sub = ic_panel[(ic_panel.date > cutoff_lo) & (ic_panel.date <= t)]
        if sub.empty: continue
        disc_map = {fid: cutoff_lo.strftime("%Y-%m-%d") for fid in X_full.index}
        try:
            feat_B = extract_early_ic_features_batch(sub, K_values=[3,6,12], discovery_dates=disc_map)
        except Exception:
            continue
        X_t = pd.concat([feat_B.reindex(X_full.index), X_full[other_cols]], axis=1)
        X_t = X_t.reindex(columns=X_full.columns)
        ok = X_t.notna().all(axis=1)
        if ok.sum() == 0: continue
        try:
            preds_t = m.predict_median_survival(X_t.loc[ok])
            out.loc[t, ok[ok].index] = preds_t
        except Exception:
            continue
    return out


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    labels = pd.read_parquet(DATA / "labels.parquet").set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    # Sign + |IC0| + ICIR_6m
    ic0_abs, signs, icir6 = {}, {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        m_ = g.mean(); s_ = g.std()
        ic0_abs[fid] = abs(m_)
        signs[fid]   = 1.0 if m_ >= 0 else -1.0
        icir6[fid]   = abs(m_) / (s_ if s_ > 0 else np.nan)
    ic0  = pd.Series(ic0_abs)
    ICIR = pd.Series(icir6).dropna()
    print(f"Pool: {len(ic0)} factors  |  ICIR_6m valid: {len(ICIR)}")
    print(f"Spearman corr(|IC0|, ICIR_6m) = {pd.concat([ic0, ICIR], axis=1).corr(method='spearman').iloc[0,1]:+.3f}")

    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp("2012-01-01")]
    print(f"Sim months: {len(sim_dates)}")

    def active_at(t):
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out.append(fid)
        return out

    pred_hl = cv_pred_hl(X, labels)
    rolling_hl = build_rolling_pred_hl(ic, X, labels, sim_dates)

    z_ic0 = (ic0 - ic0.mean()) / ic0.std()
    z_lhl = (np.log1p(pred_hl) - np.log1p(pred_hl).mean()) / np.log1p(pred_hl).std()
    z_icir = (ICIR - ICIR.mean()) / ICIR.std()

    cs_static = z_ic0 + z_lhl
    cs_prod   = ic0 * np.log1p(pred_hl)

    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    def comp_ic0_rollhl(cands, t, K):
        if t not in rolling_hl.index: return topk(ic0, cands, K)
        rhl = rolling_hl.loc[t].reindex(cands)
        z_r = (rhl - rhl.mean()) / (rhl.std() if rhl.std() > 0 else 1)
        score = z_ic0.reindex(cands).fillna(0) + z_r.fillna(0)
        return score.nlargest(min(K, len(score))).index.tolist()

    def ic_momentum_topk(cands, t, K, lookback=3):
        cutoff = t - pd.DateOffset(months=lookback)
        recent = ic[(ic.factor_id.isin(cands)) &
                    (ic.date > cutoff) & (ic.date <= t)]
        if recent.empty: return cands[:K]
        adj = recent.groupby("factor_id").apply(
            lambda g: float((g.ic * signs.get(g.name, 1.0)).mean())
        ).abs()
        return adj.nlargest(min(K, len(adj))).index.tolist()

    def oracle_topk(cands, t, K):
        nxt = t + pd.DateOffset(months=1)
        sub = ic[(ic.factor_id.isin(cands)) & (ic.date == nxt)]
        if sub.empty: return cands[:K]
        v = sub.set_index("factor_id").ic
        v = v * pd.Series({f: signs.get(f, 1) for f in v.index})
        return v.nlargest(min(K, len(v))).index.tolist()

    def comp_ic(t, sel):
        sub = ic[(ic.factor_id.isin(sel)) & (ic.date == t)]
        if sub.empty: return np.nan
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return float(v.mean())

    rules = {
        "Top|IC0|":        lambda c, t, K: topk(ic0, c, K),
        "Top_ICIR6":       lambda c, t, K: topk(ICIR, c, K),
        "ICmom3":          ic_momentum_topk,
        "Composite_static":lambda c, t, K: topk(cs_static, c, K),
        "Composite_prod":  lambda c, t, K: topk(cs_prod, c, K),
        "Comp_IC0+RollHL": comp_ic0_rollhl,
        "Oracle":          oracle_topk,
    }

    rows = []
    cs_vol = 0.10
    for K in [5, 10, 20]:
        print(f"\n=== K={K} ===")
        for name, fn in rules.items():
            rets = {}
            for t in sim_dates:
                cands = active_at(t)
                if len(cands) < K: continue
                sel = fn(cands, t, K)
                if len(sel) < 3: continue
                v = comp_ic(t, sel)
                if not np.isnan(v):
                    rets[t] = v * cs_vol
            r = pd.Series(rets).sort_index()
            mt = metrics(r)
            print(f"  {name:<22} CR={mt['CR']:>+8.2%} AR={mt['AR']:>+7.2%} "
                  f"SR={mt['SR']:>6.3f} MDD={mt['MDD']:>+7.2%} N={mt['N']}")
            rows.append({"K": K, "Rule": name, **mt})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v3_ceiling.csv", index=False)

    # Gap-closed analysis
    print("\n========== Gap closed (rule_AR - base_AR) / (oracle_AR - base_AR) ==========")
    for K in [5, 10, 20]:
        sub = df[df.K == K].set_index("Rule")
        base = sub.loc["Top|IC0|", "AR"]
        ora  = sub.loc["Oracle", "AR"]
        gap  = ora - base
        print(f"K={K}  Top|IC0|={base:+.2%}  Oracle={ora:+.2%}  gap={gap:.2%}")
        for r in ["Top_ICIR6", "ICmom3", "Composite_static",
                   "Composite_prod", "Comp_IC0+RollHL"]:
            ar = sub.loc[r, "AR"]
            cl = (ar - base) / gap if gap > 1e-9 else np.nan
            print(f"  {r:<22} AR={ar:+.2%}  ΔAR={ar-base:+.2%}  closes {cl:>5.1%} of base→oracle gap")

    # Per-year SR for K=5 winners
    print("\n========== Per-year SR breakdown (K=5) ==========")
    yearly = []
    for name in ["Top|IC0|", "Composite_prod", "Comp_IC0+RollHL"]:
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < 5: continue
            sel = rules[name](cands, t, 5)
            v = comp_ic(t, sel)
            if not np.isnan(v):
                rets[t] = v * cs_vol
        r = pd.Series(rets).sort_index()
        r.index = pd.to_datetime(r.index)
        for year, sub in r.groupby(r.index.year):
            if len(sub) >= 6:
                ar  = float((1+sub).prod() - 1)
                sr  = float(sub.mean()/sub.std()*np.sqrt(12)) if sub.std()>0 else np.nan
                mdd = float((((1+sub).cumprod() - (1+sub).cumprod().cummax()) /
                            (1+sub).cumprod().cummax()).min())
                yearly.append({"Year": year, "Rule": name, "AR_year": ar,
                               "SR_year": sr, "MDD_year": mdd, "N": len(sub)})
    ydf = pd.DataFrame(yearly)
    print(ydf.pivot(index="Year", columns="Rule", values="SR_year").round(2).to_string())
    ydf.to_csv(OUT / "anchor_v3_yearly.csv", index=False)


if __name__ == "__main__":
    main()
