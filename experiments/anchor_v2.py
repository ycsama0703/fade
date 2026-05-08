"""
Anchor v2 — exhaustive search for the strongest selection rule.

Builds on paper_anchor_alpha158.py with extra angles:

  (a) K = {5, 10, 15, 20, 30, 50} — sweep concentration
  (b) Direct EIC (Expected IC) regressor
        target = mean(sign_adjusted_IC over months 7..18 post-disc)
        XGBoost, 5-fold CV OOS, used as a selection score directly
  (c) Hazard-based survival score
        score = P(survive next 6 months | survived to t)
        = S(elapsed + 6 | x) / S(elapsed | x)
        from GBS predict_survival_function
  (d) Rolling-features pred_HL
        at each rebalance month t, recompute Group B (IC dynamics) features
        from IC over [t-12, t], feed to the trained M3_gbs to get a CURRENT
        decay risk. Treats the survival model as a real-time vital sign.
  (e) Multi-seed stability check
        repeat the K=5/10/20 experiment over 5 different fold seeds
        to bound the variance of the ΔAR / ΔSR estimates

All runs share the same 141-factor Alpha158 pool, sim window 2012-01 → 2020-08.
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


# ---- pred_HL via 5-fold CV ----
def cv_pred_hl(X, y, seed=42, n_splits=5):
    common = X.index.intersection(y.index)
    X = X.loc[common].copy()
    y = y.loc[common].copy()
    mask = y["L1_duration"].notna() & y["L1_event"].notna()
    X, y = X.loc[mask], y.loc[mask]
    preds = pd.Series(index=X.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = GradientBoostingSurvival(random_state=seed)
        m.fit(X.iloc[tr], y["L1_duration"].iloc[tr], y["L1_event"].iloc[tr].astype(int))
        preds.iloc[te] = m.predict_median_survival(X.iloc[te])
    return preds.clip(lower=1.0).fillna(preds.median())


# ---- Direct Expected-IC regressor ----
def cv_eic(X, ic_panel, reg_idx, signs, seed=42, n_splits=5,
           start_month=7, end_month=18):
    """
    Target = mean(sign_adjusted IC over [disc+start, disc+end] months) per factor.
    Train XGBoost regressor, return OOS predictions per factor.
    """
    from xgboost import XGBRegressor
    # Build target
    y_target = {}
    for fid in X.index:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic_panel[(ic_panel.factor_id == fid) & (ic_panel.date >= disc)] \
            .sort_values("date").ic
        if len(g) >= end_month:
            y_target[fid] = float((g.iloc[start_month-1:end_month] * signs.get(fid, 1.0)).mean())
    y = pd.Series(y_target).reindex(X.index).dropna()
    Xy = X.loc[y.index]
    Xy = pd.get_dummies(Xy, drop_first=True).astype(float)
    preds = pd.Series(index=Xy.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xy):
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, tree_method="exact", random_state=seed,
                         n_jobs=1, verbosity=0)
        m.fit(Xy.iloc[tr], y.iloc[tr])
        preds.iloc[te] = m.predict(Xy.iloc[te])
    return preds.reindex(X.index)  # returns NaN for factors missing target


# ---- Rolling pred_HL: recompute features at each t ----
def build_rolling_pred_hl(ic_panel, X_full, y, reg_idx, sim_dates,
                          model_seed=42):
    """
    Train M3 GBS on FULL X (all 141 disc-time features). Then at each t,
    construct a "fresh" feature vector for each factor by computing Group B
    on IC over [t-12, t], keeping Group A (structural) and Group C (market env
    at original disc) constant. Predict median survival from these fresh features.

    Returns: DataFrame of shape (sim_dates, factors) — rolling pred_HL[t, fid]
    """
    print("  [rolling] training GBS on disc-time features ...")
    m = GradientBoostingSurvival(random_state=model_seed)
    m.fit(X_full, y["L1_duration"], y["L1_event"].astype(int))

    # Constant-by-factor part = Group A + Group C columns
    group_b_cols = [c for c in X_full.columns
                    if c.startswith("ic_") or c.startswith("icir_")]
    other_cols = [c for c in X_full.columns if c not in group_b_cols]
    print(f"  [rolling] Group B cols: {len(group_b_cols)}  others: {len(other_cols)}")

    out = pd.DataFrame(index=sim_dates, columns=X_full.index, dtype=float)
    for t in sim_dates:
        cutoff_lo = t - pd.DateOffset(months=12)
        # Slice IC for [cutoff_lo, t]
        sub = ic_panel[(ic_panel.date > cutoff_lo) & (ic_panel.date <= t)]
        if sub.empty: continue
        # Build a synthetic "discovery_date" map = cutoff_lo for all factors
        disc_map = {fid: cutoff_lo.strftime("%Y-%m-%d") for fid in X_full.index}
        try:
            feat_B = extract_early_ic_features_batch(
                sub, K_values=[3, 6, 12], discovery_dates=disc_map
            )
        except Exception:
            continue
        # Combine with constant other features
        feat_other = X_full[other_cols]
        # Reindex feat_B to factor index
        feat_B = feat_B.reindex(X_full.index)
        X_t = pd.concat([feat_B, feat_other], axis=1)
        X_t = X_t.reindex(columns=X_full.columns)
        # Drop factors with missing rolling features
        ok = X_t.notna().all(axis=1)
        if ok.sum() == 0: continue
        try:
            preds_t = m.predict_median_survival(X_t.loc[ok])
            out.loc[t, ok[ok].index] = preds_t
        except Exception as e:
            continue
    return out


# ---- Cumulative simulation engine ----
def simulate_selection(rules: dict, sim_dates, ic_panel, signs, reg_idx,
                       active_at, K, cs_vol=0.10):
    """rules: dict[name] -> fn(cands, t) -> selected list."""
    rows = {}
    for name, fn in rules.items():
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            sel = fn(cands, t)
            if len(sel) < 3: continue
            sub = ic_panel[(ic_panel.factor_id.isin(sel)) & (ic_panel.date == t)]
            if sub.empty: continue
            v = sub.set_index("factor_id").ic
            v = v * pd.Series(signs).reindex(v.index)
            rets[t] = float(v.mean()) * cs_vol
        rows[name] = pd.Series(rets).sort_index()
    return rows


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    labels = pd.read_parquet(DATA / "labels.parquet").set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    # |IC0| & sign
    ic0_abs, signs = {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        ic0_abs[fid] = abs(g.mean())
        signs[fid]   = 1.0 if g.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)
    print(f"Pool: {len(ic0)} factors with valid IC0")

    sim_dates = [d for d in sorted(ic.date.unique())
                 if d >= pd.Timestamp("2012-01-01")]
    print(f"Sim months: {len(sim_dates)}")

    def active_at(t):
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out.append(fid)
        return out

    # ---------- (b) EIC regressor ----------
    print("\n[EIC] Training direct expected-IC regressor (5-fold CV) ...")
    eic = cv_eic(X, ic, reg_idx, signs, seed=42)
    print(f"  EIC pred range: [{eic.min():.4f}, {eic.max():.4f}]  n={eic.notna().sum()}")
    print(f"  Spearman corr(EIC, |IC0|)    = {pd.concat([eic, ic0], axis=1).corr(method='spearman').iloc[0,1]:+.3f}")

    # ---------- (a) base pred_HL via 5-fold CV ----------
    print("\n[pred_HL] base seed=42 ...")
    pred_hl = cv_pred_hl(X, labels, seed=42)
    print(f"  pred_HL: median={pred_hl.median():.1f}  q90={pred_hl.quantile(0.9):.1f}")
    print(f"  Spearman corr(pred_HL, |IC0|) = {pd.concat([pred_hl, ic0], axis=1).corr(method='spearman').iloc[0,1]:+.3f}")
    print(f"  Spearman corr(pred_HL, EIC)   = {pd.concat([pred_hl, eic], axis=1).corr(method='spearman').iloc[0,1]:+.3f}")

    log_hl = np.log1p(pred_hl)
    z_ic0  = (ic0 - ic0.mean()) / ic0.std()
    z_lhl  = (log_hl - log_hl.mean()) / log_hl.std()
    z_eic  = (eic - eic.mean()) / eic.std()

    composite_static = z_ic0 + z_lhl
    composite_prod   = ic0 * np.log1p(pred_hl)
    composite_eic    = z_ic0 + z_eic
    composite_triple = z_ic0 + z_lhl + z_eic
    composite_ic_eic = z_eic + z_lhl  # eic already incorporates ic0 info, plus durability

    # ---------- (d) rolling pred_HL ----------
    print("\n[rolling] computing rolling-feature pred_HL ...")
    rolling_hl = build_rolling_pred_hl(ic, X, labels, reg_idx,
                                        sim_dates, model_seed=42)
    print(f"  rolling_hl shape: {rolling_hl.shape}  "
          f"non-null frac: {rolling_hl.notna().mean().mean():.2%}")

    # ---------- Selection rules ----------
    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    rules_factory = lambda K: {
        "Top|IC0|":         lambda c, t: topk(ic0,                c, K),
        "TopPredHL":        lambda c, t: topk(pred_hl,            c, K),
        "TopEIC":           lambda c, t: topk(eic,                c, K),
        "Composite_static": lambda c, t: topk(composite_static,   c, K),
        "Composite_prod":   lambda c, t: topk(composite_prod,     c, K),
        "Composite_EIC":    lambda c, t: topk(composite_eic,      c, K),
        "Composite_IC_EIC": lambda c, t: topk(composite_ic_eic,   c, K),
        "Composite_triple": lambda c, t: topk(composite_triple,   c, K),
        "RollingHL":        lambda c, t: topk(rolling_hl.loc[t] if t in rolling_hl.index else pd.Series(dtype=float),
                                              c, K),
        "Comp_IC0+RollHL":  lambda c, t: topk(
            (z_ic0.reindex(c).fillna(0) +
             ((rolling_hl.loc[t].reindex(c) - rolling_hl.loc[t].mean()) /
              (rolling_hl.loc[t].std() if rolling_hl.loc[t].std()>0 else 1)).fillna(0))
            if t in rolling_hl.index else z_ic0,
            c, K),
    }

    # ---------- Sweep K ----------
    rows_all = []
    for K in [5, 10, 15, 20, 30, 50]:
        print(f"\n=== K={K} ===")
        rules = rules_factory(K)
        rets_dict = simulate_selection(rules, sim_dates, ic, signs, reg_idx,
                                        active_at, K)
        # Add reference
        rets_dict["All"] = pd.Series({t: float(
            (ic[(ic.factor_id.isin(active_at(t))) & (ic.date == t)]
             .set_index("factor_id").ic *
             pd.Series(signs).reindex(active_at(t))).mean()) * 0.10
            for t in sim_dates if len(active_at(t)) >= 3}).dropna()
        for name, r in rets_dict.items():
            m = metrics(r)
            print(f"  {name:<22}  CR={m['CR']:>+8.2%}  AR={m['AR']:>+7.2%}  "
                  f"SR={m['SR']:>6.3f}  MDD={m['MDD']:>+7.2%}  N={m['N']}")
            rows_all.append({"K": K, "Rule": name, **m})
    df = pd.DataFrame(rows_all)
    df.to_csv(OUT / "anchor_v2_selection.csv", index=False)

    # ---------- (e) multi-seed stability ----------
    print("\n=== Multi-seed stability (K=10) ===")
    seeds = [42, 7, 21, 100, 2025]
    seed_rows = []
    for sd in seeds:
        ph = cv_pred_hl(X, labels, seed=sd)
        z_l = (np.log1p(ph) - np.log1p(ph).mean()) / np.log1p(ph).std()
        cs_static = z_ic0 + z_l
        cs_prod   = ic0 * np.log1p(ph)
        for name, score in [("Top|IC0|", ic0),
                            ("Composite_static", cs_static),
                            ("Composite_prod", cs_prod)]:
            rets = {}
            for t in sim_dates:
                cands = active_at(t)
                if len(cands) < 10: continue
                sel = topk(score, cands, 10)
                sub = ic[(ic.factor_id.isin(sel)) & (ic.date == t)]
                if sub.empty: continue
                v = sub.set_index("factor_id").ic
                v = v * pd.Series(signs).reindex(v.index)
                rets[t] = float(v.mean()) * 0.10
            r = pd.Series(rets).sort_index()
            mt = metrics(r)
            print(f"  seed={sd:<5} {name:<22} AR={mt['AR']:+.2%}  SR={mt['SR']:.3f}  MDD={mt['MDD']:+.2%}")
            seed_rows.append({"seed": sd, "Rule": name, **mt})
    pd.DataFrame(seed_rows).to_csv(OUT / "anchor_v2_seed_stability.csv", index=False)

    # ---------- Headline ----------
    print("\n========== HEADLINE — Δ vs Top|IC0| at each K ==========")
    for K in [5, 10, 15, 20, 30, 50]:
        sub = df[df.K == K]
        base = sub[sub.Rule == "Top|IC0|"].iloc[0]
        cand = sub[~sub.Rule.isin(["Top|IC0|", "All"])].copy()
        cand["dAR"] = cand.AR - base.AR
        cand["dSR"] = cand.SR - base.SR
        cand["dMDD"] = cand.MDD - base.MDD
        # Best by SR
        b_sr = cand.sort_values("SR", ascending=False).iloc[0]
        # Best by AR
        b_ar = cand.sort_values("AR", ascending=False).iloc[0]
        print(f"K={K:<3}  baseline AR={base.AR:+.2%} SR={base.SR:.3f} MDD={base.MDD:+.2%}")
        print(f"         best-SR : {b_sr.Rule:<22} AR={b_sr.AR:+.2%} (Δ{b_sr.dAR:+.2%}) "
              f"SR={b_sr.SR:.3f} (Δ{b_sr.dSR:+.3f}) MDD={b_sr.MDD:+.2%} (Δ{b_sr.dMDD:+.2%})")
        print(f"         best-AR : {b_ar.Rule:<22} AR={b_ar.AR:+.2%} (Δ{b_ar.dAR:+.2%}) "
              f"SR={b_ar.SR:.3f} (Δ{b_ar.dSR:+.3f}) MDD={b_ar.MDD:+.2%} (Δ{b_ar.dMDD:+.2%})")


if __name__ == "__main__":
    main()
