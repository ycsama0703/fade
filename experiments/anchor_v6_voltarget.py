"""
Anchor v6 — vol-target the rules to a common risk level, then compare AR.

Each rule produces a monthly return series r_t. We compute a 12-month rolling
realized vol sigma_t = std(r_{t-12..t}) and apply leverage L_t = target_vol / sigma_t
(capped at L_max). The vol-targeted return is r_t * L_{t-1} (using prior month's
estimate to avoid lookahead).

After matching all rules to the same risk level, AR comparison is honest:
which rule has the highest risk-adjusted return.

Also produces per-year SR/MDD breakdown for the K=10 leading rules.
"""
from __future__ import annotations
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT  = ROOT / "outputs"

from src.models.survival_ensemble import GradientBoostingSurvival
from src.features.early_ic import extract_early_ic_features_batch


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return dict(CR=np.nan, AR=np.nan, SR=np.nan, MDD=np.nan, N=0,
                    AnnualVol=np.nan)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    av  = float(r.std() * np.sqrt(12))
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r), AnnualVol=av)


def vol_target(r: pd.Series, target_vol: float = 0.06,
               window: int = 12, l_max: float = 4.0) -> pd.Series:
    """Apply rolling-vol leverage to hit `target_vol`. Lag leverage by 1 month."""
    r = r.dropna().sort_index()
    sigma = r.rolling(window).std() * np.sqrt(12)  # annualised
    L = (target_vol / sigma).clip(upper=l_max).shift(1)  # lag to avoid lookahead
    return (r * L).dropna()


def cv_pred_hl(X, y, seed=42, n_splits=5):
    common = X.index.intersection(y.index)
    X = X.loc[common]; y = y.loc[common]
    mask = y["L1_duration"].notna() & y["L1_event"].notna()
    X, y = X.loc[mask], y.loc[mask]
    preds = pd.Series(index=X.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = GradientBoostingSurvival(random_state=seed)
        m.fit(X.iloc[tr], y["L1_duration"].iloc[tr], y["L1_event"].iloc[tr].astype(int))
        preds.iloc[te] = m.predict_median_survival(X.iloc[te])
    return preds.clip(lower=1.0).fillna(preds.median())


def cv_eic(X, ic_panel, reg_idx, signs, seed=42, n_splits=5,
           start_month=7, end_month=18):
    from xgboost import XGBRegressor
    y_target = {}
    for fid in X.index:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic_panel[(ic_panel.factor_id == fid) & (ic_panel.date >= disc)] \
            .sort_values("date").ic
        if len(g) >= end_month:
            y_target[fid] = float((g.iloc[start_month-1:end_month] * signs.get(fid, 1.0)).mean())
    y = pd.Series(y_target).reindex(X.index).dropna()
    Xy = pd.get_dummies(X.loc[y.index], drop_first=True).astype(float)
    preds = pd.Series(index=Xy.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xy):
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, tree_method="exact", random_state=seed,
                         n_jobs=1, verbosity=0)
        m.fit(Xy.iloc[tr], y.iloc[tr])
        preds.iloc[te] = m.predict(Xy.iloc[te])
    return preds.reindex(X.index)


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

    ic0_abs, signs = {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        ic0_abs[fid] = abs(g.mean())
        signs[fid]   = 1.0 if g.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)
    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp("2012-01-01")]
    print(f"Pool: {len(ic0)} | Sim months: {len(sim_dates)}")

    print("[CV] pred_HL ...")
    pred_hl = cv_pred_hl(X, labels)
    print("[CV] EIC ...")
    eic = cv_eic(X, ic, reg_idx, signs)
    print("[rolling] pred_HL ...")
    rolling_hl = build_rolling_pred_hl(ic, X, labels, sim_dates)

    def active_at(t):
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out.append(fid)
        return out

    def comp_ic(t, sel):
        sub = ic[(ic.factor_id.isin(sel)) & (ic.date == t)]
        if sub.empty: return np.nan
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return float(v.mean())

    def icmom_score(t, cands, lookback=3):
        cutoff = t - pd.DateOffset(months=lookback)
        recent = ic[(ic.factor_id.isin(cands)) &
                    (ic.date > cutoff) & (ic.date <= t)]
        if recent.empty: return pd.Series(dtype=float)
        adj = recent.groupby("factor_id")["ic"].apply(
            lambda s: float((s * signs.get(s.name, 1.0)).mean())
        ).abs()
        return adj

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def select_combo(cands, t, K, components):
        z_total = None
        for fn in components:
            s = fn(cands, t).reindex(cands)
            zs = zscore(s).fillna(0)
            z_total = zs if z_total is None else z_total + zs
        if z_total is None: return cands[:K]
        return z_total.nlargest(min(K, len(z_total))).index.tolist()

    f_ic0    = lambda c, t: ic0.reindex(c)
    f_icmom  = lambda c, t: icmom_score(t, c)
    f_predhl = lambda c, t: pred_hl.reindex(c)
    f_eic    = lambda c, t: eic.reindex(c)
    f_rollhl = lambda c, t: (rolling_hl.loc[t].reindex(c)
                             if t in rolling_hl.index else pd.Series(dtype=float))

    rule_def = {
        "Top|IC0|":          [f_ic0],
        "ICmom3":            [f_icmom],
        "ICmom3+IC0":        [f_icmom, f_ic0],
        "ICmom3+IC0+EIC":    [f_icmom, f_ic0, f_eic],
        "ICmom3+IC0+PredHL": [f_icmom, f_ic0, f_predhl],
        "ICmom3+IC0+RollHL": [f_icmom, f_ic0, f_rollhl],
    }

    cs_vol = 0.10
    K = 10  # focus on the most informative concentration

    # ---------- Compute raw return series for each rule ----------
    raw_rets = {}
    for name, components in rule_def.items():
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            sel = select_combo(cands, t, K, components)
            if len(sel) < 3: continue
            v = comp_ic(t, sel)
            if not np.isnan(v):
                rets[t] = v * cs_vol
        raw_rets[name] = pd.Series(rets).sort_index()

    # ---------- Headline: raw vs vol-targeted ----------
    print(f"\n=== K={K}, RAW (no vol target) ===")
    raw_rows = []
    for name, r in raw_rets.items():
        m = metrics(r)
        print(f"  {name:<22} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
              f"MDD={m['MDD']:>+7.2%} Vol_ann={m['AnnualVol']:>5.2%}")
        raw_rows.append({"Rule": name, "Variant": "raw", **m})

    for tgt_vol in [0.04, 0.06, 0.08, 0.10]:
        print(f"\n=== K={K}, VOL-TARGETED to {tgt_vol*100:.0f}% annual ===")
        for name, r in raw_rets.items():
            r_vt = vol_target(r, target_vol=tgt_vol)
            m = metrics(r_vt)
            print(f"  {name:<22} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
                  f"MDD={m['MDD']:>+7.2%} Vol_ann={m['AnnualVol']:>5.2%}")
            raw_rows.append({"Rule": name, "Variant": f"vt_{tgt_vol:.2f}", **m})

    pd.DataFrame(raw_rows).to_csv(OUT / "anchor_v6_voltarget.csv", index=False)

    # ---------- Per-year SR/AR/MDD breakdown ----------
    print(f"\n========== Per-year breakdown (K={K}, raw returns) ==========")
    yearly = []
    for name, r in raw_rets.items():
        r.index = pd.to_datetime(r.index)
        for year, sub in r.groupby(r.index.year):
            if len(sub) >= 6:
                cum = (1 + sub).cumprod()
                ar = float(cum.iloc[-1] - 1)
                sr = float(sub.mean() / sub.std() * np.sqrt(12)) if sub.std() > 0 else np.nan
                mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
                yearly.append({"Year": year, "Rule": name,
                               "Year_AR": ar, "Year_SR": sr, "Year_MDD": mdd})
    ydf = pd.DataFrame(yearly)
    print("\n--- Year × Rule SR ---")
    print(ydf.pivot(index="Year", columns="Rule", values="Year_SR").round(2).to_string())
    print("\n--- Year × Rule MDD ---")
    print(ydf.pivot(index="Year", columns="Rule", values="Year_MDD").round(3).to_string())
    print("\n--- Year × Rule AR ---")
    print(ydf.pivot(index="Year", columns="Rule", values="Year_AR").round(3).to_string())
    ydf.to_csv(OUT / "anchor_v6_yearly.csv", index=False)

    # ---------- Drawdown-averse score: AR / |MDD| ----------
    print("\n========== Calmar (AR/|MDD|) at each vol target ==========")
    df = pd.DataFrame(raw_rows)
    df["Calmar"] = df.AR / df.MDD.abs()
    pivot = df.pivot(index="Rule", columns="Variant", values="Calmar").round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
