"""
Anchor v7 — strict time-split OOS validation.

The v6 result used pred_HL labels computed from the FULL 2010-2020 IC panel.
For sim months 2012-2020, this means the model implicitly knew which factors
would eventually decay — partial time-leakage.

Strict time-split:
  cutoff_date C = 2017-01-01
  - Compute labels using only IC[0..C]: durations censored at C-disc.
  - Train M3 GBS on this censored panel (5-fold CV at factor level).
  - Sim window: C..end (2017-01 → 2020-08, 44 months — half what v6 used).
  - All feature computations identical to v6.

If Pareto domination of ICmom3+IC0+PredHL over ICmom3+IC0 still holds, the
paper claim is bulletproof. If not, we honestly report shrinkage.

Also tests cutoff = 2015-01 for a longer sim window cross-check.
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
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r),
                AnnualVol=float(r.std() * np.sqrt(12)))


def vol_target(r, target_vol=0.06, window=12, l_max=4.0):
    r = r.dropna().sort_index()
    sigma = r.rolling(window).std() * np.sqrt(12)
    L = (target_vol / sigma).clip(upper=l_max).shift(1)
    return (r * L).dropna()


def censored_labels(ic_panel, registry, cutoff_date,
                    initial_window=6, persist_months=6):
    """Recompute L1_duration / L1_event using only IC data through cutoff_date.

    For each factor:
      - sign = sign of mean of first 6 months of IC after disc
      - rolling-3m mean of sign-adjusted IC (called s)
      - L1_event=1, L1_duration=k if there is a month k where s[k] < 0 AND
        next persist_months months also have negative s, AND k < cutoff months
        post-disc.
      - Otherwise L1_event=0, L1_duration = months from disc to cutoff
    """
    out = []
    cutoff = pd.Timestamp(cutoff_date)
    for _, row in registry.iterrows():
        fid = row["factor_id"]
        disc = pd.Timestamp(row["discovery_date"])
        if disc >= cutoff:
            continue
        g = ic_panel[(ic_panel.factor_id == fid) & (ic_panel.date >= disc)
                     & (ic_panel.date <= cutoff)].sort_values("date")
        if len(g) < initial_window + persist_months:
            continue
        ic_vals = g["ic"].values
        early_mean = ic_vals[:initial_window].mean()
        sign = 1.0 if early_mean >= 0 else -1.0
        signed = ic_vals * sign
        # rolling-3m mean
        roll = pd.Series(signed).rolling(3).mean()
        # find first month where roll < 0 AND next persist_months are all < 0
        L1_dur = len(signed)
        L1_evt = 0
        for k in range(initial_window, len(signed) - persist_months):
            if roll.iloc[k] >= 0:
                continue
            future = signed[k+1: k+1+persist_months]
            if all(v < 0 for v in future):
                L1_dur = k
                L1_evt = 1
                break
        out.append({"factor_id": fid, "L1_duration": float(L1_dur),
                    "L1_event": L1_evt})
    return pd.DataFrame(out).set_index("factor_id")


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


def cv_eic(X, ic_panel, reg_idx, signs, cutoff,
           seed=42, n_splits=5, start_month=7, end_month=18):
    """EIC target also uses only data up to cutoff."""
    from xgboost import XGBRegressor
    cutoff = pd.Timestamp(cutoff)
    y_target = {}
    for fid in X.index:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        # Need disc + end_month <= cutoff
        if (disc + pd.DateOffset(months=end_month)) >= cutoff:
            continue
        g = ic_panel[(ic_panel.factor_id == fid) & (ic_panel.date >= disc)
                     & (ic_panel.date <= cutoff)].sort_values("date").ic
        if len(g) >= end_month:
            y_target[fid] = float((g.iloc[start_month-1:end_month] * signs.get(fid, 1.0)).mean())
    y = pd.Series(y_target).reindex(X.index).dropna()
    if len(y) < 30:
        return pd.Series(np.nan, index=X.index)
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


def build_rolling_pred_hl(ic_panel, X_full, censored_y, sim_dates, seed=42):
    """Train GBS on CENSORED labels only (no future leak), then predict at each t."""
    common = X_full.index.intersection(censored_y.index)
    Xc = X_full.loc[common]
    yc = censored_y.loc[common]
    m = GradientBoostingSurvival(random_state=seed)
    m.fit(Xc, yc["L1_duration"], yc["L1_event"].astype(int))
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


def run_for_cutoff(cutoff_date: str, ic, reg, reg_idx, X, signs, ic0):
    print(f"\n{'='*60}")
    print(f"  STRICT TIME-SPLIT — cutoff = {cutoff_date}")
    print(f"  Sim window: {cutoff_date} → end")
    print(f"{'='*60}")

    cutoff = pd.Timestamp(cutoff_date)
    censored = censored_labels(ic, reg, cutoff_date)
    print(f"  Censored labels: {len(censored)} factors")
    print(f"  L1 event rate within cutoff: {censored.L1_event.mean():.1%}")

    # Sim dates strictly after cutoff
    sim_dates = [d for d in sorted(ic.date.unique()) if d >= cutoff]
    if len(sim_dates) < 12:
        print(f"  WARN: only {len(sim_dates)} sim months — insufficient.")
        return None
    print(f"  Sim months: {len(sim_dates)} ({sim_dates[0].date()} → {sim_dates[-1].date()})")

    # OOS pred_HL via 5-fold CV on CENSORED labels
    pred_hl = cv_pred_hl(X, censored)
    eic     = cv_eic(X, ic, reg_idx, signs, cutoff_date)
    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)

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
        co = t - pd.DateOffset(months=lookback)
        recent = ic[(ic.factor_id.isin(cands)) &
                    (ic.date > co) & (ic.date <= t)]
        if recent.empty: return pd.Series(dtype=float)
        return recent.groupby("factor_id")["ic"].apply(
            lambda s: float((s * signs.get(s.name, 1.0)).mean())
        ).abs()

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def select_combo(cands, t, K, comps):
        z_total = None
        for fn in comps:
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
    K = 10

    raw_rets = {}
    for name, comps in rule_def.items():
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            sel = select_combo(cands, t, K, comps)
            if len(sel) < 3: continue
            v = comp_ic(t, sel)
            if not np.isnan(v):
                rets[t] = v * cs_vol
        raw_rets[name] = pd.Series(rets).sort_index()

    print(f"\n  RAW (K={K}):")
    rows = []
    for name, r in raw_rets.items():
        m = metrics(r)
        print(f"    {name:<22} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
              f"MDD={m['MDD']:>+7.2%} Vol={m['AnnualVol']:.2%}")
        rows.append({"cutoff": cutoff_date, "Rule": name, "Variant": "raw", **m})

    for tgt in [0.04, 0.06]:
        print(f"\n  VOL-TARGETED to {tgt*100:.0f}%:")
        for name, r in raw_rets.items():
            r_vt = vol_target(r, target_vol=tgt)
            m = metrics(r_vt)
            print(f"    {name:<22} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
                  f"MDD={m['MDD']:>+7.2%} Vol={m['AnnualVol']:.2%}")
            rows.append({"cutoff": cutoff_date, "Rule": name,
                          "Variant": f"vt_{tgt:.2f}", **m})

    # Pareto check vs ICmom3+IC0
    print(f"\n  Pareto vs ICmom3+IC0 (vt=6%):")
    df = pd.DataFrame(rows)
    sub = df[(df.cutoff == cutoff_date) & (df.Variant == "vt_0.06")].set_index("Rule")
    if "ICmom3+IC0" in sub.index:
        b = sub.loc["ICmom3+IC0"]
        for name, row in sub.iterrows():
            if name == "ICmom3+IC0": continue
            dar = row.AR - b.AR
            dsr = row.SR - b.SR
            dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0:
                tag = "  ★ PARETO-DOMINATES"
            elif dsr > 0 and dmdd > 0:
                tag = "  ◇ SR+MDD up"
            print(f"    {name:<22} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")

    return rows


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    ic0_abs, signs = {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        ic0_abs[fid] = abs(g.mean())
        signs[fid]   = 1.0 if g.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)

    all_rows = []
    for cutoff in ["2015-01-01", "2017-01-01"]:
        rows = run_for_cutoff(cutoff, ic, reg, reg_idx, X, signs, ic0)
        if rows: all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(OUT / "anchor_v7_strict_oos.csv", index=False)
    print("\nSaved → outputs/anchor_v7_strict_oos.csv")


if __name__ == "__main__":
    main()
