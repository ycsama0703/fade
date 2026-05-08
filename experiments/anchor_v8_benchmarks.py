"""
Anchor v8 — broader benchmark comparison.

Adds richer baselines beyond ICmom3+IC0:
  - ICmom6 / ICmom12 (different look-backs)
  - 1/vol-weighted top-K (within selected K, weight by 1/factor_vol)
  - Random-K (control: random selection)
  - Equal-weight buy-and-hold (decided once at sim start, no rebalance)
  - ML stacking baseline: XGBoost regressor on (|IC0|, ICmom3, factor_age, IC_std)
    predicting next-month sign-adj IC, top-K by prediction
  - 4-signal "kitchen-sink" without FADE: ICmom3+ICmom6+ICmom12+IC0
  - 4-signal with FADE: ICmom3+IC0+RollHL+ICmom6 (does extra horizon help?)

All evaluated under strict time-split (cutoff=2017-01) at vol_target=6%.

Output: outputs/anchor_v8_benchmarks.csv
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
                    Vol=np.nan, Calmar=np.nan)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    vol = float(r.std() * np.sqrt(12))
    cal = ar / abs(mdd) if mdd < 0 else np.nan
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r), Vol=vol, Calmar=cal)


def vol_target(r, target=0.06, window=12, l_max=4.0):
    r = r.dropna().sort_index()
    sigma = r.rolling(window).std() * np.sqrt(12)
    L = (target / sigma).clip(upper=l_max).shift(1)
    return (r * L).dropna()


def censored_labels(ic_panel, registry, cutoff_date,
                    initial_window=6, persist_months=6):
    out = []
    cutoff = pd.Timestamp(cutoff_date)
    for _, row in registry.iterrows():
        fid = row["factor_id"]
        disc = pd.Timestamp(row["discovery_date"])
        if disc >= cutoff: continue
        g = ic_panel[(ic_panel.factor_id == fid) & (ic_panel.date >= disc)
                     & (ic_panel.date <= cutoff)].sort_values("date")
        if len(g) < initial_window + persist_months: continue
        ic_vals = g["ic"].values
        early = ic_vals[:initial_window].mean()
        sign  = 1.0 if early >= 0 else -1.0
        signed = ic_vals * sign
        roll = pd.Series(signed).rolling(3).mean()
        L1d, L1e = len(signed), 0
        for k in range(initial_window, len(signed) - persist_months):
            if roll.iloc[k] >= 0: continue
            future = signed[k+1: k+1+persist_months]
            if all(v < 0 for v in future):
                L1d, L1e = k, 1
                break
        out.append({"factor_id": fid, "L1_duration": float(L1d), "L1_event": L1e})
    return pd.DataFrame(out).set_index("factor_id")


def build_rolling_pred_hl(ic_panel, X_full, censored_y, sim_dates, seed=42):
    common = X_full.index.intersection(censored_y.index)
    Xc, yc = X_full.loc[common], censored_y.loc[common]
    m = GradientBoostingSurvival(random_state=seed)
    m.fit(Xc, yc["L1_duration"], yc["L1_event"].astype(int))
    group_b = [c for c in X_full.columns if c.startswith("ic_") or c.startswith("icir_")]
    other   = [c for c in X_full.columns if c not in group_b]
    out = pd.DataFrame(index=sim_dates, columns=X_full.index, dtype=float)
    for t in sim_dates:
        co = t - pd.DateOffset(months=12)
        sub = ic_panel[(ic_panel.date > co) & (ic_panel.date <= t)]
        if sub.empty: continue
        disc_map = {fid: co.strftime("%Y-%m-%d") for fid in X_full.index}
        try:
            feat_B = extract_early_ic_features_batch(sub, K_values=[3,6,12], discovery_dates=disc_map)
        except Exception:
            continue
        X_t = pd.concat([feat_B.reindex(X_full.index), X_full[other]], axis=1)
        X_t = X_t.reindex(columns=X_full.columns)
        ok = X_t.notna().all(axis=1)
        if ok.sum() == 0: continue
        try:
            out.loc[t, ok[ok].index] = m.predict_median_survival(X_t.loc[ok])
        except Exception:
            continue
    return out


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    ic0_abs, signs, ic_vol_per_factor = {}, {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic
        if len(g) < 6: continue
        first6 = g.iloc[:6]
        ic0_abs[fid] = abs(first6.mean())
        signs[fid]   = 1.0 if first6.mean() >= 0 else -1.0
        ic_vol_per_factor[fid] = float(g.std()) if g.std() > 0 else 0.05
    ic0 = pd.Series(ic0_abs)
    factor_vol = pd.Series(ic_vol_per_factor)

    cutoff = "2017-01-01"
    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp(cutoff)]
    print(f"Pool: {len(ic0)} | Cutoff: {cutoff} | Sim: {len(sim_dates)} months")

    print("[censored labels] ...")
    censored = censored_labels(ic, reg, cutoff)
    print(f"  L1 event rate (within cutoff): {censored.L1_event.mean():.1%}")

    print("[rolling pred_HL] ...")
    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)

    def active_at(t):
        out_ = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t or months_between(disc, t) < 6: continue
            out_.append(fid)
        return out_

    def signed_ic(t, fids):
        sub = ic[(ic.factor_id.isin(fids)) & (ic.date == t)]
        if sub.empty: return pd.Series(dtype=float)
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return v

    def ic_lookback_score(t, cands, lookback):
        co = t - pd.DateOffset(months=lookback)
        recent = ic[(ic.factor_id.isin(cands)) & (ic.date > co) & (ic.date <= t)]
        if recent.empty: return pd.Series(dtype=float)
        return recent.groupby("factor_id")["ic"].apply(
            lambda s: float((s * signs.get(s.name, 1.0)).mean())
        ).abs()

    def factor_recent_vol(t, cands, window=12):
        co = t - pd.DateOffset(months=window)
        recent = ic[(ic.factor_id.isin(cands)) & (ic.date > co) & (ic.date <= t)]
        if recent.empty: return pd.Series(1.0, index=cands)
        return recent.groupby("factor_id")["ic"].std().reindex(cands).fillna(0.05)

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def select_combo(cands, t, K, comps):
        z = None
        for fn in comps:
            s = fn(cands, t).reindex(cands)
            zs = zscore(s).fillna(0)
            z = zs if z is None else z + zs
        if z is None: return cands[:K]
        return z.nlargest(min(K, len(z))).index.tolist()

    def composite_return(t, sel, weights=None):
        v = signed_ic(t, sel)
        if v.empty: return np.nan
        if weights is None:
            return float(v.mean())
        w = weights.reindex(v.index).fillna(0)
        return float((v * w).sum() / w.sum()) if w.sum() > 0 else float(v.mean())

    f_ic0    = lambda c, t: ic0.reindex(c)
    f_mom3   = lambda c, t: ic_lookback_score(t, c, 3)
    f_mom6   = lambda c, t: ic_lookback_score(t, c, 6)
    f_mom12  = lambda c, t: ic_lookback_score(t, c, 12)
    f_rollhl = lambda c, t: (rolling_hl.loc[t].reindex(c)
                             if t in rolling_hl.index else pd.Series(dtype=float))

    K = 10
    cs_vol = 0.10

    rules_eq = {
        # singletons (sanity)
        "Top|IC0|":               [f_ic0],
        "ICmom3":                 [f_mom3],
        "ICmom6":                 [f_mom6],
        "ICmom12":                [f_mom12],
        # pairs
        "ICmom3+IC0":             [f_mom3, f_ic0],
        "ICmom6+IC0":             [f_mom6, f_ic0],
        "ICmom12+IC0":            [f_mom12, f_ic0],
        # 3-sig FADE
        "ICmom3+IC0+RollHL":      [f_mom3, f_ic0, f_rollhl],
        "ICmom6+IC0+RollHL":      [f_mom6, f_ic0, f_rollhl],
        # multi-momentum no FADE
        "ICmom3+ICmom6+IC0":      [f_mom3, f_mom6, f_ic0],
        "ICmom3+ICmom6+ICmom12+IC0": [f_mom3, f_mom6, f_mom12, f_ic0],
        # multi-momentum + FADE
        "ICmom3+ICmom6+IC0+RollHL": [f_mom3, f_mom6, f_ic0, f_rollhl],
    }

    raw_rets = {}
    print(f"\nBuilding return series for {len(rules_eq) + 4} rules...")
    for name, comps in rules_eq.items():
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            sel = select_combo(cands, t, K, comps)
            if len(sel) < 3: continue
            v = composite_return(t, sel)
            if not np.isnan(v):
                rets[t] = v * cs_vol
        raw_rets[name] = pd.Series(rets).sort_index()

    # Inverse-vol weighting within top-K (using champion selector)
    rets_invvol = {}
    rets_invvol_fade = {}
    for t in sim_dates:
        cands = active_at(t)
        if len(cands) < K: continue
        sel = select_combo(cands, t, K, [f_mom3, f_ic0])
        sel_f = select_combo(cands, t, K, [f_mom3, f_ic0, f_rollhl])
        rv = factor_recent_vol(t, sel)
        rvf = factor_recent_vol(t, sel_f)
        w_iv = (1.0 / rv).rename("w")
        w_ivf = (1.0 / rvf).rename("w")
        v_iv = composite_return(t, sel, weights=w_iv)
        v_ivf = composite_return(t, sel_f, weights=w_ivf)
        if not np.isnan(v_iv): rets_invvol[t] = v_iv * cs_vol
        if not np.isnan(v_ivf): rets_invvol_fade[t] = v_ivf * cs_vol
    raw_rets["ICmom3+IC0 (inv-vol)"]        = pd.Series(rets_invvol).sort_index()
    raw_rets["ICmom3+IC0+RollHL (inv-vol)"] = pd.Series(rets_invvol_fade).sort_index()

    # Random-K control (10 seeds averaged)
    import random
    rets_rand = {}
    for t in sim_dates:
        cands = active_at(t)
        if len(cands) < K: continue
        vals = []
        for seed in range(10):
            rng = random.Random(seed * 7919 + hash(str(t)) % 99991)
            sel = rng.sample(cands, K)
            v = composite_return(t, sel)
            if not np.isnan(v): vals.append(v)
        if vals: rets_rand[t] = float(np.mean(vals)) * cs_vol
    raw_rets["Random-K"] = pd.Series(rets_rand).sort_index()

    # ML stacking baseline: XGBoost predicting next-month sign-adj IC from
    # cross-sectional features at time t. 5-fold CV by month index.
    print("[ML stacking] training XGBoost predictor of next-month signed IC ...")
    from xgboost import XGBRegressor
    panel_rows = []
    for t in sim_dates[:-1]:
        cands = active_at(t)
        if not cands: continue
        next_t = sim_dates[sim_dates.index(t) + 1] if t in sim_dates else None
        if next_t is None: continue
        nxt = signed_ic(next_t, cands)
        if nxt.empty: continue
        for fid in nxt.index:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            elapsed = months_between(disc, t)
            row = {
                "t": t, "fid": fid,
                "ic0": ic0.get(fid, np.nan),
                "mom3": ic_lookback_score(t, [fid], 3).get(fid, 0),
                "mom6": ic_lookback_score(t, [fid], 6).get(fid, 0),
                "mom12": ic_lookback_score(t, [fid], 12).get(fid, 0),
                "vol_recent": factor_recent_vol(t, [fid]).get(fid, 0.05),
                "rollhl": rolling_hl.loc[t, fid] if t in rolling_hl.index else np.nan,
                "elapsed": elapsed,
                "y": float(nxt[fid]),
            }
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows).dropna()
    print(f"  panel rows: {len(panel)}")
    feat_cols = ["ic0","mom3","mom6","mom12","vol_recent","rollhl","elapsed"]
    if len(panel) > 50:
        # Time-ordered 5-fold (earliest 80% train, latest 20% test, rolling)
        # Simpler: by date, use first 70% as train.
        unique_t = sorted(panel.t.unique())
        split_t = unique_t[int(len(unique_t)*0.5)]
        train = panel[panel.t < split_t]
        test  = panel[panel.t >= split_t]
        m_xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, tree_method="exact",
                             random_state=42, n_jobs=1, verbosity=0)
        m_xgb.fit(train[feat_cols], train["y"])
        pred = m_xgb.predict(panel[feat_cols])
        panel["pred"] = pred

        # Build returns: at each t in test period, top-K by pred
        rets_ml = {}
        for t in sorted(test.t.unique()):
            sub = panel[panel.t == t]
            if len(sub) < K: continue
            sel = sub.nlargest(K, "pred")["fid"].tolist()
            v = composite_return(t, sel)
            if not np.isnan(v):
                rets_ml[t] = v * cs_vol
        raw_rets["ML_stack_XGB (test half)"] = pd.Series(rets_ml).sort_index()

    # Equal-weight buy-hold (decided at sim start, no rebalance)
    cands0 = active_at(sim_dates[0])
    sel0   = select_combo(cands0, sim_dates[0], K, [f_mom3, f_ic0])
    rets_bh = {}
    for t in sim_dates:
        v = composite_return(t, sel0)
        if not np.isnan(v): rets_bh[t] = v * cs_vol
    raw_rets["BuyHold (top-K @start)"] = pd.Series(rets_bh).sort_index()

    # ---- Print results ----
    print(f"\n=== K={K}, cutoff={cutoff}, RAW ===")
    rows = []
    for name, r in raw_rets.items():
        m = metrics(r)
        rows.append({"Variant": "raw", "Rule": name, **m})
    rows_df = pd.DataFrame(rows).sort_values("SR", ascending=False)
    for _, row in rows_df.iterrows():
        print(f"  {row.Rule:<32} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
              f"MDD={row.MDD:>+7.2%} Vol={row.Vol:.2%} Calmar={row.Calmar:.2f}")

    for tgt in [0.04, 0.06]:
        print(f"\n=== K={K}, VOL-TARGETED to {tgt*100:.0f}% ===")
        for name, r in raw_rets.items():
            r_vt = vol_target(r, target=tgt)
            m = metrics(r_vt)
            rows.append({"Variant": f"vt_{tgt:.2f}", "Rule": name, **m})
        sub = pd.DataFrame([row for row in rows if row["Variant"] == f"vt_{tgt:.2f}"]).sort_values("SR", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row.Rule:<32} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
                  f"MDD={row.MDD:>+7.2%} Calmar={row.Calmar:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v8_benchmarks.csv", index=False)

    # Pareto vs ICmom3+IC0 (no FADE) at vt=6%
    print("\n========== Pareto vs ICmom3+IC0 baseline (vt=6%) ==========")
    sub = df[df.Variant == "vt_0.06"].set_index("Rule")
    if "ICmom3+IC0" in sub.index:
        b = sub.loc["ICmom3+IC0"]
        print(f"Baseline ICmom3+IC0: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%} Cal={b.Calmar:.2f}")
        print()
        for name, row in sub.sort_values("SR", ascending=False).iterrows():
            if name == "ICmom3+IC0": continue
            dar  = row.AR  - b.AR
            dsr  = row.SR  - b.SR
            dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0:
                tag = "  ★ PARETO"
            elif dar > 0 and dsr > 0:
                tag = "  ◇ AR+SR"
            elif dsr > 0 and dmdd > 0:
                tag = "  ◇ SR+MDD"
            print(f"  {name:<32} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
