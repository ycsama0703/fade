"""
Anchor v13 — direct prediction of forward ICIR with purged temporal CV.

Theoretical motivation:
  ICIR_12m baseline picks top-K by trailing 12m ICIR. This is a naive
  estimator of forward ICIR (just plug-in trailing). Any consistent
  predictor of forward ICIR using MORE features should weakly improve
  in MSE. The portfolio question is whether the improvement in MSE
  translates to improvement in top-K selection rank.

New features we add that ICIR cannot see:
  - Cross-factor crowding (factor's avg/max abs corr with peers, last 12m)
  - Cross-factor IC dispersion (regime proxy)
  - Factor age (months since disc)
  - Sign-stability (pct of months with sign match)
  - Multi-window IC stats (3m / 6m / 12m means, stds, ICIRs, trends, drawdowns)

Methodology (López de Prado purged + embargoed temporal CV):
  Train  anchors:  t < 2015-06-01     (5+ years)
  Embargo gap   :  ~7 months
  Val    anchors:  2016-01-01 .. 2016-09-01
  Embargo gap   :  ~6 months
  Test   anchors:  2017-01-01 .. 2020-02-01  (forward 6m target needs to fit)

Compare to:
  ICIR_12m_invvol (industry SOTA)
  MVO+LW (mu=6m) (the strongest baseline from v11)
  FADE (RollHL) — our prior method (already known to lose vs SOTA)
"""
from __future__ import annotations
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf
from xgboost import XGBRegressor

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT  = ROOT / "outputs"


def months_between(d1, d2): return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def metrics(r):
    r = r.dropna()
    if len(r) == 0: return dict(AR=np.nan, SR=np.nan, MDD=np.nan, Vol=np.nan, N=0, Calmar=np.nan)
    cum = (1 + r).cumprod()
    cr = float(cum.iloc[-1] - 1)
    ar = float((1 + cr) ** (12 / len(r)) - 1)
    sr = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    cal = ar / abs(mdd) if mdd < 0 else np.nan
    return dict(AR=ar, SR=sr, MDD=mdd, Vol=float(r.std()*np.sqrt(12)), N=len(r), Calmar=cal)


def vol_target(r, target=0.06, window=12, l_max=4.0):
    r = r.dropna().sort_index()
    sigma = r.rolling(window).std() * np.sqrt(12)
    L = (target / sigma).clip(upper=l_max).shift(1)
    return (r * L).dropna()


def mvo_lw_weights(returns_df, mu_window=6):
    if returns_df.shape[1] < 2:
        return pd.Series(1.0, index=returns_df.columns)
    try:
        lw = LedoitWolf().fit(returns_df.values)
        sigma = lw.covariance_
        n = sigma.shape[0]
        mu = returns_df.iloc[-mu_window:].mean().values
        inv_sigma = np.linalg.pinv(sigma + 1e-6 * np.eye(n))
        w = inv_sigma @ mu
        w = np.maximum(w, 0)
        if w.sum() <= 0: return pd.Series(1.0/n, index=returns_df.columns)
        return pd.Series(w / w.sum(), index=returns_df.columns)
    except Exception:
        n = returns_df.shape[1]
        return pd.Series(1.0/n, index=returns_df.columns)


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")

    # Sign + IC0
    signs = {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        signs[fid] = 1.0 if g.mean() >= 0 else -1.0

    ic_mat = ic.pivot_table(index="date", columns="factor_id", values="ic")
    ic_mat_signed = ic_mat.multiply(pd.Series(signs).reindex(ic_mat.columns).fillna(0), axis=1)
    print(f"Pool: {len(signs)} factors  IC dates: {ic_mat.index.min().date()} → {ic_mat.index.max().date()}")
    sim_dates = sorted(ic_mat.index.tolist())

    def active_at(t):
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 12: continue  # need 12m history
            out.append(fid)
        return out

    # ---------- Build panel ----------
    print("\n[panel] Building (t, factor) feature panel ...")
    fwd_window = 6
    look_backs = [3, 6, 12]
    rows = []

    for t in sim_dates:
        # forward window must fit
        fwd_end = t + pd.DateOffset(months=fwd_window)
        if fwd_end > ic_mat_signed.index.max(): continue

        cands = active_at(t)
        if len(cands) < 10: continue

        # Forward target: ICIR over (t, t+6m]
        fwd = ic_mat_signed[(ic_mat_signed.index > t) & (ic_mat_signed.index <= fwd_end)][cands]
        if fwd.empty: continue
        fwd_mean = fwd.mean()
        fwd_std  = fwd.std().replace(0, np.nan)
        target_icir = (fwd_mean / fwd_std).abs()

        # Cross-factor correlation matrix over [t-12m, t]
        co12 = t - pd.DateOffset(months=12)
        corr_panel = ic_mat_signed[(ic_mat_signed.index > co12) & (ic_mat_signed.index <= t)][cands].dropna(how="all")
        if len(corr_panel) >= 8:
            corr_mat = corr_panel.corr().abs().fillna(0)
        else:
            corr_mat = None

        # Cross-factor dispersion (regime proxy)
        cross_disp = corr_panel.std(axis=1).mean() if not corr_panel.empty else np.nan
        cross_avg  = corr_panel.mean(axis=1).mean() if not corr_panel.empty else np.nan

        for fid in cands:
            if fid not in target_icir.index or pd.isna(target_icir[fid]): continue

            row = {"t": t, "fid": fid, "y": float(target_icir[fid])}

            # Multi-window trailing stats
            for lb in look_backs:
                co = t - pd.DateOffset(months=lb)
                hist = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), fid].dropna()
                if len(hist) < max(lb // 2, 2):
                    continue
                row[f"mean_{lb}m"] = float(hist.mean())
                row[f"std_{lb}m"]  = float(hist.std()) if hist.std() == hist.std() else 0
                row[f"icir_{lb}m"] = float(hist.mean()/hist.std()) if hist.std() > 0 else 0
                row[f"abs_icir_{lb}m"] = abs(row[f"icir_{lb}m"])
                # Trend: simple slope
                if len(hist) >= 3:
                    x = np.arange(len(hist), dtype=float)
                    y = hist.values
                    row[f"trend_{lb}m"] = float(np.polyfit(x, y, 1)[0])
                # Drawdown of cumsum
                cs = hist.cumsum()
                row[f"dd_{lb}m"] = float((cs - cs.cummax()).min())
                # Sign-match rate
                row[f"signmatch_{lb}m"] = float((hist > 0).mean())

            # Crowding features
            if corr_mat is not None and fid in corr_mat.columns:
                others = corr_mat[fid].drop(index=[fid]) if fid in corr_mat.index else None
                if others is not None and len(others) > 0:
                    row["crowd_mean"] = float(others.mean())
                    row["crowd_max"]  = float(others.max())
                    row["crowd_p75"]  = float(others.quantile(0.75))
                    row["crowd_p90"]  = float(others.quantile(0.90))

            # Cross-factor / regime
            row["mkt_disp"] = float(cross_disp) if not pd.isna(cross_disp) else 0
            row["mkt_avg_ic"] = float(cross_avg) if not pd.isna(cross_avg) else 0

            # Age
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            row["age_m"] = months_between(disc, t)

            rows.append(row)

    panel = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in panel.columns if c not in ("t", "fid", "y")]
    print(f"  panel rows: {len(panel)}  features: {len(feat_cols)}")
    print(f"  feature names: {feat_cols}")

    # ---------- Purged + embargoed temporal split ----------
    cut_train = pd.Timestamp("2015-06-01")
    cut_val_lo = pd.Timestamp("2016-01-01")
    cut_val_hi = pd.Timestamp("2016-09-01")
    cut_test = pd.Timestamp("2017-01-01")
    train = panel[panel.t < cut_train]
    val   = panel[(panel.t >= cut_val_lo) & (panel.t < cut_val_hi)]
    test  = panel[panel.t >= cut_test]
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # ---------- Train XGBoost on forward ICIR ----------
    print("\n[train] XGBoost predicting forward 6m ICIR ...")
    m = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, tree_method="exact",
        random_state=42, n_jobs=1, verbosity=0,
        early_stopping_rounds=50,
    )
    m.fit(train[feat_cols], train["y"],
          eval_set=[(val[feat_cols], val["y"])], verbose=False)
    val_pred  = m.predict(val[feat_cols])
    test_pred = m.predict(test[feat_cols])
    val_mse  = ((val_pred  - val["y"]) ** 2).mean()
    test_mse = ((test_pred - test["y"]) ** 2).mean()
    val_corr = np.corrcoef(val_pred,  val["y"])[0, 1]
    test_corr= np.corrcoef(test_pred, test["y"])[0, 1]
    print(f"  val MSE={val_mse:.4f}  rho={val_corr:+.3f}")
    print(f"  test MSE={test_mse:.4f}  rho={test_corr:+.3f}")

    # Compare to trailing-ICIR predictor (a naive baseline)
    trailing_pred = test["abs_icir_12m"].values
    trail_corr = np.corrcoef(trailing_pred, test["y"])[0, 1]
    trail_mse  = ((trailing_pred - test["y"]) ** 2).mean()
    print(f"  TRAILING ICIR baseline: test MSE={trail_mse:.4f}  rho={trail_corr:+.3f}")

    # ---------- Feature importance ----------
    fi = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n  Top 12 feature importances:")
    for k, v in fi.head(12).items():
        print(f"    {k:<22} {v:.4f}")

    # ---------- Portfolio simulation ----------
    print("\n[sim] Portfolio simulation on test period ...")
    test = test.assign(pred=test_pred)
    test["t"] = pd.to_datetime(test["t"])

    K = 10
    cs_vol = 0.10
    test_dates = sorted(test["t"].unique())

    def signed_ic_at(t, fids):
        if t not in ic_mat_signed.index: return pd.Series(dtype=float)
        return ic_mat_signed.loc[t, fids].dropna()

    def factor_recent_vol(t, cands, window=12):
        co = t - pd.DateOffset(months=window)
        sub = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), cands].dropna(how="all")
        if sub.empty: return pd.Series(0.05, index=cands)
        return sub.std().replace(0, 0.05).reindex(cands).fillna(0.05)

    # Simulation runners
    def run_pred_topk():
        rets = {}
        for t in test_dates:
            sub = test[test.t == t]
            if len(sub) < K: continue
            top = sub.nlargest(K, "pred")
            sel = top["fid"].tolist()
            rv = factor_recent_vol(t, sel)
            w = (1.0 / rv).reindex(sel).fillna(0)
            v = signed_ic_at(t, sel)
            if v.empty or w.sum() <= 0: continue
            r_t = float((v * w.reindex(v.index).fillna(0)).sum() / w.reindex(v.index).fillna(0).sum())
            rets[t] = r_t * cs_vol
        return pd.Series(rets).sort_index()

    def run_trailing_icir():
        rets = {}
        for t in test_dates:
            sub = test[test.t == t]
            if len(sub) < K: continue
            top = sub.nlargest(K, "abs_icir_12m")
            sel = top["fid"].tolist()
            rv = factor_recent_vol(t, sel)
            w = (1.0 / rv).reindex(sel).fillna(0)
            v = signed_ic_at(t, sel)
            if v.empty or w.sum() <= 0: continue
            r_t = float((v * w.reindex(v.index).fillna(0)).sum() / w.reindex(v.index).fillna(0).sum())
            rets[t] = r_t * cs_vol
        return pd.Series(rets).sort_index()

    def run_mvo_lw():
        rets = {}
        for t in test_dates:
            sub = test[test.t == t]
            if len(sub) < K: continue
            cands = sub["fid"].tolist()
            co = t - pd.DateOffset(months=12)
            win = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), cands].dropna(how="all")
            win = win.dropna(axis=1, thresh=int(0.6*len(win))).fillna(0)
            if win.shape[1] < 5: continue
            w_full = mvo_lw_weights(win, 6)
            top = w_full.nlargest(K)
            sel = top.index.tolist()
            v = signed_ic_at(t, sel)
            if v.empty or top.sum() <= 0: continue
            w = top.reindex(v.index).fillna(0)
            r_t = float((v * w).sum() / w.sum())
            rets[t] = r_t * cs_vol
        return pd.Series(rets).sort_index()

    # Combo: pred + invvol; pred + MVO weights
    def run_pred_then_mvo():
        rets = {}
        for t in test_dates:
            sub = test[test.t == t]
            if len(sub) < 2*K: continue
            pre = sub.nlargest(2*K, "pred")["fid"].tolist()
            co = t - pd.DateOffset(months=12)
            win = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), pre].dropna(how="all")
            win = win.dropna(axis=1, thresh=int(0.6*len(win))).fillna(0)
            if win.shape[1] < 5: continue
            w_full = mvo_lw_weights(win, 6)
            top = w_full.nlargest(K)
            sel = top.index.tolist()
            v = signed_ic_at(t, sel)
            if v.empty or top.sum() <= 0: continue
            w = top.reindex(v.index).fillna(0)
            r_t = float((v * w).sum() / w.sum())
            rets[t] = r_t * cs_vol
        return pd.Series(rets).sort_index()

    runs = {
        "Pred-ICIR (XGB) + invvol":    run_pred_topk(),
        "Trailing ICIR_12m + invvol":  run_trailing_icir(),
        "MVO+LW (mu=6m)":              run_mvo_lw(),
        "Pred-ICIR + MVO+LW":          run_pred_then_mvo(),
    }

    rows = []
    print("\n=== K=10, RAW (test period 2017-2020) ===")
    for name, r in runs.items():
        m_ = metrics(r)
        print(f"  {name:<32} AR={m_['AR']:>+7.2%} SR={m_['SR']:>6.3f} "
              f"MDD={m_['MDD']:>+7.2%} Vol={m_['Vol']:.2%} Calmar={m_['Calmar']:.2f} N={m_['N']}")
        rows.append({"Variant": "raw", "Rule": name, **m_})

    for tgt in [0.04, 0.06]:
        print(f"\n=== VOL-TARGET {tgt*100:.0f}% ===")
        for name, r in runs.items():
            r_vt = vol_target(r, target=tgt)
            m_ = metrics(r_vt)
            rows.append({"Variant": f"vt_{tgt:.2f}", "Rule": name, **m_})
        sub = pd.DataFrame([r for r in rows if r["Variant"] == f"vt_{tgt:.2f}"]).sort_values("SR", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row.Rule:<32} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
                  f"MDD={row.MDD:>+7.2%} Calmar={row.Calmar:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v13_direct_target.csv", index=False)

    # Pareto check vs trailing ICIR (the SOTA baseline)
    print("\n========== Pareto vs Trailing ICIR_12m+invvol (vt=6%) ==========")
    sub6 = df[df.Variant == "vt_0.06"].set_index("Rule")
    if "Trailing ICIR_12m + invvol" in sub6.index:
        b = sub6.loc["Trailing ICIR_12m + invvol"]
        print(f"Baseline: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%} Cal={b.Calmar:.2f}")
        for n, row in sub6.iterrows():
            if n == "Trailing ICIR_12m + invvol": continue
            dar = row.AR - b.AR
            dsr = row.SR - b.SR
            dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0: tag = "  ★ PARETO BEATS SOTA"
            elif dar < 0 and dsr < 0 and dmdd < 0: tag = "  ✗ LOSES"
            print(f"  {n:<32} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
