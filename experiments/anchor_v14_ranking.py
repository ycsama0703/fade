"""
Anchor v14 — Learning to Rank + cross-sectional detrending.

V13 found: XGBoost regression on forward ICIR achieves better MSE than
trailing-ICIR baseline (rho 0.10 vs 0.02), but the resulting selection is
WORSE in portfolio terms. The cause: top features (age, mkt_avg_ic, mkt_disp)
are time-level features that cannot distinguish factors within a t-slice.
The model learned "when is ICIR high overall" — useless for cross-sectional
ranking.

Two fixes (standard in IR / LambdaMART literature):

  (a) Detrend target per t: y' = y - mean(y | t)
      Forces the model to learn within-t cross-sectional signal only.

  (b) XGBRanker with per-t groups (pairwise / NDCG loss)
      Directly optimizes ranking within each t.

Both should produce predictions whose cross-sectional rank order matches
forward ICIR rank order — which is what top-K selection actually needs.
"""
from __future__ import annotations
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf
from xgboost import XGBRegressor, XGBRanker

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

    signs = {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        signs[fid] = 1.0 if g.mean() >= 0 else -1.0

    ic_mat = ic.pivot_table(index="date", columns="factor_id", values="ic")
    ic_mat_signed = ic_mat.multiply(pd.Series(signs).reindex(ic_mat.columns).fillna(0), axis=1)
    print(f"Pool: {len(signs)} factors")
    sim_dates = sorted(ic_mat.index.tolist())

    def active_at(t):
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 12: continue
            out.append(fid)
        return out

    print("\n[panel] Building features ...")
    fwd_window = 6
    look_backs = [3, 6, 12]
    rows = []

    for t in sim_dates:
        fwd_end = t + pd.DateOffset(months=fwd_window)
        if fwd_end > ic_mat_signed.index.max(): continue
        cands = active_at(t)
        if len(cands) < 10: continue

        fwd = ic_mat_signed[(ic_mat_signed.index > t) & (ic_mat_signed.index <= fwd_end)][cands]
        if fwd.empty: continue
        fwd_std = fwd.std().replace(0, np.nan)
        target_icir = (fwd.mean() / fwd_std).abs()

        co12 = t - pd.DateOffset(months=12)
        corr_panel = ic_mat_signed[(ic_mat_signed.index > co12) & (ic_mat_signed.index <= t)][cands].dropna(how="all")
        corr_mat = corr_panel.corr().abs().fillna(0) if len(corr_panel) >= 8 else None

        for fid in cands:
            if fid not in target_icir.index or pd.isna(target_icir[fid]): continue
            row = {"t": t, "fid": fid, "y": float(target_icir[fid])}

            for lb in look_backs:
                co = t - pd.DateOffset(months=lb)
                hist = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), fid].dropna()
                if len(hist) < max(lb // 2, 2): continue
                row[f"mean_{lb}m"]  = float(hist.mean())
                row[f"std_{lb}m"]   = float(hist.std()) if hist.std() == hist.std() else 0
                row[f"icir_{lb}m"]  = float(hist.mean()/hist.std()) if hist.std() > 0 else 0
                row[f"abs_icir_{lb}m"] = abs(row[f"icir_{lb}m"])
                if len(hist) >= 3:
                    x = np.arange(len(hist), dtype=float)
                    row[f"trend_{lb}m"] = float(np.polyfit(x, hist.values, 1)[0])
                cs = hist.cumsum()
                row[f"dd_{lb}m"] = float((cs - cs.cummax()).min())
                row[f"signmatch_{lb}m"] = float((hist > 0).mean())

            if corr_mat is not None and fid in corr_mat.columns:
                others = corr_mat[fid].drop(index=[fid]) if fid in corr_mat.index else None
                if others is not None and len(others) > 0:
                    row["crowd_mean"] = float(others.mean())
                    row["crowd_max"]  = float(others.max())
                    row["crowd_p75"]  = float(others.quantile(0.75))
                    row["crowd_p90"]  = float(others.quantile(0.90))

            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            row["age_m"] = months_between(disc, t)
            rows.append(row)

    panel = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in panel.columns if c not in ("t", "fid", "y")]
    print(f"  panel rows: {len(panel)}  features: {len(feat_cols)}")

    # Cross-sectional detrending of target
    panel["y_xs"] = panel["y"] - panel.groupby("t")["y"].transform("mean")

    # Splits
    cut_train = pd.Timestamp("2015-06-01")
    cut_val_lo = pd.Timestamp("2016-01-01")
    cut_val_hi = pd.Timestamp("2016-09-01")
    cut_test = pd.Timestamp("2017-01-01")
    train = panel[panel.t < cut_train].copy()
    val   = panel[(panel.t >= cut_val_lo) & (panel.t < cut_val_hi)].copy()
    test  = panel[panel.t >= cut_test].copy()
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # ----- (a) XGBoost regression on detrended target -----
    print("\n[A] XGBoost on detrended (cross-sectional) target ...")
    m_xs = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, tree_method="exact",
        random_state=42, n_jobs=1, verbosity=0,
        early_stopping_rounds=50,
    )
    m_xs.fit(train[feat_cols], train["y_xs"],
             eval_set=[(val[feat_cols], val["y_xs"])], verbose=False)
    test["pred_xs"] = m_xs.predict(test[feat_cols])

    # Per-t Spearman: how well predictions rank within slice
    def per_t_rank_corr(panel_, score_col, target_col="y"):
        rows = []
        for t, sub in panel_.groupby("t"):
            if len(sub) >= 5:
                rows.append(sub[score_col].corr(sub[target_col], method="spearman"))
        return float(np.nanmean(rows))
    rho_xs = per_t_rank_corr(test, "pred_xs", "y")
    rho_trail = per_t_rank_corr(test, "abs_icir_12m", "y")
    print(f"  test mean per-t Spearman rho — pred_xs={rho_xs:+.3f}  trailing ICIR={rho_trail:+.3f}")

    # Top feature importances
    fi_xs = pd.Series(m_xs.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("  Top 10 features (detrended-y model):")
    for k, v in fi_xs.head(10).items():
        print(f"    {k:<22} {v:.4f}")

    # ----- (b) XGBRanker with per-t groups -----
    print("\n[B] XGBRanker (LambdaMART pairwise) on per-t groups ...")
    train_sorted = train.sort_values("t")
    val_sorted   = val.sort_values("t")
    train_groups = train_sorted.groupby("t").size().tolist()
    val_groups   = val_sorted.groupby("t").size().tolist()
    # Convert continuous y to integer relevance grades (1-10)
    def to_grades(y_series, n_bins=10):
        ranks = y_series.rank(pct=True)
        return (ranks * n_bins).clip(0, n_bins-1).astype(int)
    train_y_g = train_sorted.groupby("t")["y"].transform(to_grades)
    val_y_g   = val_sorted.groupby("t")["y"].transform(to_grades)

    m_rk = XGBRanker(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, tree_method="exact",
        objective="rank:pairwise", random_state=42, n_jobs=1, verbosity=0,
        early_stopping_rounds=50,
    )
    m_rk.fit(train_sorted[feat_cols], train_y_g,
             group=train_groups,
             eval_set=[(val_sorted[feat_cols], val_y_g)],
             eval_group=[val_groups],
             verbose=False)
    test["pred_rk"] = m_rk.predict(test[feat_cols])
    rho_rk = per_t_rank_corr(test, "pred_rk", "y")
    print(f"  test mean per-t Spearman rho — pred_rk={rho_rk:+.3f}")

    fi_rk = pd.Series(m_rk.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("  Top 10 features (ranker):")
    for k, v in fi_rk.head(10).items():
        print(f"    {k:<22} {v:.4f}")

    # ---------- Portfolio simulation ----------
    K = 10
    cs_vol = 0.10
    test["t"] = pd.to_datetime(test["t"])
    test_dates = sorted(test["t"].unique())

    def signed_ic_at(t, fids):
        if t not in ic_mat_signed.index: return pd.Series(dtype=float)
        return ic_mat_signed.loc[t, fids].dropna()

    def factor_recent_vol(t, cands, window=12):
        co = t - pd.DateOffset(months=window)
        sub = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), cands].dropna(how="all")
        if sub.empty: return pd.Series(0.05, index=cands)
        return sub.std().replace(0, 0.05).reindex(cands).fillna(0.05)

    def run_topk(score_col, weighting="invvol", combine_with_mvo=False):
        rets = {}
        for t in test_dates:
            sub = test[test.t == t]
            if len(sub) < K: continue
            top = sub.nlargest(K, score_col)
            sel = top["fid"].tolist()
            v = signed_ic_at(t, sel)
            if v.empty: continue
            if combine_with_mvo:
                co = t - pd.DateOffset(months=12)
                pre = sub.nlargest(2*K, score_col)["fid"].tolist()
                win = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), pre].dropna(how="all")
                win = win.dropna(axis=1, thresh=int(0.6*len(win))).fillna(0)
                if win.shape[1] < 5: continue
                w_full = mvo_lw_weights(win, 6)
                top2 = w_full.nlargest(K)
                sel = top2.index.tolist()
                v = signed_ic_at(t, sel)
                if v.empty: continue
                w = top2.reindex(v.index).fillna(0)
            elif weighting == "invvol":
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(v.index).fillna(0)
            else:
                w = pd.Series(1.0, index=v.index)
            if w.sum() <= 0: continue
            r_t = float((v * w).sum() / w.sum())
            rets[t] = r_t * cs_vol
        return pd.Series(rets).sort_index()

    runs = {
        "Trailing ICIR_12m + invvol":     run_topk("abs_icir_12m"),
        "Pred-ICIR (XS-detrend) + invvol": run_topk("pred_xs"),
        "Pred-Ranker + invvol":           run_topk("pred_rk"),
        "Pred-Ranker → MVO+LW":           run_topk("pred_rk", combine_with_mvo=True),
        "Pred-XS → MVO+LW":               run_topk("pred_xs", combine_with_mvo=True),
        "MVO+LW (mu=6m, no rank)":        None,  # special — uses full pool
    }
    # Fill the MVO-only entry
    rets_mvo = {}
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
        rets_mvo[t] = float((v * w).sum() / w.sum()) * cs_vol
    runs["MVO+LW (mu=6m, no rank)"] = pd.Series(rets_mvo).sort_index()

    rows = []
    print("\n=== K=10 RAW ===")
    for n, r in runs.items():
        m_ = metrics(r)
        print(f"  {n:<32} AR={m_['AR']:>+7.2%} SR={m_['SR']:>6.3f} "
              f"MDD={m_['MDD']:>+7.2%} Vol={m_['Vol']:.2%} Calmar={m_['Calmar']:.2f} N={m_['N']}")
        rows.append({"Variant":"raw","Rule":n, **m_})

    for tgt in [0.04, 0.06]:
        print(f"\n=== VOL-TARGET {tgt*100:.0f}% ===")
        for n, r in runs.items():
            r_vt = vol_target(r, target=tgt)
            m_ = metrics(r_vt)
            rows.append({"Variant": f"vt_{tgt:.2f}", "Rule": n, **m_})
        sub = pd.DataFrame([r for r in rows if r["Variant"] == f"vt_{tgt:.2f}"]).sort_values("SR", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row.Rule:<32} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
                  f"MDD={row.MDD:>+7.2%} Calmar={row.Calmar:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v14_ranking.csv", index=False)

    # Pareto vs trailing ICIR_12m+invvol AND MVO+LW
    print("\n========== Pareto vs Trailing ICIR_12m+invvol (vt=6%) ==========")
    sub6 = df[df.Variant == "vt_0.06"].set_index("Rule")
    if "Trailing ICIR_12m + invvol" in sub6.index:
        b = sub6.loc["Trailing ICIR_12m + invvol"]
        for n, row in sub6.iterrows():
            if n == "Trailing ICIR_12m + invvol": continue
            dar = row.AR - b.AR; dsr = row.SR - b.SR; dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0: tag = "  ★ PARETO"
            print(f"  {n:<32} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")

    print("\n========== Pareto vs MVO+LW (mu=6m) (vt=6%) ==========")
    if "MVO+LW (mu=6m, no rank)" in sub6.index:
        b = sub6.loc["MVO+LW (mu=6m, no rank)"]
        for n, row in sub6.iterrows():
            if n == "MVO+LW (mu=6m, no rank)": continue
            dar = row.AR - b.AR; dsr = row.SR - b.SR; dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0: tag = "  ★ PARETO"
            print(f"  {n:<32} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
