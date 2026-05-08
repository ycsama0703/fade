"""
Anchor v11 — true SOTA baselines vs FADE.

Mainstream factor-selection / portfolio-construction baselines we hadn't tested:

  1. ICIR_12m (top-K by 12m ic_mean/ic_std)
       — the de-facto industrial standard for factor ranking.
       — both equal-weight and inv-vol-weight variants
  2. Min-Variance + Ledoit-Wolf shrinkage (full pool weighting)
  3. HRP (Hierarchical Risk Parity, López de Prado 2016, full pool)
  4. MVO + Ledoit-Wolf with mu = trailing-6m sign-adj IC mean (top-K by mu)
  5. MV-LW within top-K (top-K by ICmom3, MV weights inside)
  6. HRP within top-K
  7. FADE: ICmom3+IC0+RollHL + inv-vol (our champion)

All evaluated under strict-OOS cutoff=2017-01-01, vt=6%, on CSI500 (141 factors).
Pareto check vs each baseline.
"""
from __future__ import annotations
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.covariance import LedoitWolf
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

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
        return dict(AR=np.nan, SR=np.nan, MDD=np.nan, Vol=np.nan, N=0)
    cum = (1 + r).cumprod()
    cr = float(cum.iloc[-1] - 1)
    ar = float((1 + cr) ** (12 / len(r)) - 1)
    sr = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(AR=ar, SR=sr, MDD=mdd, Vol=float(r.std()*np.sqrt(12)), N=len(r))


def vol_target(r, target=0.06, window=12, l_max=4.0):
    r = r.dropna().sort_index()
    sigma = r.rolling(window).std() * np.sqrt(12)
    L = (target / sigma).clip(upper=l_max).shift(1)
    return (r * L).dropna()


def censored_labels(ic_panel, registry, cutoff_date, initial_window=6, persist_months=6):
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
            if all(v < 0 for v in signed[k+1: k+1+persist_months]):
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


# ---------- HRP implementation ----------
def hrp_weights(returns_df: pd.DataFrame) -> pd.Series:
    """López de Prado 2016 Hierarchical Risk Parity. returns_df: T×N."""
    if returns_df.shape[1] < 2:
        return pd.Series(1.0, index=returns_df.columns)
    cov = returns_df.cov().values
    corr = returns_df.corr().fillna(0).values
    n = corr.shape[0]
    if np.allclose(cov, 0):
        return pd.Series(1.0/n, index=returns_df.columns)
    dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
    np.fill_diagonal(dist, 0)
    try:
        link = sch.linkage(squareform(dist, checks=False), method="single")
        sort_ix = list(sch.leaves_list(link))
    except Exception:
        sort_ix = list(range(n))
    w = np.ones(n)
    items = [sort_ix]
    while items:
        items_ = [c[i:j] for c in items for i, j in
                  ((0, len(c)//2), (len(c)//2, len(c))) if len(c) > 1]
        for i in range(0, len(items_), 2):
            if i + 1 >= len(items_): continue
            cI, cJ = items_[i], items_[i+1]
            def cv(c):
                v = np.diag(cov)[c]
                v[v <= 0] = 1e-8
                inv_v = 1.0 / v
                wts = inv_v / inv_v.sum()
                C = cov[np.ix_(c, c)]
                return float(wts @ C @ wts)
            v_i, v_j = cv(cI), cv(cJ)
            alpha = 1 - v_i / (v_i + v_j) if (v_i + v_j) > 0 else 0.5
            w[cI] *= alpha
            w[cJ] *= (1 - alpha)
        items = items_
    w_total = w.sum()
    if w_total <= 0: w_total = 1.0
    return pd.Series(w / w_total, index=returns_df.columns)


# ---------- Min-Variance + Ledoit-Wolf ----------
def min_var_lw_weights(returns_df: pd.DataFrame) -> pd.Series:
    if returns_df.shape[1] < 2:
        return pd.Series(1.0, index=returns_df.columns)
    try:
        lw = LedoitWolf().fit(returns_df.values)
        sigma = lw.covariance_
        n = sigma.shape[0]
        # min-variance: w = sigma^-1 * 1 / (1' sigma^-1 1)
        ones = np.ones(n)
        inv_sigma = np.linalg.pinv(sigma + 1e-6 * np.eye(n))
        w = inv_sigma @ ones
        w = w / w.sum()
        # Long-only constraint via projection (clip negatives, renormalize)
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            return pd.Series(1.0/n, index=returns_df.columns)
        w = w / w.sum()
        return pd.Series(w, index=returns_df.columns)
    except Exception:
        n = returns_df.shape[1]
        return pd.Series(1.0/n, index=returns_df.columns)


# ---------- MVO + Ledoit-Wolf with mu = trailing mean ----------
def mvo_lw_weights(returns_df: pd.DataFrame, mu_window: int = 6) -> pd.Series:
    if returns_df.shape[1] < 2:
        return pd.Series(1.0, index=returns_df.columns)
    try:
        lw = LedoitWolf().fit(returns_df.values)
        sigma = lw.covariance_
        n = sigma.shape[0]
        mu = returns_df.iloc[-mu_window:].mean().values  # trailing 6m
        inv_sigma = np.linalg.pinv(sigma + 1e-6 * np.eye(n))
        w = inv_sigma @ mu
        # Long-only projection
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            return pd.Series(1.0/n, index=returns_df.columns)
        w = w / w.sum()
        return pd.Series(w, index=returns_df.columns)
    except Exception:
        n = returns_df.shape[1]
        return pd.Series(1.0/n, index=returns_df.columns)


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
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic
        if len(g) < 6: continue
        first6 = g.iloc[:6]
        ic0_abs[fid] = abs(first6.mean())
        signs[fid]   = 1.0 if first6.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)

    cutoff = "2017-01-01"
    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp(cutoff)]
    print(f"Pool: {len(ic0)} factors  Sim: {len(sim_dates)} months")

    print("[censored labels + rolling pred_HL] ...")
    censored = censored_labels(ic, reg, cutoff)
    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)

    # Build sign-adjusted IC time series matrix [date × factor]
    print("[building IC matrix] ...")
    ic_mat = ic.pivot_table(index="date", columns="factor_id", values="ic")
    sign_ser = pd.Series(signs)
    # Apply sign per column
    ic_mat_signed = ic_mat.multiply(sign_ser.reindex(ic_mat.columns).fillna(0), axis=1)
    print(f"  ic_mat_signed shape: {ic_mat_signed.shape}")

    def active_at(t):
        out_ = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out_.append(fid)
        return out_

    def signed_ic_at(t, fids):
        if t in ic_mat_signed.index:
            v = ic_mat_signed.loc[t, fids].dropna()
            return v
        return pd.Series(dtype=float)

    def trailing_panel(t, fids, months=12):
        co = t - pd.DateOffset(months=months)
        win = ic_mat_signed.loc[co:t, fids].dropna(how="all")
        win = win.dropna(axis=1, thresh=int(0.6*len(win)))  # require 60% non-null
        win = win.fillna(0)
        return win

    def ic_lookback(t, cands, lookback=3):
        co = t - pd.DateOffset(months=lookback)
        sub = ic_mat_signed.loc[co:t, cands].dropna(how="all")
        if sub.empty: return pd.Series(dtype=float)
        return sub.mean().abs()

    def icir_score(t, cands, lookback=12):
        sub = trailing_panel(t, cands, months=lookback)
        if sub.empty: return pd.Series(dtype=float)
        mu = sub.mean()
        sd = sub.std().replace(0, np.nan)
        ir = (mu / sd).abs().fillna(0)
        return ir

    def factor_recent_vol(t, cands, window=12):
        sub = trailing_panel(t, cands, months=window)
        if sub.empty: return pd.Series(0.05, index=cands)
        return sub.std().replace(0, 0.05).reindex(cands).fillna(0.05)

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def topk_score(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    K = 10
    cs_vol = 0.10

    def run_rule(name):
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue

            if name == "ICmom3+IC0":
                sel = topk_score(zscore(ic_lookback(t, cands, 3)).fillna(0)
                                + zscore(ic0.reindex(cands)).fillna(0), cands, K)
                w = pd.Series(1.0, index=sel)
            elif name == "ICmom12":
                sel = topk_score(ic_lookback(t, cands, 12), cands, K)
                w = pd.Series(1.0, index=sel)
            elif name == "ICIR_12m":
                sel = topk_score(icir_score(t, cands, 12), cands, K)
                w = pd.Series(1.0, index=sel)
            elif name == "ICIR_12m+invvol":
                sel = topk_score(icir_score(t, cands, 12), cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "MV+LW (full)":
                win = trailing_panel(t, cands, 12)
                if win.shape[1] < 5: continue
                w_full = min_var_lw_weights(win)
                # Take top-K weights
                top = w_full.nlargest(K)
                sel, w = top.index.tolist(), top
            elif name == "HRP (full)":
                win = trailing_panel(t, cands, 12)
                if win.shape[1] < 5: continue
                w_full = hrp_weights(win)
                top = w_full.nlargest(K)
                sel, w = top.index.tolist(), top
            elif name == "MVO+LW (mu=6m)":
                win = trailing_panel(t, cands, 12)
                if win.shape[1] < 5: continue
                w_full = mvo_lw_weights(win, mu_window=6)
                top = w_full.nlargest(K)
                sel, w = top.index.tolist(), top
            elif name == "ICmom3 + MV-LW":
                pre_sel = topk_score(ic_lookback(t, cands, 3), cands, K)
                win = trailing_panel(t, pre_sel, 12)
                if win.shape[1] < 3: continue
                w = min_var_lw_weights(win)
                sel = list(w.index)
            elif name == "ICmom3 + HRP":
                pre_sel = topk_score(ic_lookback(t, cands, 3), cands, K)
                win = trailing_panel(t, pre_sel, 12)
                if win.shape[1] < 3: continue
                w = hrp_weights(win)
                sel = list(w.index)
            elif name == "FADE_invvol":
                # ICmom3+IC0+RollHL composite, inv-vol within top-K
                z_total = (zscore(ic_lookback(t, cands, 3)).fillna(0)
                           + zscore(ic0.reindex(cands)).fillna(0))
                if t in rolling_hl.index:
                    z_total = z_total + zscore(rolling_hl.loc[t].reindex(cands)).fillna(0)
                sel = topk_score(z_total, cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "FADE_eq":
                z_total = (zscore(ic_lookback(t, cands, 3)).fillna(0)
                           + zscore(ic0.reindex(cands)).fillna(0))
                if t in rolling_hl.index:
                    z_total = z_total + zscore(rolling_hl.loc[t].reindex(cands)).fillna(0)
                sel = topk_score(z_total, cands, K)
                w = pd.Series(1.0, index=sel)
            else:
                continue

            v_series = signed_ic_at(t, sel)
            if v_series.empty: continue
            w_use = w.reindex(v_series.index).fillna(0)
            if w_use.sum() <= 0: continue
            v = float((v_series * w_use).sum() / w_use.sum())
            rets[t] = v * cs_vol
        return pd.Series(rets).sort_index()

    rules = [
        # baselines
        "ICmom3+IC0",
        "ICmom12",
        # SOTA — selection
        "ICIR_12m",
        "ICIR_12m+invvol",
        # SOTA — weighting
        "MV+LW (full)",
        "HRP (full)",
        "MVO+LW (mu=6m)",
        "ICmom3 + MV-LW",
        "ICmom3 + HRP",
        # FADE
        "FADE_eq",
        "FADE_invvol",
    ]

    raw_rets = {}
    print(f"\n=== Running K={K} rules ===")
    for name in rules:
        r = run_rule(name)
        raw_rets[name] = r

    print(f"\n=== K={K}, RAW ===")
    rows = []
    for name, r in raw_rets.items():
        m = metrics(r)
        cal = m["AR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
        print(f"  {name:<22} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
              f"MDD={m['MDD']:>+7.2%} Vol={m['Vol']:.2%} Calmar={cal:.2f} N={m['N']}")
        rows.append({"Variant": "raw", "Rule": name, **m, "Calmar": cal})

    for tgt in [0.04, 0.06]:
        print(f"\n=== K={K}, VOL-TARGETED to {tgt*100:.0f}% ===")
        for name, r in raw_rets.items():
            r_vt = vol_target(r, target=tgt)
            m = metrics(r_vt)
            cal = m["AR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
            rows.append({"Variant": f"vt_{tgt:.2f}", "Rule": name, **m, "Calmar": cal})
        sub = pd.DataFrame([r for r in rows if r["Variant"] == f"vt_{tgt:.2f}"]).sort_values("SR", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row.Rule:<22} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
                  f"MDD={row.MDD:>+7.2%} Calmar={row.Calmar:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v11_sota_baselines.csv", index=False)

    # Pareto check FADE vs each baseline at vt=6%
    print("\n========== Pareto check at vt=6% ==========")
    sub6 = df[df.Variant == "vt_0.06"].set_index("Rule")
    for fade_rule in ["FADE_eq", "FADE_invvol"]:
        if fade_rule not in sub6.index: continue
        f = sub6.loc[fade_rule]
        print(f"\n>>> {fade_rule}: AR={f.AR:+.2%} SR={f.SR:.3f} MDD={f.MDD:+.2%} Cal={f.Calmar:.2f}")
        for name in rules:
            if name in ("FADE_eq","FADE_invvol") or name not in sub6.index: continue
            b = sub6.loc[name]
            dar  = f.AR - b.AR
            dsr  = f.SR - b.SR
            dmdd = f.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0: tag = "  ★ FADE PARETO"
            elif dar < 0 and dsr < 0 and dmdd < 0: tag = "  ✗ FADE LOSES (Pareto-dominated)"
            elif dar > 0 and dsr > 0: tag = "  ◇ FADE AR+SR up"
            elif dsr > 0 and dmdd > 0: tag = "  ◇ FADE SR+MDD up"
            print(f"  vs {name:<20} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
