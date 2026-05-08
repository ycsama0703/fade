"""
Anchor v15 — MVO+LW with survival hazard adjustment.

MVO+LW (mu=6m) is the current best baseline (SR=5.37 at vt=6%). It uses:
  mu = trailing 6m sign-adj IC mean
  Sigma = Ledoit-Wolf shrinkage of trailing 12m covariance
  w = max(Sigma^-1 mu, 0), normalized

Idea: discount mu by survival hazard. If a factor is predicted to decay
soon, its forward mu is biased upward by current rolling estimate.
Adjusted mu reflects this risk.

Variants tested:
  A) mu_adj = mu * (1 - lambda * hazard_z)  for lambda in {0.1, 0.3, 0.5, 1.0}
     where hazard_z = z-scored hazard score (higher = riskier)
  B) Pre-filter: drop factors with hazard_z above threshold q, then MVO+LW
  C) Multiplicative survival weight: w_post = w_mvo * surv_prob_norm
"""
from __future__ import annotations
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT  = ROOT / "outputs"

from src.models.survival_ensemble import GradientBoostingSurvival
from src.features.early_ic import extract_early_ic_features_batch


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


def mvo_lw_weights(returns_df, mu, mu_window=6):
    """MVO with explicit mu (so we can pass an adjusted version)."""
    if returns_df.shape[1] < 2:
        return pd.Series(1.0, index=returns_df.columns)
    try:
        lw = LedoitWolf().fit(returns_df.values)
        sigma = lw.covariance_
        n = sigma.shape[0]
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
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    signs = {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        signs[fid] = 1.0 if g.mean() >= 0 else -1.0

    ic_mat = ic.pivot_table(index="date", columns="factor_id", values="ic")
    ic_mat_signed = ic_mat.multiply(pd.Series(signs).reindex(ic_mat.columns).fillna(0), axis=1)

    cutoff = "2017-01-01"
    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp(cutoff)]
    print(f"Pool: {len(signs)}  Sim: {len(sim_dates)}")

    censored = censored_labels(ic, reg, cutoff)
    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)
    print(f"  rolling_hl shape: {rolling_hl.shape}  median: {rolling_hl.median().median():.1f}")

    def active_at(t):
        return [fid for fid in signs
                if pd.Timestamp(reg_idx.loc[fid,"discovery_date"]) <= t
                and months_between(pd.Timestamp(reg_idx.loc[fid,"discovery_date"]), t) >= 12]

    def trailing_window(t, fids, months=12):
        co = t - pd.DateOffset(months=months)
        win = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), fids].dropna(how="all")
        win = win.dropna(axis=1, thresh=int(0.6*len(win))).fillna(0)
        return win

    def signed_ic_at(t, fids):
        if t not in ic_mat_signed.index: return pd.Series(dtype=float)
        return ic_mat_signed.loc[t, fids].dropna()

    K = 10
    cs_vol = 0.10
    runs = {}

    # Plain MVO+LW (baseline) — this is current SOTA
    rets_mvo = {}
    for t in sim_dates:
        cands = active_at(t)
        if len(cands) < K: continue
        win = trailing_window(t, cands, 12)
        if win.shape[1] < 5: continue
        mu = win.iloc[-6:].mean().values
        w_full = mvo_lw_weights(win, mu)
        top = w_full.nlargest(K)
        sel = top.index.tolist()
        v = signed_ic_at(t, sel)
        if v.empty or top.sum() <= 0: continue
        w_use = top.reindex(v.index).fillna(0)
        rets_mvo[t] = float((v * w_use).sum() / w_use.sum()) * cs_vol
    runs["MVO+LW (mu=6m) [SOTA]"] = pd.Series(rets_mvo).sort_index()

    # Variant A: MVO with hazard-adjusted mu
    # hazard_score = -pred_hl (smaller pred_hl => higher hazard)
    # Or use 1/pred_hl as hazard proxy
    for lam in [0.1, 0.3, 0.5, 1.0]:
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            win = trailing_window(t, cands, 12)
            if win.shape[1] < 5: continue
            cols = win.columns.tolist()
            mu = win.iloc[-6:].mean().values
            # Get hazard score for these cols
            if t in rolling_hl.index:
                hl = rolling_hl.loc[t, cols].fillna(rolling_hl.loc[t].median())
                hazard = 1.0 / hl  # smaller pred_hl => larger hazard
                hz = (hazard - hazard.mean()) / (hazard.std() if hazard.std()>0 else 1)
                discount = (1 - lam * hz.values).clip(min=0.1)
                mu_adj = mu * discount
            else:
                mu_adj = mu
            w_full = mvo_lw_weights(win, mu_adj)
            top = w_full.nlargest(K)
            sel = top.index.tolist()
            v = signed_ic_at(t, sel)
            if v.empty or top.sum() <= 0: continue
            w_use = top.reindex(v.index).fillna(0)
            rets[t] = float((v * w_use).sum() / w_use.sum()) * cs_vol
        runs[f"MVO+LW × hazard (λ={lam})"] = pd.Series(rets).sort_index()

    # Variant B: Pre-filter low-hazard factors, then MVO+LW
    for q in [0.5, 0.7, 0.9]:
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            if t not in rolling_hl.index:
                runs_use = cands
            else:
                hl = rolling_hl.loc[t, cands].dropna()
                if hl.empty: continue
                threshold = hl.quantile(1 - q)  # keep top q quantile of pred_hl (low hazard)
                runs_use = hl[hl >= threshold].index.tolist()
            if len(runs_use) < K: continue
            win = trailing_window(t, runs_use, 12)
            if win.shape[1] < 5: continue
            mu = win.iloc[-6:].mean().values
            w_full = mvo_lw_weights(win, mu)
            top = w_full.nlargest(K)
            sel = top.index.tolist()
            v = signed_ic_at(t, sel)
            if v.empty or top.sum() <= 0: continue
            w_use = top.reindex(v.index).fillna(0)
            rets[t] = float((v * w_use).sum() / w_use.sum()) * cs_vol
        runs[f"MVO+LW + pre-filter (top {int(q*100)}%)"] = pd.Series(rets).sort_index()

    # Variant C: Post-multiply MVO weights by survival score
    rets = {}
    for t in sim_dates:
        cands = active_at(t)
        if len(cands) < K: continue
        win = trailing_window(t, cands, 12)
        if win.shape[1] < 5: continue
        cols = win.columns.tolist()
        mu = win.iloc[-6:].mean().values
        w_full = mvo_lw_weights(win, mu)
        if t in rolling_hl.index:
            hl = rolling_hl.loc[t, cols].fillna(rolling_hl.loc[t].median())
            surv = (hl / hl.max()).clip(0, 1)  # survival likelihood proxy
            w_full = (w_full * surv).fillna(0)
            if w_full.sum() <= 0: continue
            w_full = w_full / w_full.sum()
        top = w_full.nlargest(K)
        sel = top.index.tolist()
        v = signed_ic_at(t, sel)
        if v.empty or top.sum() <= 0: continue
        w_use = top.reindex(v.index).fillna(0)
        rets[t] = float((v * w_use).sum() / w_use.sum()) * cs_vol
    runs["MVO+LW × survival weight"] = pd.Series(rets).sort_index()

    rows = []
    print("\n=== K=10 RAW ===")
    for n, r in runs.items():
        m_ = metrics(r)
        print(f"  {n:<36} AR={m_['AR']:>+7.2%} SR={m_['SR']:>6.3f} "
              f"MDD={m_['MDD']:>+7.2%} Vol={m_['Vol']:.2%} Calmar={m_['Calmar']:.2f} N={m_['N']}")
        rows.append({"Variant":"raw","Rule":n, **m_})

    for tgt in [0.04, 0.06]:
        print(f"\n=== VOL-TARGET {tgt*100:.0f}% ===")
        for n, r in runs.items():
            r_vt = vol_target(r, target=tgt)
            m_ = metrics(r_vt)
            rows.append({"Variant":f"vt_{tgt:.2f}","Rule":n, **m_})
        sub = pd.DataFrame([row for row in rows if row["Variant"] == f"vt_{tgt:.2f}"]).sort_values("SR", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row.Rule:<36} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
                  f"MDD={row.MDD:>+7.2%} Calmar={row.Calmar:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v15_mvo_hazard.csv", index=False)

    # Pareto vs MVO+LW SOTA
    print("\n========== Pareto vs MVO+LW (mu=6m) [SOTA] (vt=6%) ==========")
    sub6 = df[df.Variant == "vt_0.06"].set_index("Rule")
    sota_key = "MVO+LW (mu=6m) [SOTA]"
    if sota_key in sub6.index:
        b = sub6.loc[sota_key]
        print(f"SOTA: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%} Cal={b.Calmar:.2f}")
        for n, row in sub6.sort_values("SR", ascending=False).iterrows():
            if n == sota_key: continue
            dar = row.AR - b.AR
            dsr = row.SR - b.SR
            dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0: tag = "  ★ FADE PARETO BEATS SOTA"
            elif dar < 0 and dsr < 0 and dmdd < 0: tag = "  ✗ LOSES"
            print(f"  {n:<36} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
