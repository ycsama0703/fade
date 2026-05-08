"""
Anchor v12 — last rescue: does FADE add marginal value on top of ICIR_12m?

ICIR_12m + invvol is the SOTA baseline (SR 4.76 at vt=6%). Question: can we
add FADE's RollHL as a third signal and Pareto-improve?

Tests:
  1. ICIR_12m + invvol (SOTA baseline)
  2. ICIR_12m + RollHL (combine ranking + decay)
  3. ICIR_12m + RollHL + invvol
  4. ICIR_12m × RollHL (multiplicative)
  5. Filter-then-rank: top-2K by ICIR, then top-K by RollHL
  6. As above: top-2K by RollHL, then top-K by ICIR

Plus same in MVO+LW combo.
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
    if len(r) == 0: return dict(AR=np.nan, SR=np.nan, MDD=np.nan, Vol=np.nan, N=0)
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
    print(f"Pool: {len(ic0)}  Sim: {len(sim_dates)}")

    censored = censored_labels(ic, reg, cutoff)
    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)

    ic_mat = ic.pivot_table(index="date", columns="factor_id", values="ic")
    ic_mat_signed = ic_mat.multiply(pd.Series(signs).reindex(ic_mat.columns).fillna(0), axis=1)

    def active_at(t):
        return [fid for fid in signs
                if pd.Timestamp(reg_idx.loc[fid,"discovery_date"]) <= t
                and months_between(pd.Timestamp(reg_idx.loc[fid,"discovery_date"]), t) >= 6]

    def trailing_panel(t, fids, months=12):
        co = t - pd.DateOffset(months=months)
        win = ic_mat_signed.loc[co:t, fids].dropna(how="all")
        win = win.dropna(axis=1, thresh=int(0.6*len(win))).fillna(0)
        return win

    def signed_ic_at(t, fids):
        if t not in ic_mat_signed.index: return pd.Series(dtype=float)
        return ic_mat_signed.loc[t, fids].dropna()

    def icir_score(t, cands, lookback=12):
        sub = trailing_panel(t, cands, months=lookback)
        if sub.empty: return pd.Series(dtype=float)
        mu = sub.mean()
        sd = sub.std().replace(0, np.nan)
        return (mu / sd).abs().fillna(0)

    def factor_recent_vol(t, cands, window=12):
        sub = trailing_panel(t, cands, months=window)
        if sub.empty: return pd.Series(0.05, index=cands)
        return sub.std().replace(0, 0.05).reindex(cands).fillna(0.05)

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    K = 10
    cs_vol = 0.10

    def run_rule(name):
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            icir = icir_score(t, cands)
            rh = (rolling_hl.loc[t].reindex(cands)
                  if t in rolling_hl.index else pd.Series(dtype=float))

            if name == "ICIR_12m (eq)":
                sel = topk(icir, cands, K)
                w = pd.Series(1.0, index=sel)
            elif name == "ICIR_12m+invvol":
                sel = topk(icir, cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "ICIR + RollHL (sum) eq":
                if rh.empty: continue
                sc = zscore(icir).fillna(0) + zscore(rh).fillna(0)
                sel = topk(sc, cands, K)
                w = pd.Series(1.0, index=sel)
            elif name == "ICIR + RollHL (sum) invvol":
                if rh.empty: continue
                sc = zscore(icir).fillna(0) + zscore(rh).fillna(0)
                sel = topk(sc, cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "ICIR × RollHL invvol":
                if rh.empty: continue
                sc = icir * np.log1p(rh)
                sel = topk(sc, cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "ICIR→RollHL filter (top2K then K)":
                if rh.empty: continue
                pre = topk(icir, cands, 2*K)
                sel = topk(rh, pre, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "RollHL→ICIR filter":
                if rh.empty: continue
                pre = topk(rh, cands, 2*K)
                sel = topk(icir, pre, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(sel).fillna(0)
            elif name == "MVO+LW (mu=6m)":
                win = trailing_panel(t, cands, 12)
                if win.shape[1] < 5: continue
                w_full = mvo_lw_weights(win, 6)
                top = w_full.nlargest(K)
                sel, w = top.index.tolist(), top
            elif name == "MVO+LW + RollHL filter":
                # Top-2K by RollHL, then MVO+LW within
                if rh.empty: continue
                pre = topk(rh, cands, 2*K)
                win = trailing_panel(t, pre, 12)
                if win.shape[1] < 5: continue
                w_full = mvo_lw_weights(win, 6)
                top = w_full.nlargest(K)
                sel, w = top.index.tolist(), top
            elif name == "MVO+LW (mu=6m × RollHL)":
                # Boost mu by predicted-survival weight
                win = trailing_panel(t, cands, 12)
                if win.shape[1] < 5: continue
                if rh.empty: continue
                # Construct mu = trailing 6m mean × normalized rh
                mu6 = win.iloc[-6:].mean()
                rh_norm = (rh.reindex(mu6.index).fillna(0) /
                           rh.reindex(mu6.index).abs().max())
                mu_boosted = mu6 * (1 + rh_norm)
                # Manual MVO with adjusted mu
                try:
                    lw = LedoitWolf().fit(win.values)
                    sigma = lw.covariance_
                    inv_sigma = np.linalg.pinv(sigma + 1e-6*np.eye(len(sigma)))
                    w_full_arr = inv_sigma @ mu_boosted.values
                    w_full_arr = np.maximum(w_full_arr, 0)
                    if w_full_arr.sum() <= 0: continue
                    w_full = pd.Series(w_full_arr / w_full_arr.sum(), index=win.columns)
                except Exception:
                    continue
                top = w_full.nlargest(K)
                sel, w = top.index.tolist(), top
            else:
                continue

            v = signed_ic_at(t, sel)
            if v.empty: continue
            w_use = w.reindex(v.index).fillna(0)
            if w_use.sum() <= 0: continue
            r_t = float((v * w_use).sum() / w_use.sum())
            rets[t] = r_t * cs_vol
        return pd.Series(rets).sort_index()

    rules = [
        "ICIR_12m (eq)",
        "ICIR_12m+invvol",
        "ICIR + RollHL (sum) eq",
        "ICIR + RollHL (sum) invvol",
        "ICIR × RollHL invvol",
        "ICIR→RollHL filter (top2K then K)",
        "RollHL→ICIR filter",
        "MVO+LW (mu=6m)",
        "MVO+LW + RollHL filter",
        "MVO+LW (mu=6m × RollHL)",
    ]

    raw_rets = {n: run_rule(n) for n in rules}
    print("\n=== RAW ===")
    rows = []
    for n, r in raw_rets.items():
        m = metrics(r)
        cal = m["AR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
        print(f"  {n:<38} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
              f"MDD={m['MDD']:>+7.2%} Vol={m['Vol']:.2%} Calmar={cal:.2f} N={m['N']}")
        rows.append({"Variant":"raw","Rule":n, **m, "Calmar":cal})

    for tgt in [0.04, 0.06]:
        print(f"\n=== VOL-TARGET {tgt*100:.0f}% ===")
        for n, r in raw_rets.items():
            r_vt = vol_target(r, target=tgt)
            m = metrics(r_vt)
            cal = m["AR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
            rows.append({"Variant": f"vt_{tgt:.2f}","Rule":n, **m, "Calmar":cal})
        sub = pd.DataFrame([r for r in rows if r["Variant"] == f"vt_{tgt:.2f}"]).sort_values("SR", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row.Rule:<38} AR={row.AR:>+7.2%} SR={row.SR:>6.3f} "
                  f"MDD={row.MDD:>+7.2%} Calmar={row.Calmar:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v12_fade_on_icir.csv", index=False)

    # Pareto check at vt=6%
    print("\n========== Pareto vs ICIR_12m+invvol baseline (vt=6%) ==========")
    sub6 = df[df.Variant == "vt_0.06"].set_index("Rule")
    if "ICIR_12m+invvol" in sub6.index:
        b = sub6.loc["ICIR_12m+invvol"]
        print(f"Baseline ICIR_12m+invvol: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%} Cal={b.Calmar:.2f}")
        for n, row in sub6.sort_values("SR", ascending=False).iterrows():
            if n == "ICIR_12m+invvol": continue
            dar = row.AR - b.AR
            dsr = row.SR - b.SR
            dmdd = row.MDD - b.MDD
            tag = "  ★ PARETO" if (dar>=0 and dsr>=0 and dmdd>=0) else ""
            if (dar < 0 and dsr < 0 and dmdd < 0): tag = "  ✗ FADE-aug LOSES"
            print(f"  {n:<38} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")

    print("\n========== Pareto vs MVO+LW (mu=6m) baseline (vt=6%) ==========")
    if "MVO+LW (mu=6m)" in sub6.index:
        b = sub6.loc["MVO+LW (mu=6m)"]
        print(f"Baseline MVO+LW: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%} Cal={b.Calmar:.2f}")
        for n, row in sub6.sort_values("SR", ascending=False).iterrows():
            if n == "MVO+LW (mu=6m)": continue
            dar = row.AR - b.AR; dsr = row.SR - b.SR; dmdd = row.MDD - b.MDD
            tag = "  ★ PARETO" if (dar>=0 and dsr>=0 and dmdd>=0) else ""
            print(f"  {n:<38} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
