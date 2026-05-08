"""
Anchor v16 — transaction-cost-adjusted comparison.

Theory (DeMiguel-Garlappi-Uppal 2009 RFS): MVO often loses to naive 1/N
under realistic transaction costs because (1) MVO weights are unstable
across rebalances → high turnover, (2) estimation error in mu/Sigma is
amplified by extreme weights.

Hypothesis: MVO+LW (mu=6m) wins gross-of-cost on Alpha158 because of low
realized vol and aggressive leverage. But its concentration may cause high
turnover. FADE-augmented strategies (more diversified) may win net-of-cost.

We measure:
  - Monthly turnover = sum |w_t - w_{t-1}| (one-way)
  - Cost-adjusted return = raw_return - turnover * cost_bps
  - For cost_bps in {0, 5, 10, 20, 50} bps

Compare AR/SR/MDD/Calmar net of costs at each level.
Pareto check vs MVO+LW [SOTA] under each cost regime.
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


def mvo_lw_weights(returns_df, mu):
    if returns_df.shape[1] < 2:
        return pd.Series(1.0/returns_df.shape[1], index=returns_df.columns)
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

    def factor_recent_vol(t, cands, window=12):
        co = t - pd.DateOffset(months=window)
        sub = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), cands].dropna(how="all")
        if sub.empty: return pd.Series(0.05, index=cands)
        return sub.std().replace(0, 0.05).reindex(cands).fillna(0.05)

    def icir_score(t, cands, lookback=12):
        sub = trailing_window(t, cands, lookback)
        if sub.empty: return pd.Series(dtype=float)
        mu = sub.mean(); sd = sub.std().replace(0, np.nan)
        return (mu / sd).abs().fillna(0)

    def ic_lookback(t, cands, lookback=3):
        co = t - pd.DateOffset(months=lookback)
        recent = ic_mat_signed.loc[(ic_mat_signed.index > co) & (ic_mat_signed.index <= t), cands].dropna(how="all")
        if recent.empty: return pd.Series(dtype=float)
        return recent.mean().abs()

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    K = 10
    cs_vol = 0.10

    # ---- Track weight series for each rule ----
    def run_rule_with_weights(name):
        """Returns: returns_series, weights_series_per_t (dict t -> Series)."""
        rets, weights_by_t = {}, {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            win = trailing_window(t, cands, 12)
            if win.shape[1] < 5: continue

            if name == "MVO+LW":
                cols = win.columns.tolist()
                mu = win.iloc[-6:].mean().values
                w_full = mvo_lw_weights(win, mu)
                top = w_full.nlargest(K)
            elif name == "ICIR_12m+invvol":
                sel = topk(icir_score(t, cands), cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv)
                w = w / w.sum()
                top = w
            elif name == "ICmom3+IC0+RollHL+invvol":
                ic0_proxy = ic_lookback(t, cands, 6)  # proxy; using 6m as ic0 stand-in
                z = (zscore(ic_lookback(t, cands, 3)).fillna(0) +
                     zscore(ic0_proxy).fillna(0))
                if t in rolling_hl.index:
                    z = z + zscore(rolling_hl.loc[t].reindex(cands)).fillna(0)
                sel = topk(z, cands, K)
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv); w = w / w.sum()
                top = w
            elif name == "Equal-K (1/N)":
                sel = topk(icir_score(t, cands), cands, K)
                top = pd.Series(1.0/K, index=sel)
            elif name == "MVO+LW + RollHL pre-filter (top80%)":
                if t not in rolling_hl.index: continue
                hl = rolling_hl.loc[t, cands].dropna()
                if hl.empty: continue
                threshold = hl.quantile(0.20)
                kept = hl[hl >= threshold].index.tolist()
                if len(kept) < K: continue
                win2 = trailing_window(t, kept, 12)
                if win2.shape[1] < 5: continue
                mu2 = win2.iloc[-6:].mean().values
                w_full = mvo_lw_weights(win2, mu2)
                top = w_full.nlargest(K)
            else:
                continue

            sel = list(top.index)
            v = signed_ic_at(t, sel)
            if v.empty: continue
            w_use = top.reindex(v.index).fillna(0)
            if w_use.sum() <= 0: continue
            r_t = float((v * w_use).sum() / w_use.sum())
            rets[t] = r_t * cs_vol
            weights_by_t[t] = w_use / w_use.sum()
        return pd.Series(rets).sort_index(), weights_by_t

    rules = ["MVO+LW", "ICIR_12m+invvol", "ICmom3+IC0+RollHL+invvol",
             "Equal-K (1/N)", "MVO+LW + RollHL pre-filter (top80%)"]

    raw_data = {}
    for name in rules:
        rets, w_t = run_rule_with_weights(name)
        raw_data[name] = (rets, w_t)
        print(f"  {name}: {len(rets)} months, {len(w_t)} weight snapshots")

    # ---- Compute turnover and cost-adjusted returns ----
    def compute_turnover_series(weights_by_t):
        ts = sorted(weights_by_t.keys())
        if len(ts) < 2: return pd.Series(dtype=float)
        out = {}
        prev_w = None
        all_factors = set()
        for t in ts: all_factors.update(weights_by_t[t].index)
        all_factors = sorted(all_factors)
        for t in ts:
            w_now = weights_by_t[t].reindex(all_factors).fillna(0)
            if prev_w is None:
                turnover = w_now.abs().sum()  # initial buy
            else:
                turnover = (w_now - prev_w).abs().sum()
            out[t] = float(turnover)
            prev_w = w_now
        return pd.Series(out).sort_index()

    print("\n=== Turnover (avg / median per rebalance) ===")
    turnover_data = {}
    for name, (rets, w_t) in raw_data.items():
        turn = compute_turnover_series(w_t)
        turnover_data[name] = turn
        print(f"  {name:<40} mean turnover = {turn.mean():.2%}  median = {turn.median():.2%}")

    # ---- Apply cost regimes ----
    cost_levels_bps = [0, 5, 10, 20, 50]
    print("\n=== Net-of-cost performance (vt=6%) ===")
    rows = []
    for cost_bps in cost_levels_bps:
        cost_frac = cost_bps / 10000
        print(f"\n  --- Cost = {cost_bps} bps ---")
        for name in rules:
            rets, _ = raw_data[name]
            turn = turnover_data[name]
            # Subtract per-month transaction cost = cost_frac * turnover
            # Cost is on portfolio NAV before vol-target leverage; for vol-targeted realised return,
            # cost should be on gross exposure (which scales with leverage). Conservatively:
            #   net_ret_t = ret_t - cost_frac * turnover_t * leverage_t
            # But we apply cost to RAW returns first, then vol-target.
            cost_series = turn.reindex(rets.index).fillna(0) * cost_frac
            net_rets = rets - cost_series
            net_vt   = vol_target(net_rets, target=0.06)
            m_ = metrics(net_vt)
            print(f"    {name:<40} AR={m_['AR']:>+7.2%} SR={m_['SR']:>6.3f} "
                  f"MDD={m_['MDD']:>+7.2%} Cal={m_['Calmar']:.2f}")
            rows.append({
                "cost_bps": cost_bps, "Rule": name, **m_,
                "mean_turnover": turn.mean(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v16_txcost.csv", index=False)

    # ---- Pareto cross-over analysis ----
    print("\n========== When does each rule beat MVO+LW (vt=6%) ==========")
    for cost_bps in cost_levels_bps:
        sub = df[df.cost_bps == cost_bps].set_index("Rule")
        if "MVO+LW" not in sub.index: continue
        b = sub.loc["MVO+LW"]
        print(f"\n  Cost={cost_bps}bps  MVO+LW: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%} Cal={b.Calmar:.2f}")
        for n, row in sub.iterrows():
            if n == "MVO+LW": continue
            dar = row.AR - b.AR
            dsr = row.SR - b.SR
            dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0: tag = "  ★ BEATS MVO PARETO"
            elif dar > 0 and dsr > 0: tag = "  ◇ AR+SR up"
            elif dsr > 0 and dmdd > 0: tag = "  ◇ SR+MDD up"
            print(f"    {n:<40} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
