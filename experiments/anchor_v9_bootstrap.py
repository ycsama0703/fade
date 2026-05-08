"""
Anchor v9 — bootstrap robustness over factor universe.

For B trials, randomly subsample N_sub of 141 Alpha158 factors. For each
subsampled universe, run the strict-OOS comparison:
  ICmom3+IC0           (no FADE baseline)
  ICmom3+IC0+RollHL    (FADE)

at K ∈ {5, 10, 20}, vol_target = 6%, cutoff = 2017-01.

Tracks distribution of ΔAR, ΔSR, ΔMDD across trials and reports the fraction
of trials where ICmom3+IC0+RollHL strictly Pareto-dominates ICmom3+IC0.

Note on methodology: rolling-feature pred_HL is computed using the model
TRAINED ON THE FULL 141 with cutoff-2017 censored labels. Per trial we
subsample only the SELECTION pool, not the model training. This tests pool
sensitivity (would the result hold on a different 100-factor universe?).
"""
from __future__ import annotations
import os, sys, random
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
    print(f"Pool: {len(ic0)} | Cutoff: {cutoff} | Sim: {len(sim_dates)} months")

    print("[censored labels + rolling pred_HL] (computed once on full pool) ...")
    censored = censored_labels(ic, reg, cutoff)
    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)
    print(f"  rolling_hl shape: {rolling_hl.shape}")

    def active_at(t, pool):
        out_ = []
        for fid in pool:
            if fid not in signs: continue
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out_.append(fid)
        return out_

    def signed_ic(t, fids):
        sub = ic[(ic.factor_id.isin(fids)) & (ic.date == t)]
        if sub.empty: return pd.Series(dtype=float)
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return v

    def ic_lookback(t, cands, lookback=3):
        co = t - pd.DateOffset(months=lookback)
        recent = ic[(ic.factor_id.isin(cands)) & (ic.date > co) & (ic.date <= t)]
        if recent.empty: return pd.Series(dtype=float)
        return recent.groupby("factor_id")["ic"].apply(
            lambda s: float((s * signs.get(s.name, 1.0)).mean())
        ).abs()

    def zscore(s):
        m, sd = s.mean(), s.std()
        return (s - m) / (sd if sd > 0 else 1)

    def select_combo(cands, t, K, scores):
        z = None
        for s in scores:
            zs = zscore(s.reindex(cands)).fillna(0)
            z = zs if z is None else z + zs
        if z is None: return cands[:K]
        return z.nlargest(min(K, len(z))).index.tolist()

    def composite_return(t, sel):
        v = signed_ic(t, sel)
        return float(v.mean()) if not v.empty else np.nan

    def run_rule(pool, K, mode):
        """mode in {'mom3+ic0', 'mom3+ic0+rollhl', 'mom12'}."""
        rets = {}
        for t in sim_dates:
            cands = active_at(t, pool)
            if len(cands) < K: continue
            if mode == "mom12":
                mom12 = ic_lookback(t, cands, 12)
                scores = [mom12]
            else:
                mom3 = ic_lookback(t, cands, 3)
                scores = [mom3, ic0.reindex(cands)]
                if mode == "mom3+ic0+rollhl" and t in rolling_hl.index:
                    scores.append(rolling_hl.loc[t].reindex(cands))
            sel = select_combo(cands, t, K, scores)
            if len(sel) < 3: continue
            v = composite_return(t, sel)
            if not np.isnan(v): rets[t] = v * 0.10
        return pd.Series(rets).sort_index()

    # ---- Bootstrap loop ----
    all_factors = sorted(ic0.index.tolist())
    rng = random.Random(42)
    B = 50
    N_sub = 100
    K_values = [5, 10, 20]

    print(f"\nBootstrap: B={B} trials, subsample {N_sub} of {len(all_factors)} factors, "
          f"K ∈ {K_values}")
    print(f"Comparing ICmom3+IC0 (base) vs ICmom3+IC0+RollHL (FADE) at vt=6%")

    rows = []
    for b in range(B):
        pool = rng.sample(all_factors, N_sub)
        for K in K_values:
            r_b3 = vol_target(run_rule(pool, K, "mom3+ic0"),         target=0.06)
            r_b12= vol_target(run_rule(pool, K, "mom12"),            target=0.06)
            r_f  = vol_target(run_rule(pool, K, "mom3+ic0+rollhl"),  target=0.06)
            m_b3, m_b12, m_f = metrics(r_b3), metrics(r_b12), metrics(r_f)
            for base_name, m_b in [("ICmom3+IC0", m_b3), ("ICmom12", m_b12)]:
                rows.append({
                    "trial": b, "K": K, "baseline": base_name,
                    "AR_base": m_b["AR"],   "AR_fade": m_f["AR"],
                    "SR_base": m_b["SR"],   "SR_fade": m_f["SR"],
                    "MDD_base": m_b["MDD"], "MDD_fade": m_f["MDD"],
                    "dAR": m_f["AR"] - m_b["AR"],
                    "dSR": m_f["SR"] - m_b["SR"],
                    "dMDD": m_f["MDD"] - m_b["MDD"],
                    "pareto": int((m_f["AR"] >= m_b["AR"]) and
                                  (m_f["SR"] >= m_b["SR"]) and
                                  (m_f["MDD"] >= m_b["MDD"])),
                    "ar_up":  int(m_f["AR"]  > m_b["AR"]),
                    "sr_up":  int(m_f["SR"]  > m_b["SR"]),
                    "mdd_up": int(m_f["MDD"] > m_b["MDD"]),
                })
        if (b + 1) % 10 == 0:
            print(f"  trials done: {b+1}/{B}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v9_bootstrap.csv", index=False)

    print("\n========== Bootstrap results (vt=6%, cutoff=2017) ==========")
    for base_name in ["ICmom3+IC0", "ICmom12"]:
        print(f"\n>>> FADE (ICmom3+IC0+RollHL) vs {base_name} <<<")
        for K in K_values:
            sub = df[(df.K == K) & (df.baseline == base_name)]
            print(f"\nK={K}  ({N_sub}-of-{len(all_factors)} factor subsamples × {B} trials)")
            print(f"  ΔAR   : mean {sub.dAR.mean():+.2%}  std {sub.dAR.std():.2%}  "
                  f"median {sub.dAR.median():+.2%}  pct>0: {(sub.dAR>0).mean():.0%}")
            print(f"  ΔSR   : mean {sub.dSR.mean():+.3f}  std {sub.dSR.std():.3f}  "
                  f"median {sub.dSR.median():+.3f}  pct>0: {(sub.dSR>0).mean():.0%}")
            print(f"  ΔMDD  : mean {sub.dMDD.mean():+.2%}  std {sub.dMDD.std():.2%}  "
                  f"median {sub.dMDD.median():+.2%}  pct>0: {(sub.dMDD>0).mean():.0%}")
            print(f"  Strict Pareto (all 3 axes ≥): {sub.pareto.mean():.0%} of trials")
            two_of_three = ((sub.ar_up + sub.sr_up + sub.mdd_up) >= 2).mean()
            print(f"  Win on 2-of-3 axes: {two_of_three:.0%}")
            calmar_base = sub.AR_base / sub.MDD_base.abs()
            calmar_fade = sub.AR_fade / sub.MDD_fade.abs()
            print(f"  Calmar base : mean {calmar_base.mean():.2f}  median {calmar_base.median():.2f}")
            print(f"  Calmar fade : mean {calmar_fade.mean():.2f}  median {calmar_fade.median():.2f}")
            ratio = calmar_fade.mean() / calmar_base.mean() if calmar_base.mean() != 0 else np.nan
            print(f"  Calmar fade/base ratio (means): {ratio:.2f}×")


if __name__ == "__main__":
    main()
