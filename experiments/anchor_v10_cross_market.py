"""
Anchor v10 — cross-market validation (CSI300 vs CSI500).

Same strict-OOS comparison as anchor_v7 but on a DIFFERENT market universe
(CSI300 — 300 large-cap stocks vs CSI500 — 500 mid-cap). The factor library
(Alpha158) and methodology are identical, only the underlying stock universe
differs.

If FADE's Pareto improvement holds on CSI300 too, the result is not specific
to one stock universe — strong evidence the method generalizes.
"""
from __future__ import annotations
import os, sys, argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/yuncongliu/projects/FADE")
sys.path.insert(0, str(ROOT))

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


def run_anchor(data_dir: str, label: str):
    data_dir = ROOT / data_dir
    print(f"\n{'='*60}")
    print(f"  CROSS-MARKET ANCHOR — {label}  (data_dir={data_dir.name})")
    print(f"{'='*60}")
    ic = pd.read_parquet(data_dir / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(data_dir / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    X = pd.read_parquet(data_dir / "feature_matrix.parquet")

    ic0_abs, signs = {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic
        if len(g) < 6: continue
        first6 = g.iloc[:6]
        ic0_abs[fid] = abs(first6.mean())
        signs[fid]   = 1.0 if first6.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)
    print(f"  Pool: {len(ic0)} factors  IC range "
          f"{ic.date.min().date()} → {ic.date.max().date()}")

    cutoff = "2017-01-01"
    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp(cutoff)]
    print(f"  Cutoff {cutoff}  Sim {len(sim_dates)} months")

    censored = censored_labels(ic, reg, cutoff)
    print(f"  Censored L1 event rate: {censored.L1_event.mean():.1%}")

    rolling_hl = build_rolling_pred_hl(ic, X, censored, sim_dates)

    def active_at(t):
        return [fid for fid in signs
                if pd.Timestamp(reg_idx.loc[fid, "discovery_date"]) <= t
                and months_between(pd.Timestamp(reg_idx.loc[fid, "discovery_date"]), t) >= 6]

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

    def factor_recent_vol(t, cands, window=12):
        co = t - pd.DateOffset(months=window)
        recent = ic[(ic.factor_id.isin(cands)) & (ic.date > co) & (ic.date <= t)]
        if recent.empty: return pd.Series(1.0, index=cands)
        return recent.groupby("factor_id")["ic"].std().reindex(cands).fillna(0.05)

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

    def run_rule(K, mode):
        rets = {}
        for t in sim_dates:
            cands = active_at(t)
            if len(cands) < K: continue
            mom3  = ic_lookback(t, cands, 3)
            mom12 = ic_lookback(t, cands, 12)
            if mode == "ICmom3+IC0":
                scores = [mom3, ic0.reindex(cands)]
                weights = None
            elif mode == "ICmom12":
                scores = [mom12]
                weights = None
            elif mode == "ICmom3+IC0+RollHL":
                scores = [mom3, ic0.reindex(cands)]
                if t in rolling_hl.index:
                    scores.append(rolling_hl.loc[t].reindex(cands))
                weights = None
            elif mode == "ICmom3+IC0+RollHL_invvol":
                scores = [mom3, ic0.reindex(cands)]
                if t in rolling_hl.index:
                    scores.append(rolling_hl.loc[t].reindex(cands))
                weights = None  # will compute after sel
            sel = select_combo(cands, t, K, scores)
            if len(sel) < 3: continue
            v_series = signed_ic(t, sel)
            if v_series.empty: continue
            if mode == "ICmom3+IC0+RollHL_invvol":
                rv = factor_recent_vol(t, sel)
                w = (1.0 / rv).reindex(v_series.index).fillna(0)
                if w.sum() > 0:
                    v = float((v_series * w).sum() / w.sum())
                else:
                    v = float(v_series.mean())
            else:
                v = float(v_series.mean())
            rets[t] = v * 0.10
        return pd.Series(rets).sort_index()

    K = 10
    print(f"\n  K={K}, vt=6%:")
    rules = ["ICmom3+IC0", "ICmom12", "ICmom3+IC0+RollHL", "ICmom3+IC0+RollHL_invvol"]
    rows = []
    for mode in rules:
        r = vol_target(run_rule(K, mode), target=0.06)
        m = metrics(r)
        cal = m["AR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
        print(f"    {mode:<32} AR={m['AR']:>+7.2%} SR={m['SR']:>6.3f} "
              f"MDD={m['MDD']:>+7.2%} Calmar={cal:.2f}")
        rows.append({"market": label, "Rule": mode, **m, "Calmar": cal})

    # Pareto check
    print(f"\n  Pareto check (FADE vs ICmom3+IC0):")
    df = pd.DataFrame(rows)
    base = df[df.Rule == "ICmom3+IC0"].iloc[0]
    for _, row in df.iterrows():
        if row.Rule == "ICmom3+IC0": continue
        dar  = row.AR  - base.AR
        dsr  = row.SR  - base.SR
        dmdd = row.MDD - base.MDD
        tag = "  ★ PARETO" if (dar>=0 and dsr>=0 and dmdd>=0) else ""
        print(f"    {row.Rule:<32} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")

    return rows


def main():
    rows = []
    rows += run_anchor("data",         "CSI500 (original)")
    rows += run_anchor("data_csi300",  "CSI300")
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "anchor_v10_cross_market.csv", index=False)


if __name__ == "__main__":
    main()
