"""
Anchor v4 — fix Oracle, beat ICmom3 baseline.

Key new questions:
  1. Is ICmom3 (3m IC-momentum top-K) really a tighter baseline than Top|IC0|?
  2. Can we improve ICmom3 by combining with pred_HL / RollHL / EIC?
     Specifically: Comp_ICmom3+RollHL, Comp_ICmom3+EIC, etc.
  3. Does pred_HL specifically help in BAD years for ICmom3 (regime stability)?

Also fixes Oracle to use ic[t] (same period as portfolio realization).
"""
from __future__ import annotations
import os, sys, pickle
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
        return dict(CR=np.nan, AR=np.nan, SR=np.nan, MDD=np.nan, N=0)
    cum = (1 + r).cumprod()
    cr  = float(cum.iloc[-1] - 1)
    ar  = float((1 + cr) ** (12 / len(r)) - 1)
    sr  = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(CR=cr, AR=ar, SR=sr, MDD=mdd, N=len(r))


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


def cv_eic(X, ic_panel, reg_idx, signs, seed=42, n_splits=5,
           start_month=7, end_month=18):
    from xgboost import XGBRegressor
    y_target = {}
    for fid in X.index:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic_panel[(ic_panel.factor_id == fid) & (ic_panel.date >= disc)] \
            .sort_values("date").ic
        if len(g) >= end_month:
            y_target[fid] = float((g.iloc[start_month-1:end_month] * signs.get(fid, 1.0)).mean())
    y = pd.Series(y_target).reindex(X.index).dropna()
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


def build_rolling_pred_hl(ic_panel, X_full, y, sim_dates, seed=42):
    m = GradientBoostingSurvival(random_state=seed)
    m.fit(X_full, y["L1_duration"], y["L1_event"].astype(int))
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


def main():
    ic = pd.read_parquet(DATA / "ic_series_filtered.parquet")
    ic["date"] = pd.to_datetime(ic["date"])
    reg = pd.read_csv(DATA / "factor_registry.csv")
    reg["discovery_date"] = pd.to_datetime(reg["discovery_date"])
    reg_idx = reg.set_index("factor_id")
    labels = pd.read_parquet(DATA / "labels.parquet").set_index("factor_id")
    X = pd.read_parquet(DATA / "feature_matrix.parquet")

    ic0_abs, signs = {}, {}
    for fid in reg.factor_id:
        disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
        g = ic[(ic.factor_id == fid) & (ic.date >= disc)].sort_values("date").ic.iloc[:6]
        if len(g) < 6: continue
        ic0_abs[fid] = abs(g.mean())
        signs[fid]   = 1.0 if g.mean() >= 0 else -1.0
    ic0 = pd.Series(ic0_abs)

    sim_dates = [d for d in sorted(ic.date.unique()) if d >= pd.Timestamp("2012-01-01")]
    print(f"Pool: {len(ic0)} | Sim months: {len(sim_dates)}")

    print("[CV] pred_HL ...")
    pred_hl = cv_pred_hl(X, labels)
    print("[CV] EIC ...")
    eic = cv_eic(X, ic, reg_idx, signs)
    print("[rolling] pred_HL ...")
    rolling_hl = build_rolling_pred_hl(ic, X, labels, sim_dates)

    z_ic0 = (ic0 - ic0.mean()) / ic0.std()
    z_lhl = (np.log1p(pred_hl) - np.log1p(pred_hl).mean()) / np.log1p(pred_hl).std()
    z_eic = (eic - eic.mean()) / eic.std()

    def active_at(t):
        out = []
        for fid in signs:
            disc = pd.Timestamp(reg_idx.loc[fid, "discovery_date"])
            if disc > t: continue
            if months_between(disc, t) < 6: continue
            out.append(fid)
        return out

    def topk(score, cands, k):
        s = score.reindex(cands).dropna()
        return s.nlargest(min(k, len(s))).index.tolist()

    def icmom_score(t, cands, lookback=3):
        cutoff = t - pd.DateOffset(months=lookback)
        recent = ic[(ic.factor_id.isin(cands)) &
                    (ic.date > cutoff) & (ic.date <= t)]
        if recent.empty: return pd.Series(dtype=float)
        adj = recent.groupby("factor_id")["ic"].apply(
            lambda s: float((s * signs.get(s.name, 1.0)).mean())
        ).abs()
        return adj

    def topk_t_dependent(score_t, cands, K):
        s = score_t.reindex(cands).dropna()
        return s.nlargest(min(K, len(s))).index.tolist()

    def comp_ic(t, sel):
        sub = ic[(ic.factor_id.isin(sel)) & (ic.date == t)]
        if sub.empty: return np.nan
        v = sub.set_index("factor_id").ic
        v = v * pd.Series(signs).reindex(v.index)
        return float(v.mean())

    # Fixed Oracle: pick by ic at month t (the same realization the portfolio receives)
    def oracle_at(t, cands, K):
        sub = ic[(ic.factor_id.isin(cands)) & (ic.date == t)]
        if sub.empty: return cands[:K]
        v = sub.set_index("factor_id").ic
        v = v * pd.Series({f: signs.get(f, 1) for f in v.index})
        return v.nlargest(min(K, len(v))).index.tolist()

    rows = []
    cs_vol = 0.10
    for K in [5, 10, 20]:
        print(f"\n=== K={K} ===")

        rules_with_t = {}

        rules_with_t["Top|IC0|"] = lambda c, t: topk(ic0, c, K)
        rules_with_t["ICmom3"] = lambda c, t: topk_t_dependent(icmom_score(t, c), c, K)
        rules_with_t["Composite_prod"] = lambda c, t: topk(ic0 * np.log1p(pred_hl), c, K)
        rules_with_t["Composite_static"] = lambda c, t: topk(z_ic0 + z_lhl, c, K)
        rules_with_t["Composite_EIC"] = lambda c, t: topk(z_ic0 + z_eic, c, K)

        # Comp_ICmom3 + various
        def make_rule(extra_score_callable):
            def fn(c, t):
                m_score = icmom_score(t, c)
                if m_score.empty: return c[:K]
                z_m = (m_score - m_score.mean()) / (m_score.std() if m_score.std()>0 else 1)
                extra = extra_score_callable(t, c)
                z_e = (extra - extra.mean()) / (extra.std() if extra.std()>0 else 1)
                combined = z_m.fillna(0).add(z_e.fillna(0), fill_value=0)
                return combined.nlargest(min(K, len(combined))).index.tolist()
            return fn

        rules_with_t["Comp_ICmom3+pred_HL"] = make_rule(
            lambda t, c: pred_hl.reindex(c)
        )
        rules_with_t["Comp_ICmom3+RollHL"] = make_rule(
            lambda t, c: rolling_hl.loc[t].reindex(c) if t in rolling_hl.index else pd.Series(dtype=float)
        )
        rules_with_t["Comp_ICmom3+EIC"] = make_rule(
            lambda t, c: eic.reindex(c)
        )
        rules_with_t["Comp_ICmom3+IC0"] = make_rule(
            lambda t, c: ic0.reindex(c)
        )
        rules_with_t["Oracle"] = lambda c, t: oracle_at(t, c, K)

        for name, fn in rules_with_t.items():
            rets = {}
            for t in sim_dates:
                cands = active_at(t)
                if len(cands) < K: continue
                sel = fn(cands, t)
                if len(sel) < 3: continue
                v = comp_ic(t, sel)
                if not np.isnan(v):
                    rets[t] = v * cs_vol
            r = pd.Series(rets).sort_index()
            mt = metrics(r)
            print(f"  {name:<22} CR={mt['CR']:>+8.2%} AR={mt['AR']:>+7.2%} "
                  f"SR={mt['SR']:>6.3f} MDD={mt['MDD']:>+7.2%} N={mt['N']}")
            rows.append({"K": K, "Rule": name, **mt})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_v4.csv", index=False)

    # Headline: which rule beats ICmom3 baseline on a Pareto basis?
    print("\n========== Pareto vs ICmom3 baseline ==========")
    for K in [5, 10, 20]:
        sub = df[df.K == K].set_index("Rule")
        if "ICmom3" not in sub.index: continue
        b = sub.loc["ICmom3"]
        print(f"\nK={K}  ICmom3 baseline: AR={b.AR:+.2%} SR={b.SR:.3f} MDD={b.MDD:+.2%}")
        for name, row in sub.iterrows():
            if name == "ICmom3": continue
            dar  = row.AR  - b.AR
            dsr  = row.SR  - b.SR
            dmdd = row.MDD - b.MDD
            tag = ""
            if dar >= 0 and dsr >= 0 and dmdd >= 0:
                tag = "  ★ PARETO-DOMINATES"
            elif (dar > 0 and dsr > 0):
                tag = "  ◇ AR & SR up"
            elif (dsr > 0 and dmdd > 0):
                tag = "  ◇ SR & MDD up"
            print(f"  {name:<22} ΔAR={dar:+.2%} ΔSR={dsr:+.3f} ΔMDD={dmdd:+.2%}{tag}")


if __name__ == "__main__":
    main()
