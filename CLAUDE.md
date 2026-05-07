# FADE — AI Handoff Guide

## What This Project Is

FADE (Factor Alpha Decay Estimation) predicts how quickly a quantitative factor's IC (Information Coefficient) will decay. Framed as a survival analysis problem: given a factor's structural features and early IC dynamics, predict its half-life in months.

Target venue: quantitative finance ML paper (ICML/NeurIPS workshop or AAAI FinAI track).

---

## Current State (as of 2026-05-07)

### What's Done and Working

**Stage 1–4 pipeline is complete and runs end-to-end.**

| Stage | Script | Status | Key Output |
|-------|--------|--------|------------|
| 1. Data generation | `experiments/01_data_generation.py` | ✓ Done | `data/ic_series_filtered.parquet`, `data/labels.parquet` |
| 2. Feature extraction | `experiments/02_feature_extraction.py` | ✓ Done | `data/feature_matrix.parquet` |
| 3. Model training | `experiments/03_train_models.py` | ✓ Done | `models/checkpoints/trained_models.pkl` |
| 4. Evaluation | `experiments/04_evaluation.py` | ✓ Done | `outputs/main_results.csv` |
| 5. Ablation | `experiments/05_ablation.py` | ✓ Done | `outputs/ablation_cox.csv`, `ablation_xgboost.csv` |
| 6. Portfolio sim | `experiments/06_portfolio_simulation.py` | ⚠ Weak results | See below |
| 7. Robustness | `experiments/07_strategy_robustness.py` | ⚠ Weak results | See below |

### Key Results (Stage 4)

Best model is **M3_gbs** (Gradient Boosting Survival from scikit-survival):

| Model | C-index | MAE | Spearman |
|-------|---------|-----|----------|
| B1 GlobalMean (baseline) | 0.500 | 4.28 | — |
| B2 TypeMean | 0.553 | 4.35 | 0.186 |
| B4 XGBoost | 0.702 | 3.63 | 0.743 |
| M1 Cox PH | 0.623 | 3.74 | 0.407 |
| M2 RSF | 0.737 | 2.67 | 0.708 |
| **M3 GBS** | **0.821** | **2.59** | **0.726** |
| D1 DeepSurv | 0.675 | 8.47 | 0.580 |
| D2 LSTMSurv | 0.568 | 8.62 | 0.320 |

### Key Results (Stage 5 — Ablation)

Feature groups: A = structural (tree depth, operator types), B = IC dynamics (ic_mean, ic_std, ic_momentum, etc.), C = market environment.

| Model | Best subset | C-index |
|-------|-------------|---------|
| Cox PH | B only | 0.619 |
| XGBoost | A+B | 0.690 |

**Conclusion: IC dynamics features (Group B) drive most of the predictive power.**

---

## The Core Problem That Needs Solving

### Prediction quality is strong (C-index=0.821). Portfolio downstream is weak.

**Root cause diagnosed**: The current data uses **synthetic factors** (randomly generated mathematical expressions applied to real CSI500 prices). These factors' IC does NOT permanently decay — it randomly bounces around zero. After the L1 half-life label triggers (rolling-3-month mean crosses zero), IC bounces back to positive with 72% probability. This means:

1. Timing predictions don't help (there's no sustained negative IC to avoid)
2. IC magnitude is nearly uncorrelated with half-life (r=0.035)
3. Any portfolio timing experiment produces weak results

### The Fix

Switch from synthetic factors to **Qlib Alpha158 factors** (158 real economic factors: momentum, reversal, quality, volatility, etc.). These have genuine economic decay mechanisms (arbitrage, information diffusion) so IC decays permanently once it starts. Qlib is already installed and data is at `~/.qlib/qlib_data/cn_data`.

**Alpha158 loader already exists at `src/data/alpha158_loader.py`** but Stage 1 is still using synthetic factors (`use_alpha158: false` in `configs/default.yaml`).

---

## What Needs to Be Done Next

### Priority 1: Switch to Alpha158 (rerun full pipeline)

1. Set `use_alpha158: true`, `use_synthetic: false` in `configs/default.yaml`
2. In `experiments/01_data_generation.py`, call `load_alpha158_expressions()` from `src/data/alpha158_loader.py`
3. Stage 2: Alpha158 factors have no AST structure → set structural features (Group A) to NaN for them, mark `source="alpha158"` in registry
4. Rerun stages 1→6 — Alpha158 IC computation is slow (~1–3 hrs for 158 factors × 3500 days)
5. Expected: portfolio simulation shows clear improvement when using our decay-timing predictions

### Priority 2: Portfolio Experiment Design (after Alpha158 data)

The correct experiment (already implemented in `experiments/06_portfolio_simulation.py`):

- **Hold-Forever**: include factor from discovery to end of sim (suffers negative IC after decay)
- **Fixed-K**: exit at disc + K months (K=6,12,24)
- **Predicted**: exit at disc + (pred_HL - 2) months ← our method (the -2 accounts for the 3-month rolling window lag in the label)
- **Oracle**: exit at disc + (true_HL - 2) months ← upper bound

With genuine IC decay (Alpha158), Predicted should clearly outperform Fixed-K strategies by capturing the full alpha period without the post-decay drag.

---

## Technical Notes & Gotchas

### Environment Setup
```bash
pip install -r requirements.txt
# Must set before importing torch + xgboost together:
export OMP_NUM_THREADS=1
```

### Critical macOS Fixes (already applied)
- `OMP_NUM_THREADS=1` at top of `experiments/03_train_models.py` — prevents PyTorch+XGBoost OpenMP conflict crash
- `tree_method="exact"` in `XGBoostRegressor` — `"hist"` segfaults with bool columns from `pd.get_dummies()`
- `.astype(float)` after every `pd.get_dummies()` call — bool dtype crashes XGBoost hist method
- `n_jobs=1` for RSF — multiprocessing fork + PyTorch causes segfault

### Data Split
Temporal split by discovery date (already implemented in Stage 3):
- Train: discovery_date < 2015-01-01
- Val: 2015-01 to 2017-01
- Test: ≥ 2017-01-01

### Label Definition
`L1_duration` = index where **3-month rolling mean IC first crosses zero** (opposite to IC_0 sign). This is a LAGGING indicator — actual IC flip starts ~2 months before the label triggers. When using predicted HL for portfolio timing, subtract 2 months: `exit_month = pred_HL - 2`.

### Model Interface
All models implement:
```python
model.predict_median_survival(X: pd.DataFrame) -> np.ndarray  # predicted HL in months
model.concordance_index(X, durations, events) -> float
```

Load trained models:
```python
import pickle
with open("models/checkpoints/trained_models.pkl", "rb") as f:
    models = pickle.load(f)
model = models["M3_gbs"]  # best model
```

### Cox Model Special Handling
Cox drops zero-variance columns and stores them:
```python
models["M1_cox"].dropped_cols_  # list of dropped column names
X_cox = X.drop(columns=models["M1_cox"].dropped_cols_)
```

### Sign Alignment
All factors are sign-aligned before computing composite IC: early IC sign determines if we flip the factor's IC contribution. This is standard practice — each factor contributes positively or negatively to its direction. Already implemented in all portfolio simulation scripts.

---

## File Map

```
FADE/
├── configs/
│   └── default.yaml              # Main config — change use_alpha158 here
├── src/
│   ├── data/
│   │   ├── alpha158_loader.py    # Load Alpha158 expressions from Qlib
│   │   ├── factor_generator.py   # Synthetic factor tree generation
│   │   ├── ic_computation.py     # compute_ic_series() via Qlib D.features()
│   │   └── label_generation.py   # compute_half_life_label(), generate_labels_for_panel()
│   ├── features/
│   │   ├── early_ic.py           # Group B: IC dynamic features
│   │   ├── structural.py         # Group A: factor expression features
│   │   └── market_env.py         # Group C: market environment features
│   └── models/
│       ├── baselines.py          # B1 GlobalMean, B2 TypeMean, B3 Hyperbolic
│       ├── cox_survival.py       # M1 Cox PH (lifelines)
│       ├── xgboost_baseline.py   # B4 XGBoost regression
│       ├── survival_ensemble.py  # M2 RSF, M3 GBS (scikit-survival)
│       └── deep_survival.py      # D1 DeepSurv, D2 LSTMSurv (PyTorch)
├── experiments/
│   ├── 01_data_generation.py     # Stage 1: IC computation + labels
│   ├── 02_feature_extraction.py  # Stage 2: feature matrix
│   ├── 03_train_models.py        # Stage 3: train all 9 models
│   ├── 04_evaluation.py          # Stage 4: C-index / MAE / Spearman
│   ├── 05_ablation.py            # Stage 5: feature group ablation
│   ├── 06_portfolio_simulation.py # Stage 6: timing experiment (needs Alpha158)
│   └── 07_strategy_robustness.py # Stage 7: multi-strategy robustness
├── data/                         # Generated data (gitignored)
├── models/checkpoints/           # Trained model pickle (gitignored)
└── outputs/                      # Result CSVs
```

---

## Running the Full Pipeline

```bash
# All stages in order:
python3 experiments/01_data_generation.py --config configs/default.yaml
python3 experiments/02_feature_extraction.py --config configs/default.yaml
python3 experiments/03_train_models.py --config configs/default.yaml
python3 experiments/04_evaluation.py --config configs/default.yaml
python3 experiments/05_ablation.py --config configs/default.yaml
python3 experiments/06_portfolio_simulation.py --config configs/default.yaml
python3 experiments/07_strategy_robustness.py --config configs/default.yaml
```

Stage 3 takes ~10 minutes. Stage 1 with Alpha158 will take 1–3 hours.
