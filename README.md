# FADE: Factor Alpha Decay Estimation

Predicting the IC half-life of quantitative factors using survival analysis.

## Overview

Existing AI-driven factor mining work focuses on *generating* new factors to counteract alpha decay. None address the dual problem: **given a newly discovered factor, can we predict how quickly it will decay?**

FADE frames this as a **survival analysis problem**: given a factor's structural features and early IC (Information Coefficient) dynamics, predict its half-life in months.

## Current Results

Best model: **Gradient Boosting Survival (GBS)**

| Model | C-index | MAE (months) |
|-------|---------|--------------|
| Global Mean Baseline | 0.500 | 4.28 |
| XGBoost Regression | 0.702 | 3.63 |
| Cox Proportional Hazards | 0.623 | 3.74 |
| Random Survival Forest | 0.737 | 2.67 |
| **GBS (ours)** | **0.821** | **2.59** |
| DeepSurv | 0.675 | 8.47 |

Ablation: IC dynamics features (momentum, autocorrelation, early IC mean/std) drive most of the predictive power. Structural features add ~3% C-index on top.

## Pipeline

```
Factor Expressions ──► IC Computation (Qlib/CSI500) ──► Half-Life Labels
        │                                                        │
        └──► Structural Features ─┐                             │
                                  ├──► Survival Model ──► Predicted HL
Early IC Dynamics ────────────────┤
                                  │
Market Environment ───────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1  # required on macOS

python3 experiments/01_data_generation.py --config configs/default.yaml
python3 experiments/02_feature_extraction.py --config configs/default.yaml
python3 experiments/03_train_models.py --config configs/default.yaml
python3 experiments/04_evaluation.py --config configs/default.yaml
python3 experiments/05_ablation.py --config configs/default.yaml
python3 experiments/06_portfolio_simulation.py --config configs/default.yaml
```

## For AI Assistants

See **[CLAUDE.md](CLAUDE.md)** for a complete handoff guide including:
- Current status of all stages
- Diagnosed issues with the synthetic data
- What needs to be done next (switch to Alpha158)
- Technical gotchas (OpenMP conflict, XGBoost bool dtype, etc.)

## Project Structure

```
FADE/
├── configs/default.yaml          # Main config
├── src/
│   ├── data/                     # IC computation, labeling, factor generation
│   ├── features/                 # Structural / IC / market feature extractors
│   └── models/                   # Cox, GBS, RSF, DeepSurv, LSTMSurv, XGBoost
├── experiments/                  # Stage 01–07 scripts
└── outputs/                      # Result CSVs
```

## License

MIT
