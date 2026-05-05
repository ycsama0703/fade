# FADE: Factor Alpha Decay Estimation

Predicting the IC half-life of quantitative factors using factor structural features, early IC dynamics, and market environment signals.

## Motivation

Existing AI-driven factor mining work (AlphaAgent, FactorMiner, QuantaAlpha, etc.) focuses on *generating* new factors to counteract alpha decay. None of them address the dual problem: **given a newly discovered factor, can we predict how quickly it will decay?**

This project frames factor decay prediction as a **survival analysis problem** and trains models to estimate per-factor IC half-life.

## Research Question

> Given a formulaic factor `f` and its early IC time series, predict its IC half-life `H` (months until rolling IC drops below 50% of its initial value).

## Approach

```
Factor Expression  ──┐
                     │
Early IC Dynamics  ──┼──► Cox Proportional Hazards / XGBoost ──► Predicted Half-Life
                     │
Market Environment ──┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1: Generate factor pool and compute IC time series
python experiments/01_data_generation.py --config configs/default.yaml

# Stage 2: Extract features
python experiments/02_feature_extraction.py --config configs/default.yaml

# Stage 3: Train models
python experiments/03_train_models.py --config configs/default.yaml

# Stage 4: Evaluate
python experiments/04_evaluation.py --config configs/default.yaml
```

## Project Structure

```
FADE/
├── docs/
│   ├── SCOPE.md            # Detailed research scope
│   └── EXPERIMENTS.md      # Experimental plan
├── src/
│   ├── data/               # Factor generation, IC computation, labeling
│   ├── features/           # Structural / early-IC / market-env features
│   ├── models/             # Cox survival, XGBoost, baselines
│   └── evaluation/         # Metrics
├── experiments/            # Stage scripts (01-04)
├── configs/                # YAML configs
├── notebooks/              # Exploratory analysis
└── tests/
```

## Documentation

- [Detailed Research Scope](docs/SCOPE.md)
- [Experimental Plan](docs/EXPERIMENTS.md)

## License

MIT
