# Alpha Hunting – High‑Dimensional AI Quant System

A production‑ready framework for developing and deploying deep learning‑based alpha models.

## Features
- Hybrid Transformer‑GRU architecture for sequential alpha extraction.
- Leakage‑proof feature engineering pipeline (20+ indicators).
- Regime‑aware validation.
- Containerised with Docker and CI/CD out of the box.

## Quick Start

```bash
# Install
uv sync

# Train and backtest
make train

# Run a custom experiment
python scripts/run_experiment.py --config configs/my_config.yaml