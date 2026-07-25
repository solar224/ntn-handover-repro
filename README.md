# NTN Handover Reproduction Package

This repository is a result-level reproduction scaffold for:

> Hybrid Model-Aided Learning for 5G-NTN Handover in High-Mobility Platforms

It rebuilds a synthetic 5G-NTN/LEO handover environment, trains a
Transformer trajectory predictor, and compares Transformer-aided A2C with
DQN, random policy, and a vanilla actor-critic ablation.

The original paper does not publish the exact dataset or code. This package
therefore uses a fixed, traceable synthetic data pipeline with optional
CelesTrak TLE download and a deterministic fallback constellation.

## Quick Start

```bash
cd ntn-handover-repro
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python -m ntn_repro.data build --config configs/paper_result.yaml
python -m ntn_repro.train_transformer --horizon 5
python -m ntn_repro.train_transformer --horizon 25
python -m ntn_repro.train_rl --agent a2c
python -m ntn_repro.train_rl --agent dqn
python -m ntn_repro.train_rl --agent actor_critic
python -m ntn_repro.train_rl --agent random
python -m ntn_repro.evaluate --runs 5
python -m ntn_repro.plot --paper-figures
```

For a smoke run:

```bash
python -m ntn_repro.reproduce_all --fast
```

For the paper-sized run:

```bash
python -m ntn_repro.reproduce_all --full
```

## Main Outputs

- `artifacts/data/ntn_dataset.npz`: generated satellite/airplane features
- `artifacts/data/metadata.json`: TLE or synthetic constellation provenance
- `artifacts/checkpoints/`: Transformer and RL checkpoints
- `artifacts/metrics/`: per-episode metrics and evaluation summaries
- `artifacts/figures/`: reproduced Figure 2/Figure 3 style plots
- `docs/reproducibility_report.md`: assumptions and result interpretation

## Reproduction Notes

- The target is trend-level reproduction, not exact point-by-point matching.
- The dataset has 8,441 timesteps at 10-second intervals by default.
- The default action space is `0 = keep`, `1..25 = switch/select satellite`.
- The default reward is `QoS - 0.05 * handover_indicator`.
- If `skyfield` cannot be used or CelesTrak is unavailable, the data builder
  falls back to a deterministic Walker-like LEO constellation and records this
  in metadata.

