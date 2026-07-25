# Reproducibility Report

## Paper

Hybrid Model-Aided Learning for 5G-NTN Handover in High-Mobility Platforms,
IEEE INFOCOM Workshops 2025.

## Target

This package targets result-level reproduction. It checks whether a
Transformer-aided A2C handover policy can reduce handovers and improve reward
and demand satisfaction relative to DQN and random baselines in a synthetic
5G-NTN/LEO high-mobility environment.

## Known Gaps

- The official source code was not found.
- The original 8,441-instance dataset is not publicly packaged with the paper.
- Several implementation details are underspecified, including exact neural
  network widths, random seeds, congestion process, and training hardware.

## Defaults Chosen

- 8,441 timesteps at 10-second intervals.
- 25 satellites and 8 synthetic high-mobility airplane routes.
- CelesTrak Starlink TLE data is attempted first; a deterministic Walker-like
  LEO constellation is used offline.
- Minimum elevation is 20 degrees.
- Demand is sampled in `[0.2, 0.5]`.
- Reward is `QoS - 0.05 * handover_indicator`.
- Proposed model uses Transformer prediction features plus masked A2C.
- DQN baseline receives no prediction features.

## Acceptance Criteria

- Proposed A2C should show fewer handovers than DQN and random policy.
- Proposed A2C should show higher tail-average reward than DQN and random.
- Proposed A2C should show higher demand satisfaction than DQN and random.
- Ablation should show Transformer+A2C no worse than vanilla actor-critic in
  reward trend.

## Running

Use `python -m ntn_repro.reproduce_all --fast` for a smoke test and
`python -m ntn_repro.reproduce_all --full` for the paper-sized run.

