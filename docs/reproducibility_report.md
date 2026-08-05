# Reproducibility Report

## Paper

Hybrid Model-Aided Learning for 5G-NTN Handover in High-Mobility Platforms,
IEEE INFOCOM Workshops 2025.

## Target

This package targets framework- and trend-level reproduction. It checks whether a
Transformer-aided A2C handover policy can reduce handovers and improve reward
and demand satisfaction relative to DQN and random baselines in a reconstructed
5G-NTN/LEO high-mobility environment.

## Known Gaps

- The official source code was not found.
- The original 8,441-instance dataset is not publicly packaged with the paper.
- Several implementation details are underspecified, including exact neural
  network widths, random seeds, congestion process, reward scale used by the
  published plots, episode construction, and training hardware.
- The reward magnitude in the published figures is not derivable from
  Eqs. (14)-(16) and the disclosed demand/QoS ranges, so exact reward matching
  is not a defensible acceptance criterion without author code or clarification.

## Defaults Chosen

- 8,441 timesteps at 10-second intervals.
- 25 satellite candidate/action slots and 8 synthetic high-mobility airplane
  routes.
- Paper-result runs require CelesTrak Starlink TLE propagation. A deterministic
  Walker-like LEO constellation is available only as an explicit trend-only
  fallback.
- The repaired paper-result scenario filters a 500-600 km Starlink candidate
  pool and fills the 25 slots by current elevation at each airplane-timestep.
  NORAD IDs provide persistent identity across slot changes.
- Orbit propagation is pinned to `2026-07-27T12:00:00+00:00`, near the cached
  TLE epochs, so repeated builds do not silently change geometry.
- Minimum elevation is 20 degrees.
- Demand is sampled in `[0.2, 0.5]`.
- Baseline congestion spans `[0.05, 0.95]`; the paper discloses only `[0, 1]`.
- All eight local airplanes act at each physical timestep. Allocations to the
  same persistent satellite share remaining capacity and are released before
  the next dataset timestep.
- The return applies no extra discount between airplanes at the same physical
  timestep; `gamma` is applied once when the physical timestep advances.
- Reward is `QoS - 0.05 * handover_indicator`.
- Proposed model uses Transformer prediction features plus masked A2C.
- DQN baseline receives no prediction features.
- Transformer output is one predicted position at `t + horizon`, matching
  Eqs. (7)-(9).
- Transformer training uses a chronological 85/15 split.
- All paper-sized variants use the same 8,391-transition comparison window:
  row 24 is the first state with 25 history samples and row 8,415 is the final
  state with a valid 25-step future target. This keeps DQN, random, horizon-5,
  and horizon-25 interaction budgets equal.

## Acceptance Criteria

- Proposed A2C should show fewer handovers than DQN and random policy.
- Proposed A2C should show higher tail-average reward than DQN and random.
- Proposed A2C should show higher demand satisfaction than DQN and random.
- Ablation should show Transformer+A2C no worse than vanilla actor-critic in
  reward trend.
- A paper-result run must report `source = celestrak_tle_live_skyfield` or
  `source = celestrak_tle_cache_skyfield` in metadata. A checked-in,
  previously downloaded CelesTrak snapshot is reported as
  `source = celestrak_tle_snapshot_skyfield`.
- Evaluation must complete the requested independent runs; a single training
  CSV is not treated as five runs.
- Geometry validation must report a no-visible fraction below 1% before RL
  training; full validation rebuilds report between 0% and approximately
  0.0015%, depending on the propagation start time.

## Running

Use `python -m ntn_repro.reproduce_all --fast` for a smoke test and
`python -m ntn_repro.reproduce_all --full` for the paper-sized run.

