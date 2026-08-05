# Paper Alignment Audit

Paper: *Hybrid Model-Aided Learning for 5G-NTN Handover in High-Mobility Platforms*  
DOI: <https://doi.org/10.1109/INFOCOMWKSHPS65812.2025.11152861>  
Author manuscript: <https://orbilu.uni.lu/bitstream/10993/64595/1/HandoverRL_paper.pdf>

## Bottom Line

The previous artifacts are not a numerical reproduction of the paper. The
result gap is not explained by random seeds alone.

There are two independent causes:

1. The generated dataset used the deterministic synthetic constellation, not
   the CelesTrak TLE propagation stated in the paper.
2. The previous implementation omitted or misimplemented several decisive
   experiment paths, including the 25-step A2C ablation and independent
   evaluation runs.

The paper also omits information required for exact numerical reproduction.
In particular, the reward scale in Figures 2b and 3 cannot be derived from the
published reward equation and disclosed parameter ranges.

## 1. Paper Requirements

### System model

- 25 LEO satellites propagated from CelesTrak TLEs.
- Minimum elevation angle: 20 degrees.
- Airplane demand in `[0.2, 0.5]`.
- Allocation:
  `alloc[k,n,t] = min(demand[k,t], 1 - congestion[n,t])`.
- Congestion update:
  `congestion[n,t+1] = congestion[n,t] + alloc[k,n,t]`.
- QoS follows Eqs. (3)-(4).
- Reward follows Eq. (14):
  `alpha * QoS - beta * handover_indicator`.

### Hybrid model

- Transformer input: historical airplane positions.
- Transformer output: one airplane position at `t + horizon`.
- RL state combines current position, predicted future position, demand,
  satellite position/mobility, and congestion. The surrounding text also
  mentions elevation and historical QoS.
- Action dimension: number of satellites plus one no-handover action.
- Proposed RL algorithm: A2C.

### Experiments

- 8,441 instances at 10-second intervals.
- 256 reported dataset features.
- Learning rate: `0.0001`.
- Batch size: `256`.
- Baselines: DQN and random.
- Ablations: A2C horizon 5, A2C horizon 25, and actor-critic horizon 5.
- Figure 2 demand satisfaction: `0.82` proposed, `0.18` DQN, `0.31` random.

## 2. Previous Artifact Evidence

The inspected `artifacts/data/metadata.json` reports:

- `source = synthetic`
- 25 synthetic satellites
- empty TLE lines

Dataset diagnostics:

- no visible satellite in `46.63%` of airplane-timesteps
- mean visible satellites: `0.706`
- maximum visible satellites: `3`
- stored arrays flatten to 533 scalar fields per timestep, not the reported
  256-feature schema
- RL state dimension is 158

The old environment treated a no-coverage timestep as an invalid action. The
observed invalid-action rates were therefore approximately the same as the
no-coverage rate:

| Variant | Invalid-action rate |
|---|---:|
| A2C | 45.15% |
| DQN | 44.17% |
| Random | 46.99% |
| Actor-Critic | 45.15% |

This penalty dominated the previous rewards.

The old tail summaries also contradicted the paper:

| Variant | Reward | Handovers | Satisfaction |
|---|---:|---:|---:|
| A2C | -6.842 | 39.60 | 0.549 |
| DQN | 9.798 | 26.06 | 0.591 |
| Random | -4.659 | 31.26 | 0.538 |
| Actor-Critic | -6.823 | 39.60 | 0.549 |

The A2C and actor-critic outputs were effectively duplicates: satisfaction
matched in all 300 episodes, handovers matched in 296 episodes, and reward
matched exactly in 281 episodes.

## 3. Confirmed Implementation Problems

### Fixed

- Transformer now predicts only the position at `t + horizon`, as specified by
  Eqs. (7)-(9).
- Missing Transformer checkpoints now fail explicitly. Oracle future positions
  are no longer used as a hidden fallback.
- Transformer validation uses a chronological holdout, and prediction-boundary
  rows are excluded instead of being filled with current true positions.
- The no-handover action remains valid during an outage, so lack of coverage is
  not mislabeled as an invalid agent action.
- Every configured airplane is processed at each physical timestep. Allocation
  is reserved by persistent satellite ID, so airplanes selecting the same
  satellite share its remaining capacity.
- Dataset congestion is interpreted as baseline load. Within-timestep
  allocations implement the paper's admission update; allocations are released
  before the next dataset timestep because the paper omits a release process.
- Discounting uses factor 1 between airplane decisions at the same physical
  timestep and applies \(\gamma\) only when advancing to the next timestep,
  matching the time index in Eq. (16).
- A2C now performs synchronous joint actor/value updates over n-step rollouts.
- The vanilla actor-critic ablation uses separate actor and critic updates.
- Metrics and checkpoints are horizon-specific:
  `a2c_h5`, `a2c_h25`, and `actor_critic_h5`.
- The full pipeline now trains the 25-step A2C variant.
- Figure 3 now loads all three required ablation variants.
- `evaluate --runs 5` now performs five independent evaluation seeds instead
  of relabeling one training CSV as one of five requested runs.
- Paper-result data generation now fails if CelesTrak/Skyfield propagation is
  unavailable. Synthetic fallback must be explicitly requested.
- Paper-sized comparisons use one common 8,391-transition window from the
  8,441-instance trace (history-valid row 24 through horizon-25-valid row
  8,415), with one action per airplane per physical timestep. This prevents
  the non-predictive, horizon-5, and horizon-25 variants from receiving
  different interaction budgets. It remains an explicit alignment assumption
  because the paper does not disclose episode construction.

### Still not reproducible from the paper

- Exact TLE snapshot and propagation start time.
- Airplane trajectory source and preprocessing.
- Exact 256-feature schema.
- Number of airplanes used jointly in each RL episode.
- Congestion initialization/process and precise multi-airplane update order.
- `alpha`, `beta`, discount factor, network widths, optimizer details, and
  random seeds.
- Reward scaling or additional penalties used to produce the approximately
  `-600,000` to positive reward range in the published figures.
- Training energy measurement method and hardware.

## 4. Interpretation Rule

A run may be called paper-framework-aligned only if:

1. metadata reports `source = celestrak_tle_live_skyfield` or
   `source = celestrak_tle_cache_skyfield`, or the verified recovery source
   `source = celestrak_tle_snapshot_skyfield`;
2. both Transformer horizons and all three RL ablation variants were trained;
3. independent evaluation runs completed;
4. the proposed A2C variants beat the baselines in the paper's qualitative
   ordering.

It must not be called an exact numerical reproduction unless the missing
dataset, reward scaling, and experiment construction are supplied by the
authors.

## 5. CelesTrak Retrieval Correction

The original downloader used Python's default `urllib` request. CelesTrak
returned HTTP 403, and the old code discarded the exception before silently
using synthetic satellites.

The corrected downloader:

- sends an identifiable User-Agent;
- preserves the exact HTTP/download error;
- does not retry HTTP 403;
- caches successful downloads for at least two hours;
- stores the raw TLE used by the dataset;
- uses the explicitly configured propagation start
  `2026-07-27T12:00:00+00:00`;
- retains the earlier 25-record snapshot for fixed-subset audit only;
- never falls back to synthetic data under `paper_result.yaml`.

The fixed-first-25 audit produced an 86.78% no-coverage rate, confirming that
an arbitrary catalog prefix is not a meaningful constellation. While awaiting
the authors' satellite IDs and scenario details, the repaired configuration:

- `source = celestrak_tle_cache_skyfield`;
- filters the full CelesTrak response to 1,585 records in the 500-600 km shell;
- exposes the 25 highest-elevation candidates per airplane-timestep;
- tracks persistent NORAD IDs independently of candidate slots;
- reports 0% no-visible timesteps and 11.557 mean visible candidates in the
  latest full rebuild;
- provenance validation passed.

This dynamic candidate rule is a documented reconstruction assumption, not a
detail confirmed by the paper.
