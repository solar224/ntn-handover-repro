# NTN Handover Reproduction Package

This repository provides a result-level reproduction of **Hybrid Model-Aided Learning for 5G-NTN Handover in High-Mobility Platforms**.

It builds a synthetic 5G-NTN/LEO handover environment, trains Transformer trajectory predictors, and compares Transformer-aided A2C against DQN, random policy, and a vanilla actor-critic ablation. Because the original paper does not publish its exact dataset or code, this project uses a fixed, traceable data pipeline with CelesTrak TLE propagation for paper-sized experiments and a deterministic synthetic constellation for smoke tests.

## Transformer-aided A2C Architecture

```mermaid
flowchart LR
    DATA[("NTN Dataset")]

    subgraph TP["1. Trajectory Prediction"]
        direction TB
        TLOAD["Transformer Loader<br/>25 historical positions<br/>mini-batch: 256"]
        TRANS["Transformer<br/>predict future position<br/>t + 5 or t + 25"]
        PRED["Predicted<br/>Airplane Position"]
        TLOAD --> TRANS --> PRED
    end

    ACTUAL["Actual Network State<br/>position, signal, congestion"]
    ENV["2. Augmented State<br/>& NTN Environment"]

    subgraph A2C["3. A2C Handover Agent"]
        direction TB
        ACTOR["Actor<br/>choose handover action"]
        CRITIC["Critic<br/>estimate state value"]
        ROLLOUT["A2C Loader<br/>32-step online rollout"]

        ACTOR --> ROLLOUT
        CRITIC --> ROLLOUT
        ROLLOUT -. "update" .-> ACTOR
        ROLLOUT -. "update" .-> CRITIC
    end

    DATA --> TLOAD
    DATA --> ACTUAL
    ACTUAL --> ENV
    PRED --> ENV
    ENV -- "State" --> ACTOR
    ENV -- "State" --> CRITIC
    ACTOR -- "Action" --> ENV
    ENV -- "Reward" --> ROLLOUT

    classDef source fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef model fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef environment fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef agent fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    class DATA,ACTUAL source
    class TLOAD,TRANS,PRED model
    class ENV environment
    class ACTOR,CRITIC,ROLLOUT agent
```

The Transformer loader converts 25 historical positions into mini-batches for future-position prediction. A2C learns from a 32-step online rollout collected from `HandoverEnv`, rather than a static dataset or replay buffer. GitHub renders the Mermaid diagram natively, so no additional diagram package is required.

## Get Started

See **[Quick Start](docs/quick_start.md)** for:

- Windows, Linux, and macOS setup
- CPU and NVIDIA CUDA dependency installation
- smoke tests and the complete reproduction pipeline
- individual data, validation, model-training, evaluation, and plotting commands

## Main Outputs

- `artifacts/data/`: generated satellite and airplane features plus provenance metadata
- `artifacts/checkpoints/`: trained Transformer and RL checkpoints
- `artifacts/metrics/`: episode metrics and evaluation summaries
- `artifacts/figures/`: Figure 2/Figure 3-style plots
- [Reproducibility report](docs/reproducibility_report.md): assumptions and result interpretation

## Reproduction Scope

- The goal is trend-level reproduction, not exact point-by-point matching.
- The paper-sized configuration uses 8,441 timesteps at 10-second intervals and dynamically selects 25 high-elevation candidates from a filtered Starlink pool.
- The default action space is `0 = keep` and `1..25 = switch/select satellite`; persistent NORAD IDs define handovers.
- The default reward is `QoS - 0.05 * handover_indicator`.
- Paper-sized runs require CelesTrak/Skyfield propagation and fail instead of silently falling back to incomparable synthetic results.
- `configs/synthetic_smoke.yaml` and `reproduce_all --fast` provide a small deterministic end-to-end check.

For detailed methodology and limitations, see the [reproducibility report](docs/reproducibility_report.md).
