# Quick Start

This guide contains the setup, testing, data generation, model training, evaluation, and plotting commands for the NTN handover reproduction package. Run all commands from the repository root.

## Requirements

- CPython 3.10 or newer
- Internet access for dependency installation
- Internet access to CelesTrak for a paper-sized run
- Optional: an NVIDIA GPU for CUDA-enabled PyTorch

On Windows, use the standard CPython distribution from python.org or `winget`. Do not use MSYS2/Cygwin Python because scientific packages such as NumPy and PyTorch require standard Windows wheels.

## Windows Setup

If CPython is not installed, install Python 3.12 and open a new PowerShell window:

```powershell
winget install --id Python.Python.3.12 -e
```

Create a clean virtual environment and install the CPU dependencies:

```powershell
cd C:\path\to\ntn-handover-repro
.\scripts\setup_windows.ps1 -Recreate
.\.venv\Scripts\Activate.ps1
```

For an NVIDIA GPU, install the CUDA PyTorch dependencies instead:

```powershell
.\scripts\setup_windows.ps1 -Recreate -Cuda
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the setup script, allow it for the current shell only and rerun the setup command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The virtual-environment interpreter should be `.venv\Scripts\python.exe`. If it is `.venv\bin\python.exe`, recreate the environment with standard Windows CPython:

```powershell
.\scripts\setup_windows.ps1 -Recreate
```

## Linux/macOS Setup

```bash
cd /path/to/ntn-handover-repro
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the Installation

Run the automated tests after activating the virtual environment:

```bash
python -m pytest
```

Optionally verify that the core dependencies import successfully:

```bash
python -c "import numpy, pandas, torch, matplotlib, yaml, skyfield, sgp4; print('Dependencies OK')"
```

## End-to-End Runs

Use the fast pipeline for a small deterministic smoke test. It writes results under `artifacts/smoke/`:

```bash
python -m ntn_repro.reproduce_all --fast
```

Use the full pipeline for the paper-sized experiment. It downloads or refreshes CelesTrak data as needed and writes results under `artifacts/`:

```bash
python -m ntn_repro.reproduce_all --full
```

Successful CelesTrak downloads are cached for at least two hours. The full configuration intentionally fails if the required orbital data cannot be obtained; use the fast pipeline when a deterministic offline-style smoke experiment is sufficient.

## Run Each Stage Manually

The following commands reproduce the full workflow one stage at a time with `configs/paper_result.yaml`.

### 1. Build and validate the dataset

```bash
python -m ntn_repro.data build --config configs/paper_result.yaml
python -m ntn_repro.validate_scenario --config configs/paper_result.yaml
```

To explicitly build a deterministic synthetic dataset instead of downloading CelesTrak data:

```bash
python -m ntn_repro.data build --config configs/synthetic_smoke.yaml --synthetic-only
```

### 2. Train the Transformer predictors

```bash
python -m ntn_repro.train_transformer --config configs/paper_result.yaml --horizon 5
python -m ntn_repro.train_transformer --config configs/paper_result.yaml --horizon 25
```

### 3. Train the handover policies

```bash
python -m ntn_repro.train_rl --config configs/paper_result.yaml --agent a2c --horizon 5
python -m ntn_repro.train_rl --config configs/paper_result.yaml --agent a2c --horizon 25
python -m ntn_repro.train_rl --config configs/paper_result.yaml --agent dqn
python -m ntn_repro.train_rl --config configs/paper_result.yaml --agent actor_critic --horizon 5
python -m ntn_repro.train_rl --config configs/paper_result.yaml --agent random
```

The A2C commands train the Transformer-aided policies. DQN and random do not require a prediction horizon; the actor-critic command is the horizon-5 ablation used by the reproduction workflow.

### 4. Evaluate trained policies

```bash
python -m ntn_repro.evaluate --config configs/paper_result.yaml --runs 5
```

### 5. Generate figures

```bash
python -m ntn_repro.plot --config configs/paper_result.yaml --paper-figures
```

## Fast Individual Commands

For development, individual training commands accept `--fast` and use reduced training lengths. Keep the config consistent across data generation, training, and evaluation:

```bash
python -m ntn_repro.data build --config configs/synthetic_smoke.yaml --fast
python -m ntn_repro.validate_scenario --config configs/synthetic_smoke.yaml
python -m ntn_repro.train_transformer --config configs/synthetic_smoke.yaml --horizon 5 --fast
python -m ntn_repro.train_transformer --config configs/synthetic_smoke.yaml --horizon 25 --fast
python -m ntn_repro.train_rl --config configs/synthetic_smoke.yaml --agent a2c --horizon 5 --fast
```

For a complete smoke workflow, prefer `python -m ntn_repro.reproduce_all --fast` because it runs every required stage with a consistent configuration.

## Output Locations

Paper-sized runs create:

- `artifacts/data/ntn_dataset.npz`: generated satellite and airplane features
- `artifacts/data/metadata.json`: TLE or constellation provenance
- `artifacts/checkpoints/`: Transformer and RL checkpoints
- `artifacts/metrics/`: per-episode metrics and evaluation summaries
- `artifacts/figures/`: paper-style plots

Fast runs write the corresponding files under `artifacts/smoke/`.

Return to the [README](../README.md) or read the [reproducibility report](reproducibility_report.md) for methodology, assumptions, and interpretation guidance.
