import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="NumPy is required")


def test_fast_data_build(tmp_path):
    from ntn_repro.config import load_config
    from ntn_repro.data import build_dataset

    config = tmp_path / "config.yaml"
    config.write_text(
        """
project:
  name: test
  seed: 1
  artifact_dir: artifacts
data:
  dataset_dir: artifacts/data
  dataset_file: ntn_dataset.npz
  metadata_file: metadata.json
  celestrak_url: ""
  prefer_celestrak: false
  num_satellites: 4
  num_airplanes: 2
  num_steps: 20
  step_seconds: 10
  theta_min_deg: 20.0
  theta_max_deg: 90.0
  demand_min: 0.2
  demand_max: 0.5
  history_length: 3
  default_horizon: 1
  synthetic_satellite_altitude_km: 1200.0
  airplane_altitude_km: 10.0
  route_jitter_deg: 0.1
  congestion_base_min: 0.05
  congestion_base_max: 0.45
transformer: {}
rl: {}
""",
        encoding="utf-8",
    )
    out = build_dataset(config, fast=False, prefer_celestrak=False)
    cfg = load_config(config)
    assert out.exists()
    assert out.name == cfg["data"]["dataset_file"]

