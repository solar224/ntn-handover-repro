import importlib.util
import urllib.error

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="NumPy is required")


def test_fetch_tle_sends_user_agent_and_preserves_errors(monkeypatch):
    from ntn_repro.data import TLEFetchError, fetch_tle

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"STARLINK-TEST\n1 TEST\n2 TEST\n"

    def succeed(request, timeout):
        headers = dict(request.header_items())
        assert headers["User-agent"].startswith("ntn-handover-repro/")
        assert timeout == 30
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", succeed)
    assert fetch_tle("https://example.test/tle", retries=1).startswith("STARLINK-TEST")

    calls = 0

    def forbidden(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        raise urllib.error.HTTPError("https://example.test/tle", 403, "Forbidden", None, None)

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(TLEFetchError, match="403"):
        fetch_tle("https://example.test/tle", retries=3)
    assert calls == 1


def test_tle_altitude_filter_rejects_lowering_orbits():
    from ntn_repro.data import filter_tle_records_by_altitude, tle_catalog_id

    lowering = (
        "STARLINK-LOW",
        "1 44714U 19074B   26208.66670139  .01288908  00000+0  15632-1 0  9990",
        "2 44714  53.1481 234.1726 0006710 358.5783 340.0023 15.57417403  5949",
    )
    stable_shell = (
        "STARLINK-STABLE",
        "1 99999U 24001A   26208.50000000  .00000000  00000+0  00000+0 0  9990",
        "2 99999  53.0000 120.0000 0001000   0.0000   0.0000 15.05000000    10",
    )
    selected = filter_tle_records_by_altitude(
        [lowering, stable_shell],
        minimum_km=500.0,
        maximum_km=600.0,
    )
    assert [tle_catalog_id(record) for record in selected] == [99999]


def test_vectorized_elevation_matches_scalar_geometry():
    import numpy as np

    from ntn_repro.data import compute_elevation_matrix
    from ntn_repro.geometry import elevation_angle_deg

    planes = np.array([[[0.0, 0.0, 10.0]]], dtype=np.float32)
    satellites = np.array([[[0.0, 0.0, 550.0], [0.0, 30.0, 550.0]]], dtype=np.float32)
    actual = compute_elevation_matrix(planes, satellites)
    expected = [
        elevation_angle_deg(0.0, 0.0, 10.0, *satellite)
        for satellite in satellites[0]
    ]
    assert np.allclose(actual[0, 0], expected, atol=1e-4)


def test_fast_data_build(tmp_path):
    from ntn_repro.config import load_config
    from ntn_repro.data import build_dataset, validate_dataset_provenance

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
    cfg["data"]["allow_synthetic_fallback"] = False
    with pytest.raises(RuntimeError, match="source='synthetic'"):
        validate_dataset_provenance(cfg)

