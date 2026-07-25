import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="NumPy is required")


def test_qos_and_invalid_action_behavior():
    import numpy as np

    from ntn_repro.env import HandoverEnv

    dataset = {
        "sat_pos": np.array([[[0.0, 0.0, 1200.0], [0.0, 30.0, 1200.0]]] * 4, dtype=np.float32),
        "plane_pos": np.array([[[0.0, 0.0, 10.0]]] * 4, dtype=np.float32),
        "elevation": np.array([[[80.0, 10.0]]] * 4, dtype=np.float32),
        "coverage": np.array([[[True, False]]] * 4),
        "congestion": np.array([[0.1, 0.2]] * 4, dtype=np.float32),
        "demand": np.array([[0.3]] * 4, dtype=np.float32),
    }
    cfg = {
        "data": {"theta_max_deg": 90.0, "default_horizon": 1},
        "rl": {"episode_steps": 2, "alpha": 1.0, "beta": 0.05, "invalid_action_penalty": -0.25},
    }
    env = HandoverEnv(dataset, cfg, seed=1)
    state = env.reset(plane_id=0, start_index=0)
    assert state.shape[0] == env.state_dim
    mask = env.valid_action_mask()
    assert mask.tolist() == [True, True, False]
    result = env.step(2)
    assert result.info["invalid"] is True
    assert result.reward < 1.0

