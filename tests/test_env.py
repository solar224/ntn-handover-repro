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


def test_no_coverage_is_an_outage_not_an_invalid_action():
    import numpy as np

    from ntn_repro.env import HandoverEnv

    dataset = {
        "sat_pos": np.zeros((3, 1, 3), dtype=np.float32),
        "plane_pos": np.zeros((3, 1, 3), dtype=np.float32),
        "elevation": np.full((3, 1, 1), -90.0, dtype=np.float32),
        "coverage": np.zeros((3, 1, 1), dtype=np.bool_),
        "congestion": np.zeros((3, 1), dtype=np.float32),
        "demand": np.full((3, 1), 0.3, dtype=np.float32),
    }
    cfg = {
        "data": {"theta_max_deg": 90.0},
        "rl": {"episode_steps": 2, "alpha": 1.0, "beta": 0.05, "invalid_action_penalty": -0.25},
    }
    env = HandoverEnv(dataset, cfg, use_predictions=False, seed=1)
    assert env.valid_action_mask().tolist() == [True, False]
    result = env.step(0)
    assert result.info["invalid"] is False
    assert result.info["satellite"] == -1
    assert result.reward == 0.0


def test_dynamic_candidate_slot_reordering_is_not_a_handover():
    import numpy as np

    from ntn_repro.env import HandoverEnv

    dataset = {
        "sat_pos": np.array(
            [
                [[[0.0, 0.0, 550.0], [0.0, 10.0, 550.0]]],
                [[[0.0, 10.0, 550.0], [0.0, 0.0, 550.0]]],
                [[[0.0, 10.0, 550.0], [0.0, 0.0, 550.0]]],
            ],
            dtype=np.float32,
        ),
        "sat_ids": np.array(
            [
                [[101, 102]],
                [[102, 101]],
                [[102, 101]],
            ],
            dtype=np.int32,
        ),
        "plane_pos": np.array([[[0.0, 0.0, 10.0]]] * 3, dtype=np.float32),
        "elevation": np.array(
            [
                [[80.0, 40.0]],
                [[40.0, 80.0]],
                [[40.0, 80.0]],
            ],
            dtype=np.float32,
        ),
        "coverage": np.ones((3, 1, 2), dtype=np.bool_),
        "congestion": np.full((3, 1, 2), 0.1, dtype=np.float32),
        "demand": np.full((3, 1), 0.3, dtype=np.float32),
    }
    cfg = {
        "data": {"theta_max_deg": 90.0},
        "rl": {"episode_steps": 2, "alpha": 1.0, "beta": 0.05, "invalid_action_penalty": -0.25},
    }
    env = HandoverEnv(dataset, cfg, use_predictions=False, seed=1)
    env.reset(plane_id=0, start_index=0)
    first = env.step(0)
    assert first.info["satellite"] == 101
    assert first.info["handover"] == 0
    assert env._current_slot(env.t) == 1

    second = env.step(0)
    assert second.info["satellite"] == 101
    assert second.info["handover"] == 0


def test_multi_airplane_allocation_enforces_shared_capacity():
    import numpy as np

    from ntn_repro.env import HandoverEnv
    from ntn_repro.train_rl import transition_discount

    dataset = {
        "sat_pos": np.array([[[0.0, 0.0, 550.0]]] * 3, dtype=np.float32),
        "plane_pos": np.zeros((3, 2, 3), dtype=np.float32),
        "elevation": np.full((3, 2, 1), 90.0, dtype=np.float32),
        "coverage": np.ones((3, 2, 1), dtype=np.bool_),
        "congestion": np.full((3, 1), 0.2, dtype=np.float32),
        "demand": np.full((3, 2), 0.6, dtype=np.float32),
    }
    cfg = {
        "data": {"theta_max_deg": 90.0},
        "rl": {
            "episode_steps": 1,
            "alpha": 1.0,
            "beta": 0.05,
            "invalid_action_penalty": -0.25,
            "multi_airplane_allocation": True,
            "shuffle_airplanes_each_timestep": False,
            "capacity_epsilon": 1e-8,
        },
    }
    env = HandoverEnv(dataset, cfg, use_predictions=False, seed=1)
    env.reset(plane_id=0, start_index=0)

    first = env.step(0)
    assert first.info["airplane"] == 0
    assert first.info["allocation"] == pytest.approx(0.6)
    assert first.info["timestep_completed"] is False
    assert first.done is False
    # The next airplane observes the first airplane's reservation:
    # plane(3) + prediction(3) + demand(1) + elevation(1) -> congestion.
    assert first.state[8] == pytest.approx(0.8)

    second = env.step(0)
    assert second.info["airplane"] == 1
    assert second.info["allocation"] == pytest.approx(0.2)
    assert second.info["satisfaction"] == pytest.approx(1.0 / 3.0)
    assert second.info["congestion_after"] == pytest.approx(1.0)
    assert second.info["timestep_completed"] is True
    assert second.done is True
    assert first.info["allocation"] + second.info["allocation"] == pytest.approx(0.8)
    assert transition_discount(first, 0.99) == pytest.approx(1.0)
    assert transition_discount(second, 0.99) == pytest.approx(0.99)


def test_prediction_boundaries_are_excluded_from_episodes():
    import numpy as np

    from ntn_repro.env import HandoverEnv

    dataset = {
        "sat_pos": np.array([[[0.0, 0.0, 550.0]]] * 4, dtype=np.float32),
        "plane_pos": np.zeros((4, 1, 3), dtype=np.float32),
        "elevation": np.full((4, 1, 1), 90.0, dtype=np.float32),
        "coverage": np.ones((4, 1, 1), dtype=np.bool_),
        "congestion": np.zeros((4, 1), dtype=np.float32),
        "demand": np.full((4, 1), 0.3, dtype=np.float32),
    }
    predictions = np.full((4, 1, 3), np.nan, dtype=np.float32)
    predictions[1:3] = 1.0
    cfg = {
        "data": {"theta_max_deg": 90.0},
        "rl": {
            "episode_steps": 3,
            "alpha": 1.0,
            "beta": 0.05,
            "invalid_action_penalty": -0.25,
        },
    }
    env = HandoverEnv(
        dataset,
        cfg,
        predicted_positions=predictions,
        use_predictions=True,
        seed=1,
    )
    assert env.start_t == 1
    assert env.end_t == 2
    with pytest.raises(ValueError, match="outside the valid range"):
        env.reset(start_index=0)


def test_common_experiment_window_matches_all_policy_variants():
    import numpy as np

    from ntn_repro.env import HandoverEnv

    dataset = {
        "sat_pos": np.zeros((10, 1, 3), dtype=np.float32),
        "plane_pos": np.zeros((10, 1, 3), dtype=np.float32),
        "elevation": np.full((10, 1, 1), 90.0, dtype=np.float32),
        "coverage": np.ones((10, 1, 1), dtype=np.bool_),
        "congestion": np.zeros((10, 1), dtype=np.float32),
        "demand": np.full((10, 1), 0.3, dtype=np.float32),
    }
    predictions = np.full((10, 1, 3), np.nan, dtype=np.float32)
    predictions[1:9] = 1.0
    cfg = {
        "data": {
            "theta_max_deg": 90.0,
            "history_length": 2,
            "max_prediction_horizon": 3,
            "use_common_experiment_window": True,
        },
        "rl": {
            "episode_steps": 20,
            "alpha": 1.0,
            "beta": 0.05,
            "invalid_action_penalty": -0.25,
        },
    }

    baseline = HandoverEnv(dataset, cfg, use_predictions=False, seed=1)
    predictive = HandoverEnv(
        dataset,
        cfg,
        predicted_positions=predictions,
        use_predictions=True,
        seed=1,
    )

    assert (baseline.start_t, baseline.end_t) == (1, 6)
    assert (predictive.start_t, predictive.end_t) == (1, 6)

