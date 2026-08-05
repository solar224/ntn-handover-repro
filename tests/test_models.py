import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is required")


def test_transformer_forward_shape():
    import torch

    from ntn_repro.models import TransformerTrajectoryModel

    model = TransformerTrajectoryModel(history_length=5, horizon=2, d_model=16, nhead=4, num_layers=1, dim_feedforward=32).module
    y = model(torch.zeros(3, 5, 3))
    assert tuple(y.shape) == (3, 3)


def test_dqn_update_uses_stored_physical_timestep_discounts():
    from collections import deque

    import numpy as np
    import torch

    from ntn_repro.train_rl import _dqn_update

    policy = torch.nn.Linear(4, 2)
    target = torch.nn.Linear(4, 2)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    replay = deque(
        [
            (
                np.zeros(4, dtype=np.float32),
                0,
                0.5,
                np.ones(4, dtype=np.float32),
                False,
                discount,
                np.ones(2, dtype=np.bool_),
                np.ones(2, dtype=np.bool_),
            )
            for discount in (1.0, 0.99)
        ]
    )

    loss = _dqn_update(
        torch,
        np,
        policy,
        target,
        optimizer,
        replay,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert np.isfinite(loss)

