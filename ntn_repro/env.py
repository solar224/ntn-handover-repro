from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .deps import require_numpy


@dataclass
class StepResult:
    state: Any
    reward: float
    done: bool
    info: dict[str, float | int | bool]


class HandoverEnv:
    def __init__(
        self,
        dataset: Any,
        cfg: dict[str, Any],
        predicted_positions: Any | None = None,
        use_predictions: bool = True,
        seed: int = 0,
        episode_steps: int | None = None,
    ) -> None:
        self.np = require_numpy()
        self.dataset = dataset
        self.cfg = cfg
        self.rng = self.np.random.default_rng(seed)
        self.sat_pos = dataset["sat_pos"]
        self.plane_pos = dataset["plane_pos"]
        self.elevation = dataset["elevation"]
        self.coverage = dataset["coverage"]
        self.congestion = dataset["congestion"]
        self.demand = dataset["demand"]
        self.predicted_positions = predicted_positions
        self.use_predictions = use_predictions
        self.num_steps, self.num_airplanes = self.plane_pos.shape[:2]
        self.num_satellites = self.sat_pos.shape[1]
        self.episode_steps = int(episode_steps or cfg["rl"]["episode_steps"])
        self.theta_max = float(cfg["data"]["theta_max_deg"])
        self.alpha = float(cfg["rl"]["alpha"])
        self.beta = float(cfg["rl"]["beta"])
        self.invalid_penalty = float(cfg["rl"]["invalid_action_penalty"])
        self.reset()

    @property
    def action_dim(self) -> int:
        return self.num_satellites + 1

    @property
    def state_dim(self) -> int:
        return int(self._make_state().shape[0])

    def reset(self, plane_id: int | None = None, start_index: int | None = None):
        self.plane_id = int(plane_id if plane_id is not None else self.rng.integers(0, self.num_airplanes))
        max_start = max(1, self.num_steps - self.episode_steps - 2)
        self.t = int(start_index if start_index is not None else self.rng.integers(0, max_start))
        self.start_t = self.t
        self.end_t = min(self.num_steps - 1, self.start_t + self.episode_steps)
        self.current_sat = self._best_visible_sat(self.t)
        self.prev_qos = 0.0
        self.last_action_mask = self.valid_action_mask()
        return self._make_state()

    def valid_action_mask(self):
        mask = self.np.zeros(self.action_dim, dtype=self.np.bool_)
        if self.current_sat >= 0 and self._is_sat_valid(self.current_sat, self.t):
            mask[0] = True
        for n in range(self.num_satellites):
            mask[n + 1] = self._is_sat_valid(n, self.t)
        return mask

    def step(self, action: int) -> StepResult:
        action = int(action)
        mask = self.valid_action_mask()
        penalty = 0.0
        handover = 0
        invalid = False
        if action < 0 or action >= self.action_dim or not bool(mask[action]):
            invalid = True
            penalty += self.invalid_penalty
            selected = self.current_sat if self.current_sat >= 0 and self._is_sat_valid(self.current_sat, self.t) else self._best_visible_sat(self.t)
        elif action == 0:
            selected = self.current_sat
        else:
            selected = action - 1
            handover = int(selected != self.current_sat)

        if selected < 0:
            qos = 0.0
            satisfaction = 0.0
            allocation = 0.0
            elevation = -90.0
        else:
            qos, satisfaction, allocation, elevation = self._qos_for_sat(selected, self.t)

        reward = self.alpha * qos - self.beta * handover + penalty
        self.current_sat = int(selected)
        self.prev_qos = float(qos)
        self.t += 1
        done = self.t >= self.end_t
        self.last_action_mask = self.valid_action_mask() if not done else self.np.zeros(self.action_dim, dtype=self.np.bool_)
        info = {
            "qos": float(qos),
            "satisfaction": float(satisfaction),
            "allocation": float(allocation),
            "elevation": float(elevation),
            "handover": int(handover),
            "invalid": bool(invalid),
            "satellite": int(selected),
        }
        return StepResult(self._make_state(), float(reward), bool(done), info)

    def _is_sat_valid(self, sat_id: int, t: int) -> bool:
        return bool(self.coverage[t, self.plane_id, sat_id]) and float(self.congestion[t, sat_id]) < 0.98

    def _best_visible_sat(self, t: int) -> int:
        best_sat = -1
        best_qos = -1.0
        for n in range(self.num_satellites):
            if not self._is_sat_valid(n, t):
                continue
            qos, _, _, _ = self._qos_for_sat(n, t)
            if qos > best_qos:
                best_qos = qos
                best_sat = n
        return best_sat

    def _qos_for_sat(self, sat_id: int, t: int) -> tuple[float, float, float, float]:
        demand = max(float(self.demand[t, self.plane_id]), 1e-6)
        congestion = float(self.congestion[t, sat_id])
        allocation = min(demand, max(0.0, 1.0 - congestion))
        elevation = float(self.elevation[t, self.plane_id, sat_id])
        elev_term = max(0.0, elevation / self.theta_max) ** 1.5
        alloc_term = (allocation + 0.1) / (demand + 0.1)
        qos = elev_term * alloc_term * (1.0 - congestion)
        qos = min(1.0, max(0.0, qos))
        satisfaction = min(1.0, max(0.0, allocation / demand))
        return qos, satisfaction, allocation, elevation

    def _predicted_position(self):
        if not self.use_predictions:
            return self.np.zeros(3, dtype=self.np.float32)
        if self.predicted_positions is not None:
            return self.predicted_positions[min(self.t, self.predicted_positions.shape[0] - 1), self.plane_id].astype(self.np.float32)
        horizon = int(self.cfg["data"].get("default_horizon", 5))
        future_t = min(self.num_steps - 1, self.t + horizon)
        return self.plane_pos[future_t, self.plane_id].astype(self.np.float32)

    def _make_state(self):
        t = min(self.t, self.num_steps - 1)
        plane = self._normalize_position(self.plane_pos[t, self.plane_id])
        pred = self._normalize_position(self._predicted_position())
        demand = self.np.array([self.demand[t, self.plane_id]], dtype=self.np.float32)
        elevations = self.np.clip(self.elevation[t, self.plane_id] / self.theta_max, -1.0, 1.0).astype(self.np.float32)
        congestion = self.congestion[t].astype(self.np.float32)
        current = self.np.zeros(self.num_satellites, dtype=self.np.float32)
        if self.current_sat >= 0:
            current[self.current_sat] = 1.0
        sat = self.sat_pos[t].astype(self.np.float32).copy()
        sat[:, 0] /= 90.0
        sat[:, 1] /= 180.0
        sat[:, 2] /= 2000.0
        prev = self.np.array([self.prev_qos], dtype=self.np.float32)
        return self.np.concatenate([plane, pred, demand, elevations, congestion, current, sat.reshape(-1), prev]).astype(self.np.float32)

    def _normalize_position(self, pos):
        out = pos.astype(self.np.float32).copy()
        out[0] /= 90.0
        out[1] /= 180.0
        out[2] /= 20.0
        return out

