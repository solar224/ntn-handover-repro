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
        use_predictions: bool = False,
        seed: int = 0,
        episode_steps: int | None = None,
    ) -> None:
        self.np = require_numpy()
        self.dataset = dataset
        self.cfg = cfg
        self.rng = self.np.random.default_rng(seed)
        self.sat_pos = dataset["sat_pos"]
        self.sat_ids = dataset["sat_ids"] if "sat_ids" in dataset else None
        self.dynamic_candidates = self.sat_ids is not None
        self.plane_pos = dataset["plane_pos"]
        self.elevation = dataset["elevation"]
        self.coverage = dataset["coverage"]
        self.congestion = dataset["congestion"]
        self.demand = dataset["demand"]
        self.predicted_positions = predicted_positions
        self.use_predictions = use_predictions
        self.num_steps, self.num_airplanes = self.plane_pos.shape[:2]
        self.num_satellites = self.sat_pos.shape[2] if self.dynamic_candidates else self.sat_pos.shape[1]
        self.multi_airplane_allocation = bool(cfg["rl"].get("multi_airplane_allocation", False))
        self.shuffle_airplanes = bool(cfg["rl"].get("shuffle_airplanes_each_timestep", True))
        self.capacity_epsilon = float(cfg["rl"].get("capacity_epsilon", 1e-8))
        self.episode_steps = int(episode_steps or cfg["rl"]["episode_steps"])
        self.theta_max = float(cfg["data"]["theta_max_deg"])
        self.alpha = float(cfg["rl"]["alpha"])
        self.beta = float(cfg["rl"]["beta"])
        self.invalid_penalty = float(cfg["rl"]["invalid_action_penalty"])
        if bool(cfg["data"].get("use_common_experiment_window", False)):
            self.prediction_start = int(cfg["data"]["history_length"]) - 1
            self.prediction_stop = (
                self.num_steps
                - int(cfg["data"].get("max_prediction_horizon", 0))
                - 1
            )
        else:
            self.prediction_start = 0
            self.prediction_stop = self.num_steps - 1
        if self.use_predictions:
            if self.predicted_positions is None:
                raise ValueError("Prediction features require Transformer predictions.")
            valid = self.np.isfinite(self.predicted_positions).all(axis=(1, 2))
            valid_indices = self.np.flatnonzero(valid)
            if valid_indices.size < 2:
                raise ValueError("Transformer predictions do not contain a usable time interval.")
            self.prediction_start = max(
                self.prediction_start, int(valid_indices[0])
            )
            self.prediction_stop = min(
                self.prediction_stop, int(valid_indices[-1])
            )
            if not self.np.isfinite(
                self.predicted_positions[
                    self.prediction_start : self.prediction_stop + 1
                ]
            ).all():
                raise ValueError("Transformer prediction validity must be one contiguous interval.")
        self.reset()

    @property
    def action_dim(self) -> int:
        return self.num_satellites + 1

    @property
    def state_dim(self) -> int:
        return int(self._make_state().shape[0])

    def reset(self, plane_id: int | None = None, start_index: int | None = None):
        first_t = self.prediction_start
        last_t = self.prediction_stop
        available_transitions = last_t - first_t
        if available_transitions < 1:
            raise ValueError("Dataset does not contain enough valid states for an episode.")
        episode_transitions = min(self.episode_steps, available_transitions)
        max_start = last_t - episode_transitions
        if start_index is None:
            self.t = int(self.rng.integers(first_t, max_start + 1))
        else:
            self.t = int(start_index)
            if self.t < first_t or self.t > max_start:
                raise ValueError(
                    f"start_index {self.t} is outside the valid range "
                    f"[{first_t}, {max_start}]"
                )
        self.start_t = self.t
        self.end_t = self.start_t + episode_transitions
        self.allocated_by_sat: dict[int, float] = {}

        if self.multi_airplane_allocation:
            if plane_id is not None:
                base_order = self.np.arange(self.num_airplanes, dtype=self.np.int64)
                start = int(plane_id) % self.num_airplanes
                self.airplane_order = self.np.concatenate((base_order[start:], base_order[:start]))
            elif self.shuffle_airplanes:
                self.airplane_order = self.rng.permutation(self.num_airplanes)
            else:
                self.airplane_order = self.np.arange(self.num_airplanes, dtype=self.np.int64)
            self.plane_cursor = 0
            self.plane_id = int(self.airplane_order[0])
            self.current_satellites = self.np.full(self.num_airplanes, -1, dtype=self.np.int64)
            self.prev_qos_by_plane = self.np.zeros(self.num_airplanes, dtype=self.np.float32)
            for k in range(self.num_airplanes):
                best_slot = self._best_visible_sat(self.t, plane_id=k)
                if best_slot >= 0:
                    self.current_satellites[k] = self._satellite_id(
                        best_slot, self.t, plane_id=k
                    )
        else:
            self.plane_id = int(
                plane_id if plane_id is not None else self.rng.integers(0, self.num_airplanes)
            )
            best_slot = self._best_visible_sat(self.t)
            self.current_sat = self._satellite_id(best_slot, self.t) if best_slot >= 0 else -1
            self.prev_qos = 0.0
        self.last_action_mask = self.valid_action_mask()
        return self._make_state()

    def valid_action_mask(self):
        mask = self.np.zeros(self.action_dim, dtype=self.np.bool_)
        # Action 0 means "do not hand over" and remains a valid no-op during
        # an outage. This prevents lack of coverage from being mislabeled as
        # an invalid agent action.
        mask[0] = True
        for n in range(self.num_satellites):
            mask[n + 1] = self._is_sat_valid(n, self.t)
        return mask

    def step(self, action: int) -> StepResult:
        action = int(action)
        decision_t = self.t
        decision_plane = self.plane_id
        mask = self.valid_action_mask()
        penalty = 0.0
        handover = 0
        invalid = False
        if action < 0 or action >= self.action_dim or not bool(mask[action]):
            invalid = True
            penalty += self.invalid_penalty
            selected = self._current_slot(self.t)
            if selected < 0 or not self._is_sat_valid(selected, self.t):
                selected = self._best_visible_sat(self.t)
        elif action == 0:
            selected = self._current_slot(self.t)
            if selected < 0 or not self._is_sat_valid(selected, self.t):
                selected = -1
        else:
            selected = action - 1
            handover = int(
                self._satellite_id(selected, self.t) != self._get_current_sat()
            )

        if selected < 0:
            qos = 0.0
            satisfaction = 0.0
            allocation = 0.0
            elevation = -90.0
        else:
            qos, satisfaction, allocation, elevation = self._qos_for_sat(selected, self.t)

        reward = self.alpha * qos - self.beta * handover + penalty
        selected_satellite_id = self._satellite_id(selected, self.t) if selected >= 0 else -1
        base_congestion = (
            self._base_congestion_at(selected, self.t) if selected >= 0 else 1.0
        )
        congestion_before = (
            self._congestion_at(selected, self.t) if selected >= 0 else 1.0
        )
        self._set_current_sat(selected_satellite_id)
        self._set_prev_qos(qos)
        if self.multi_airplane_allocation and selected_satellite_id >= 0:
            self.allocated_by_sat[selected_satellite_id] = (
                self.allocated_by_sat.get(selected_satellite_id, 0.0) + allocation
            )
        congestion_after = min(1.0, congestion_before + allocation)

        timestep_completed = self._advance_decision()
        done = self.t >= self.end_t
        self.last_action_mask = self.valid_action_mask() if not done else self.np.zeros(self.action_dim, dtype=self.np.bool_)
        info = {
            "qos": float(qos),
            "satisfaction": float(satisfaction),
            "allocation": float(allocation),
            "elevation": float(elevation),
            "handover": int(handover),
            "invalid": bool(invalid),
            "satellite": int(selected_satellite_id),
            "satellite_slot": int(selected),
            "airplane": int(decision_plane),
            "timestep": int(decision_t),
            "timestep_completed": bool(timestep_completed),
            "base_congestion": float(base_congestion),
            "congestion_before": float(congestion_before),
            "congestion_after": float(congestion_after),
        }
        return StepResult(self._make_state(), float(reward), bool(done), info)

    def _advance_decision(self) -> bool:
        if not self.multi_airplane_allocation:
            self.t += 1
            return True

        self.plane_cursor += 1
        if self.plane_cursor < self.num_airplanes:
            self.plane_id = int(self.airplane_order[self.plane_cursor])
            return False

        self.t += 1
        self.allocated_by_sat = {}
        if self.shuffle_airplanes:
            self.airplane_order = self.rng.permutation(self.num_airplanes)
        self.plane_cursor = 0
        self.plane_id = int(self.airplane_order[0])
        return True

    def _get_current_sat(self, plane_id: int | None = None) -> int:
        if self.multi_airplane_allocation:
            k = self.plane_id if plane_id is None else int(plane_id)
            return int(self.current_satellites[k])
        return int(self.current_sat)

    def _set_current_sat(self, satellite_id: int, plane_id: int | None = None) -> None:
        if self.multi_airplane_allocation:
            k = self.plane_id if plane_id is None else int(plane_id)
            self.current_satellites[k] = int(satellite_id)
        else:
            self.current_sat = int(satellite_id)

    def _get_prev_qos(self, plane_id: int | None = None) -> float:
        if self.multi_airplane_allocation:
            k = self.plane_id if plane_id is None else int(plane_id)
            return float(self.prev_qos_by_plane[k])
        return float(self.prev_qos)

    def _set_prev_qos(self, qos: float, plane_id: int | None = None) -> None:
        if self.multi_airplane_allocation:
            k = self.plane_id if plane_id is None else int(plane_id)
            self.prev_qos_by_plane[k] = float(qos)
        else:
            self.prev_qos = float(qos)

    def _satellite_id(self, slot: int, t: int, plane_id: int | None = None) -> int:
        if slot < 0:
            return -1
        k = self.plane_id if plane_id is None else int(plane_id)
        if self.dynamic_candidates:
            return int(self.sat_ids[t, k, slot])
        return int(slot)

    def _current_slot(self, t: int, plane_id: int | None = None) -> int:
        k = self.plane_id if plane_id is None else int(plane_id)
        current_sat = self._get_current_sat(k)
        if current_sat < 0:
            return -1
        if not self.dynamic_candidates:
            return int(current_sat)
        matches = self.np.where(self.sat_ids[t, k] == current_sat)[0]
        return int(matches[0]) if matches.size else -1

    def _base_congestion_at(
        self, slot: int, t: int, plane_id: int | None = None
    ) -> float:
        k = self.plane_id if plane_id is None else int(plane_id)
        if self.dynamic_candidates:
            return float(self.congestion[t, k, slot])
        return float(self.congestion[t, slot])

    def _congestion_at(
        self, slot: int, t: int, plane_id: int | None = None
    ) -> float:
        base = self._base_congestion_at(slot, t, plane_id=plane_id)
        if not self.multi_airplane_allocation:
            return base
        satellite_id = self._satellite_id(slot, t, plane_id=plane_id)
        return min(1.0, base + self.allocated_by_sat.get(satellite_id, 0.0))

    def _is_sat_valid(
        self, slot: int, t: int, plane_id: int | None = None
    ) -> bool:
        k = self.plane_id if plane_id is None else int(plane_id)
        congestion_limit = (
            1.0 - self.capacity_epsilon
            if self.multi_airplane_allocation
            else 0.98
        )
        return (
            self._satellite_id(slot, t, plane_id=k) >= 0
            and bool(self.coverage[t, k, slot])
            and self._congestion_at(slot, t, plane_id=k) < congestion_limit
        )

    def _best_visible_sat(self, t: int, plane_id: int | None = None) -> int:
        best_sat = -1
        best_qos = -1.0
        for n in range(self.num_satellites):
            if not self._is_sat_valid(n, t, plane_id=plane_id):
                continue
            qos, _, _, _ = self._qos_for_sat(n, t, plane_id=plane_id)
            if qos > best_qos:
                best_qos = qos
                best_sat = n
        return best_sat

    def _qos_for_sat(
        self, slot: int, t: int, plane_id: int | None = None
    ) -> tuple[float, float, float, float]:
        k = self.plane_id if plane_id is None else int(plane_id)
        demand = max(float(self.demand[t, k]), 1e-6)
        congestion = self._congestion_at(slot, t, plane_id=k)
        allocation = min(demand, max(0.0, 1.0 - congestion))
        elevation = float(self.elevation[t, k, slot])
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
            predicted = self.predicted_positions[
                min(self.t, self.predicted_positions.shape[0] - 1), self.plane_id
            ].astype(self.np.float32)
            if not self.np.isfinite(predicted).all():
                raise RuntimeError(
                    f"No valid Transformer prediction for airplane "
                    f"{self.plane_id} at timestep {self.t}."
                )
            return predicted
        raise RuntimeError("Prediction features requested without Transformer predictions.")

    def _make_state(self):
        t = min(self.t, self.num_steps - 1)
        plane = self._normalize_position(self.plane_pos[t, self.plane_id])
        pred = self._normalize_position(self._predicted_position())
        demand = self.np.array([self.demand[t, self.plane_id]], dtype=self.np.float32)
        elevations = self.np.clip(self.elevation[t, self.plane_id] / self.theta_max, -1.0, 1.0).astype(self.np.float32)
        congestion = self.np.array(
            [self._congestion_at(slot, t) for slot in range(self.num_satellites)],
            dtype=self.np.float32,
        )
        current = self.np.zeros(self.num_satellites, dtype=self.np.float32)
        current_slot = self._current_slot(t)
        if current_slot >= 0:
            current[current_slot] = 1.0
        sat = (
            self.sat_pos[t, self.plane_id]
            if self.dynamic_candidates
            else self.sat_pos[t]
        ).astype(self.np.float32).copy()
        sat[:, 0] /= 90.0
        sat[:, 1] /= 180.0
        sat[:, 2] /= 2000.0
        prev = self.np.array([self._get_prev_qos()], dtype=self.np.float32)
        return self.np.concatenate([plane, pred, demand, elevations, congestion, current, sat.reshape(-1), prev]).astype(self.np.float32)

    def _normalize_position(self, pos):
        out = pos.astype(self.np.float32).copy()
        out[0] /= 90.0
        out[1] /= 180.0
        out[2] /= 20.0
        return out

