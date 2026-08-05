from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .config import load_config, metadata_path
from .data import load_dataset, validate_dataset_provenance
from .deps import require_numpy
from .env import HandoverEnv
from .utils import read_json


def run_policy(env: HandoverEnv, policy: str) -> dict[str, float | int]:
    np = require_numpy()
    env.reset(start_index=env.prediction_start)
    total_reward = 0.0
    handovers = 0
    satisfaction: list[float] = []
    completed_timesteps = 0
    allocations: dict[tuple[int, int], float] = defaultdict(float)
    base_load: dict[tuple[int, int], float] = {}
    done = False

    while not done:
        mask = env.valid_action_mask()
        if policy == "random":
            action = int(env.rng.choice(np.flatnonzero(mask)))
        elif policy == "greedy_qos":
            valid_slots = np.flatnonzero(mask[1:])
            best_slot = (
                max(
                    valid_slots,
                    key=lambda slot: env._qos_for_sat(int(slot), env.t)[0],
                )
                if valid_slots.size
                else -1
            )
            action = int(best_slot + 1) if best_slot >= 0 else 0
        else:
            raise ValueError(f"Unsupported diagnostic policy: {policy}")

        result = env.step(action)
        info = result.info
        total_reward += float(result.reward)
        handovers += int(info["handover"])
        satisfaction.append(float(info["satisfaction"]))
        completed_timesteps += int(info["timestep_completed"])
        if int(info["satellite"]) >= 0:
            key = (int(info["timestep"]), int(info["satellite"]))
            allocations[key] += float(info["allocation"])
            base_load[key] = float(info["base_congestion"])
        done = result.done

    maximum_capacity_excess = max(
        (
            base_load[key] + allocation - 1.0
            for key, allocation in allocations.items()
        ),
        default=0.0,
    )
    return {
        "decisions": len(satisfaction),
        "physical_timesteps": completed_timesteps,
        "reward": total_reward,
        "handovers": handovers,
        "average_satisfaction": float(np.mean(satisfaction)),
        "maximum_capacity_excess": max(0.0, maximum_capacity_excess),
    }


def validate_scenario(
    config: str | Path,
    episode_steps: int = 64,
    seed: int = 7,
) -> None:
    np = require_numpy()
    cfg = load_config(config)
    validate_dataset_provenance(cfg)
    data = load_dataset(cfg)
    metadata = read_json(metadata_path(cfg))

    print(
        "dataset",
        f"source={metadata['source']}",
        f"selection={metadata['satellite_selection_mode']}",
        f"pool={metadata['candidate_pool_size']}",
        f"propagation_start={metadata['propagation_start_utc']}",
    )
    print(
        "geometry",
        f"no_visible={metadata['no_visible_fraction']:.6%}",
        f"visible_mean={metadata['visible_satellites_mean']:.3f}",
        f"congestion=[{float(data['congestion'].min()):.3f},"
        f"{float(data['congestion'].max()):.3f}]",
    )

    for policy in ("random", "greedy_qos"):
        env = HandoverEnv(
            data,
            cfg,
            use_predictions=False,
            seed=seed,
            episode_steps=episode_steps,
        )
        result = run_policy(env, policy)
        print(
            policy,
            f"decisions={result['decisions']}",
            f"timesteps={result['physical_timesteps']}",
            f"reward={result['reward']:.3f}",
            f"handovers={result['handovers']}",
            f"satisfaction={result['average_satisfaction']:.4f}",
            f"capacity_excess={result['maximum_capacity_excess']:.3e}",
        )
        if float(result["maximum_capacity_excess"]) > 1e-6:
            raise RuntimeError(
                f"{policy} violated shared satellite capacity by "
                f"{result['maximum_capacity_excess']:.6g}."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the reconstructed paper scenario and capacity invariant."
    )
    parser.add_argument("--config", default="configs/paper_result.yaml")
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    validate_scenario(args.config, episode_steps=args.episode_steps, seed=args.seed)


if __name__ == "__main__":
    main()
