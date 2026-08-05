from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .config import checkpoint_dir, load_config, metrics_dir
from .data import load_dataset, validate_dataset_provenance
from .deps import require_numpy, require_torch
from .env import HandoverEnv
from .models import ActorCriticNet, DQNNet
from .train_rl import load_transformer_predictions, masked_argmax, scenario_signature
from .utils import write_json, write_metrics_csv


VARIANTS = [
    ("a2c_h5", "a2c", 5),
    ("a2c_h25", "a2c", 25),
    ("dqn", "dqn", None),
    ("random", "random", None),
    ("actor_critic_h5", "actor_critic", 5),
]


def _load_policy(cfg: dict[str, Any], variant: str, agent: str, state_dim: int, action_dim: int, device):
    if agent == "random":
        return None
    torch = require_torch()
    ckpt = checkpoint_dir(cfg) / f"{variant}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Retrain {variant} before evaluation.")
    if agent in {"a2c", "actor_critic"}:
        policy = ActorCriticNet(state_dim, action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    else:
        policy = DQNNet(state_dim, action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    payload = torch.load(ckpt, map_location=device)
    if payload.get("scenario_signature") != scenario_signature(cfg):
        raise RuntimeError(
            f"Checkpoint {ckpt} was trained for an older or different "
            "environment reconstruction. Retrain it before evaluation."
        )
    if int(payload.get("state_dim", -1)) != state_dim or int(
        payload.get("action_dim", -1)
    ) != action_dim:
        raise RuntimeError(
            f"Checkpoint {ckpt} state/action dimensions do not match the "
            "current environment. Retrain it before evaluation."
        )
    policy.load_state_dict(payload["model_state"])
    policy.eval()
    return policy


def _evaluate_run(env: HandoverEnv, policy, agent: str, episodes: int, device):
    np = require_numpy()
    torch = require_torch()
    rewards = []
    handovers = []
    satisfaction = []
    invalid_actions = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        episode_handovers = 0
        episode_satisfaction = []
        episode_invalid = 0
        done = False
        while not done:
            mask = env.valid_action_mask()
            if agent == "random":
                choices = np.where(mask)[0]
                action = int(env.rng.choice(choices))
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    values = policy(state_t)[0] if agent in {"a2c", "actor_critic"} else policy(state_t)
                    action = masked_argmax(np, values.squeeze(0).cpu().numpy(), mask)
            result = env.step(action)
            total_reward += float(result.reward)
            episode_handovers += int(result.info["handover"])
            episode_satisfaction.append(float(result.info["satisfaction"]))
            episode_invalid += int(result.info["invalid"])
            state = result.state
            done = result.done
        rewards.append(total_reward)
        handovers.append(float(episode_handovers))
        satisfaction.append(sum(episode_satisfaction) / max(1, len(episode_satisfaction)))
        invalid_actions.append(float(episode_invalid))
    return {
        "reward_mean": mean(rewards),
        "handovers_mean": mean(handovers),
        "satisfaction_mean": mean(satisfaction),
        "invalid_actions_mean": mean(invalid_actions),
    }


def evaluate(config: str | Path, runs: int = 5, episodes: int | None = None) -> Path:
    cfg = load_config(config)
    validate_dataset_provenance(cfg)
    np = require_numpy()
    torch = require_torch()
    data = load_dataset(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_episodes = int(episodes or cfg["rl"]["eval_episodes"])
    configured_seeds = list(cfg.get("evaluation", {}).get("seeds", []))
    seeds = [int(configured_seeds[i]) if i < len(configured_seeds) else i for i in range(runs)]
    predictions: dict[int, Any] = {}
    summaries = []
    run_rows = []

    for variant, agent, horizon in VARIANTS:
        if horizon is not None and horizon not in predictions:
            predictions[horizon] = load_transformer_predictions(cfg, horizon)
        env = HandoverEnv(
            data,
            cfg,
            predicted_positions=predictions.get(horizon),
            use_predictions=horizon is not None,
            seed=seeds[0],
        )
        policy = _load_policy(cfg, variant, agent, env.state_dim, env.action_dim, device)
        variant_runs = []
        for run_index, seed in enumerate(seeds):
            env = HandoverEnv(
                data,
                cfg,
                predicted_positions=predictions.get(horizon),
                use_predictions=horizon is not None,
                seed=seed,
            )
            result = _evaluate_run(env, policy, agent, eval_episodes, device)
            row = {"variant": variant, "run": run_index, "seed": seed, **result}
            run_rows.append(row)
            variant_runs.append(result)
            print(
                f"{variant} run={run_index} reward={result['reward_mean']:.3f} "
                f"handovers={result['handovers_mean']:.3f} "
                f"satisfaction={result['satisfaction_mean']:.3f}",
                flush=True,
            )

        summary: dict[str, Any] = {
            "variant": variant,
            "agent": agent,
            "horizon": horizon,
            "runs_completed": len(variant_runs),
            "episodes_per_run": eval_episodes,
        }
        for metric in ["reward_mean", "handovers_mean", "satisfaction_mean", "invalid_actions_mean"]:
            values = [float(row[metric]) for row in variant_runs]
            summary[metric] = mean(values)
            summary[metric.replace("_mean", "_std")] = pstdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)

    mdir = metrics_dir(cfg)
    write_metrics_csv(mdir / "evaluation_runs.csv", run_rows)
    out = mdir / "evaluation_summary.json"
    write_json(
        out,
        {
            "runs_requested": runs,
            "episodes_per_run": eval_episodes,
            "seeds": seeds,
            "summaries": summaries,
        },
    )
    print(f"Wrote evaluation summary to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained handover policies on independent episodes.")
    parser.add_argument("--config", default="configs/paper_result.yaml")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent evaluation seeds.")
    parser.add_argument("--episodes", type=int, help="Episodes per run (defaults to rl.eval_episodes).")
    args = parser.parse_args(argv)
    evaluate(args.config, runs=args.runs, episodes=args.episodes)


if __name__ == "__main__":
    main()
