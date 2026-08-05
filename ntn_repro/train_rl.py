from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from .config import checkpoint_dir, load_config, metrics_dir
from .data import load_dataset, validate_dataset_provenance
from .deps import require_numpy, require_torch
from .env import HandoverEnv
from .models import ActorCriticNet, DQNNet, TransformerTrajectoryModel
from .utils import set_seed, write_metrics_csv


def scenario_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    rl_cfg = cfg["rl"]
    return {
        "version": 3,
        "satellite_selection_mode": str(
            data_cfg.get("satellite_selection_mode", "fixed_first_n")
        ),
        "num_satellites": int(data_cfg["num_satellites"]),
        "num_airplanes": int(data_cfg["num_airplanes"]),
        "theta_min_deg": float(data_cfg["theta_min_deg"]),
        "theta_max_deg": float(data_cfg["theta_max_deg"]),
        "congestion_base_min": float(data_cfg["congestion_base_min"]),
        "congestion_base_max": float(data_cfg["congestion_base_max"]),
        "history_length": int(data_cfg["history_length"]),
        "max_prediction_horizon": int(
            data_cfg.get("max_prediction_horizon", 0)
        ),
        "use_common_experiment_window": bool(
            data_cfg.get("use_common_experiment_window", False)
        ),
        "multi_airplane_allocation": bool(
            rl_cfg.get("multi_airplane_allocation", False)
        ),
        "discount_clock": (
            "physical_timestep"
            if bool(rl_cfg.get("multi_airplane_allocation", False))
            else "decision"
        ),
        "alpha": float(rl_cfg["alpha"]),
        "beta": float(rl_cfg["beta"]),
        "gamma": float(rl_cfg["gamma"]),
    }


def masked_argmax(np, values, mask):
    masked = values.copy()
    masked[~mask] = -1e30
    if not mask.any():
        return 0
    return int(masked.argmax())


def transition_discount(result, gamma: float) -> float:
    """Apply gamma once per physical timestep, not once per airplane."""

    return float(gamma) if bool(result.info["timestep_completed"]) else 1.0


def load_transformer_predictions(cfg: dict[str, Any], horizon: int, fast: bool = False):
    np = require_numpy()
    torch = require_torch()
    ckpt = checkpoint_dir(cfg) / f"transformer_h{horizon}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint {ckpt} not found. "
            f"Run: python -m ntn_repro.train_transformer --horizon {horizon}"
        )
    data = load_dataset(cfg)
    history = int(cfg["data"]["history_length"])
    plane_pos = data["plane_pos"].astype(np.float32)
    norm = plane_pos.copy()
    norm[..., 0] /= 90.0
    norm[..., 1] /= 180.0
    norm[..., 2] /= 20.0
    tcfg = cfg["transformer"]
    model = TransformerTrajectoryModel(
        history_length=history,
        horizon=horizon,
        d_model=int(tcfg["d_model"]),
        nhead=int(tcfg["nhead"]),
        num_layers=int(tcfg["num_layers"]),
        dim_feedforward=int(tcfg["dim_feedforward"]),
        dropout=float(tcfg["dropout"]),
    ).module
    state = torch.load(ckpt, map_location="cpu")
    checkpoint_cfg = state.get("config", {})
    checkpoint_transformer_cfg = checkpoint_cfg.get("transformer", {})
    expected_split = str(tcfg.get("split_mode", "chronological"))
    checkpoint_split = str(
        state.get(
            "split_mode",
            checkpoint_transformer_cfg.get("split_mode", "random_or_unspecified"),
        )
    )
    if checkpoint_split != expected_split:
        raise RuntimeError(
            f"Transformer checkpoint {ckpt} used split_mode "
            f"{checkpoint_split!r}, but the current reconstruction requires "
            f"{expected_split!r}. Retrain the Transformer."
        )
    if int(state.get("horizon", -1)) != int(horizon):
        raise RuntimeError(
            f"Transformer checkpoint {ckpt} has the wrong prediction horizon. "
            "Retrain the Transformer."
        )
    try:
        model.load_state_dict(state["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Transformer checkpoint {ckpt} is incompatible with the paper-aligned "
            "single-future-position model. Retrain the Transformer."
        ) from exc
    model.eval()
    # Invalid boundary rows stay NaN so the environment cannot silently use
    # current ground-truth positions as future predictions.
    pred = np.full_like(plane_pos, np.nan, dtype=np.float32)
    batch = []
    index = []
    with torch.no_grad():
        for t in range(history - 1, plane_pos.shape[0] - horizon):
            for k in range(plane_pos.shape[1]):
                batch.append(norm[t - history + 1 : t + 1, k])
                index.append((t, k))
                if len(batch) >= 1024:
                    _flush_predictions(torch, np, model, batch, index, pred)
                    batch, index = [], []
        if batch:
            _flush_predictions(torch, np, model, batch, index, pred)
    return pred.astype(np.float32)


def _flush_predictions(torch, np, model, batch, index, pred):
    xb = torch.tensor(np.stack(batch), dtype=torch.float32)
    y = model(xb).cpu().numpy()
    y[:, 0] *= 90.0
    y[:, 1] *= 180.0
    y[:, 2] *= 20.0
    for row, (t, k) in enumerate(index):
        pred[t, k, :] = y[row]


def run_random(env: HandoverEnv, episodes: int) -> list[dict[str, float | int | str]]:
    rows = []
    np = require_numpy()
    for episode in range(episodes):
        env.reset()
        total_reward = 0.0
        handovers = 0
        satisfaction = []
        invalid = 0
        done = False
        while not done:
            mask = env.valid_action_mask()
            choices = np.where(mask)[0]
            action = int(env.rng.choice(choices)) if len(choices) else 0
            result = env.step(action)
            total_reward += result.reward
            handovers += int(result.info["handover"])
            invalid += int(result.info["invalid"])
            satisfaction.append(float(result.info["satisfaction"]))
            done = result.done
        rows.append(_episode_row("random", episode, total_reward, handovers, satisfaction, invalid))
        print(f"random episode={episode} reward={total_reward:.3f} handovers={handovers}")
    return rows


def train_a2c(env: HandoverEnv, cfg: dict[str, Any], episodes: int, variant: str):
    np = require_numpy()
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCriticNet(env.state_dim, env.action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["rl"]["learning_rate"]))
    gamma = float(cfg["rl"]["gamma"])
    entropy_coef = float(cfg["rl"]["entropy_coef"])
    value_coef = float(cfg["rl"]["value_coef"])
    rollout_steps = int(cfg["rl"].get("a2c_rollout_steps", 32))
    rows = []
    started = time.perf_counter()
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        handovers = 0
        invalid = 0
        satisfaction = []
        done = False
        losses = []
        while not done:
            log_probs = []
            entropies = []
            values = []
            rewards = []
            discounts = []
            for _ in range(rollout_steps):
                st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                action, log_prob, entropy, value = net.act(st, mask)
                result = env.step(int(action.item()))
                log_probs.append(log_prob.squeeze(0))
                entropies.append(entropy.squeeze(0))
                values.append(value.squeeze(0))
                rewards.append(float(result.reward))
                discounts.append(transition_discount(result, gamma))
                total_reward += float(result.reward)
                handovers += int(result.info["handover"])
                invalid += int(result.info["invalid"])
                satisfaction.append(float(result.info["satisfaction"]))
                state = result.state
                done = result.done
                if done:
                    break

            with torch.no_grad():
                if done:
                    bootstrap = 0.0
                else:
                    next_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    bootstrap = float(net(next_state)[1].item())
            returns = []
            g = bootstrap
            for reward, discount in zip(reversed(rewards), reversed(discounts)):
                g = reward + discount * g
                returns.append(g)
            returns.reverse()
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            values_t = torch.stack(values)
            advantages = returns_t - values_t
            policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()
            entropy_loss = -torch.stack(entropies).mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        rows.append(_episode_row(variant, episode, total_reward, handovers, satisfaction, invalid))
        rows[-1]["loss"] = float(np.mean(losses)) if losses else 0.0
        print(f"{variant} episode={episode} reward={total_reward:.3f} handovers={handovers} sat={rows[-1]['avg_satisfaction']:.3f}")

    out = checkpoint_dir(cfg) / f"{variant}.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "agent": "a2c",
            "variant": variant,
            "horizon": int(variant.rsplit("h", 1)[1]),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "scenario_signature": scenario_signature(cfg),
            "elapsed_s": time.perf_counter() - started,
        },
        out,
    )
    return rows, out


def train_vanilla_actor_critic(env: HandoverEnv, cfg: dict[str, Any], episodes: int, variant: str):
    np = require_numpy()
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCriticNet(env.state_dim, env.action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    actor_optimizer = torch.optim.Adam(net.actor.parameters(), lr=float(cfg["rl"]["learning_rate"]))
    critic_optimizer = torch.optim.Adam(net.critic.parameters(), lr=float(cfg["rl"]["learning_rate"]))
    gamma = float(cfg["rl"]["gamma"])
    rows = []
    started = time.perf_counter()
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        handovers = 0
        invalid = 0
        satisfaction = []
        losses = []
        done = False
        while not done:
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
            logits, value = net(st)
            logits = logits.masked_fill(~mask, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            result = env.step(int(action.item()))

            with torch.no_grad():
                if result.done:
                    target = torch.tensor([result.reward], dtype=torch.float32, device=device)
                else:
                    next_st = torch.tensor(result.state, dtype=torch.float32, device=device).unsqueeze(0)
                    next_value = net(next_st)[1]
                    target = (
                        torch.tensor([result.reward], dtype=torch.float32, device=device)
                        + transition_discount(result, gamma) * next_value
                    )

            advantage = target - value
            actor_loss = -(dist.log_prob(action) * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.actor.parameters(), 5.0)
            actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.critic.parameters(), 5.0)
            critic_optimizer.step()

            losses.append(float((actor_loss.detach() + critic_loss.detach()).cpu()))
            total_reward += float(result.reward)
            handovers += int(result.info["handover"])
            invalid += int(result.info["invalid"])
            satisfaction.append(float(result.info["satisfaction"]))
            state = result.state
            done = result.done

        rows.append(_episode_row(variant, episode, total_reward, handovers, satisfaction, invalid))
        rows[-1]["loss"] = float(np.mean(losses)) if losses else 0.0
        print(f"{variant} episode={episode} reward={total_reward:.3f} handovers={handovers} sat={rows[-1]['avg_satisfaction']:.3f}")

    out = checkpoint_dir(cfg) / f"{variant}.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "agent": "actor_critic",
            "variant": variant,
            "horizon": int(variant.rsplit("h", 1)[1]),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "scenario_signature": scenario_signature(cfg),
            "elapsed_s": time.perf_counter() - started,
        },
        out,
    )
    return rows, out


def train_dqn(env: HandoverEnv, cfg: dict[str, Any], episodes: int):
    np = require_numpy()
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DQNNet(env.state_dim, env.action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    target = DQNNet(env.state_dim, env.action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg["rl"]["learning_rate"]))
    replay = deque(maxlen=int(cfg["rl"]["replay_capacity"]))
    gamma = float(cfg["rl"]["gamma"])
    batch_size = int(cfg["rl"]["batch_size"])
    eps_start = float(cfg["rl"]["epsilon_start"])
    eps_end = float(cfg["rl"]["epsilon_end"])
    eps_decay = max(1, int(cfg["rl"]["epsilon_decay_episodes"]))
    rows = []
    started = time.perf_counter()
    for episode in range(episodes):
        eps = eps_end + (eps_start - eps_end) * math.exp(-episode / eps_decay)
        state = env.reset()
        total_reward = 0.0
        handovers = 0
        satisfaction = []
        invalid = 0
        done = False
        losses = []
        while not done:
            mask = env.valid_action_mask()
            if random.random() < eps:
                choices = np.where(mask)[0]
                action = int(env.rng.choice(choices)) if len(choices) else 0
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                action = masked_argmax(np, q, mask)
            result = env.step(action)
            next_mask = env.valid_action_mask() if not result.done else np.zeros(env.action_dim, dtype=np.bool_)
            replay.append(
                (
                    state,
                    action,
                    result.reward,
                    result.state,
                    result.done,
                    transition_discount(result, gamma),
                    mask,
                    next_mask,
                )
            )
            state = result.state
            total_reward += result.reward
            handovers += int(result.info["handover"])
            invalid += int(result.info["invalid"])
            satisfaction.append(float(result.info["satisfaction"]))
            done = result.done
            if len(replay) >= batch_size:
                losses.append(
                    _dqn_update(
                        torch,
                        np,
                        policy,
                        target,
                        optimizer,
                        replay,
                        batch_size,
                        device,
                    )
                )
        if (episode + 1) % int(cfg["rl"]["dqn_target_update"]) == 0:
            target.load_state_dict(policy.state_dict())
        rows.append(_episode_row("dqn", episode, total_reward, handovers, satisfaction, invalid))
        rows[-1]["epsilon"] = eps
        rows[-1]["loss"] = float(np.mean(losses)) if losses else 0.0
        print(f"dqn episode={episode} reward={total_reward:.3f} handovers={handovers} eps={eps:.3f}")
    out = checkpoint_dir(cfg) / "dqn.pt"
    torch.save(
        {
            "model_state": policy.state_dict(),
            "agent": "dqn",
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "scenario_signature": scenario_signature(cfg),
            "elapsed_s": time.perf_counter() - started,
        },
        out,
    )
    return rows, out


def _dqn_update(torch, np, policy, target, optimizer, replay, batch_size, device):
    batch = random.sample(replay, batch_size)
    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        discounts,
        _masks,
        next_masks,
    ) = zip(*batch)
    states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    discounts_t = torch.tensor(discounts, dtype=torch.float32, device=device)
    next_masks_t = torch.tensor(np.stack(next_masks), dtype=torch.bool, device=device)
    q = policy(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_q = target(next_t).masked_fill(~next_masks_t, -1e9).max(dim=1).values
        next_q = torch.where(next_q < -1e8, torch.zeros_like(next_q), next_q)
        target_q = rewards_t + discounts_t * (1.0 - dones_t) * next_q
    loss = torch.nn.functional.smooth_l1_loss(q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach().cpu())


def _episode_row(agent: str, episode: int, total_reward: float, handovers: int, satisfaction: list[float], invalid: int):
    return {
        "agent": agent,
        "episode": episode,
        "total_reward": float(total_reward),
        "handovers": int(handovers),
        "avg_satisfaction": float(sum(satisfaction) / max(1, len(satisfaction))),
        "invalid_actions": int(invalid),
    }


def train_rl(config: str | Path, agent: str, fast: bool = False, horizon: int | None = None, seed: int | None = None) -> Path:
    cfg = load_config(config)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    set_seed(int(cfg["project"]["seed"]))
    validate_dataset_provenance(cfg)
    data = load_dataset(cfg)
    episodes = int(cfg["rl"]["fast_episodes"] if fast else cfg["rl"]["train_episodes"])
    horizon = int(horizon or cfg["data"]["default_horizon"])
    variant = f"{agent}_h{horizon}" if agent in {"a2c", "actor_critic"} else agent
    predictions = None
    use_predictions = agent in {"a2c", "actor_critic"}
    if use_predictions:
        predictions = load_transformer_predictions(cfg, horizon, fast=fast)
    episode_steps = (
        int(cfg["rl"].get("fast_episode_steps", cfg["rl"]["episode_steps"]))
        if fast
        else None
    )
    env = HandoverEnv(
        data,
        cfg,
        predicted_positions=predictions,
        use_predictions=use_predictions,
        seed=int(cfg["project"]["seed"]),
        episode_steps=episode_steps,
    )
    if agent == "random":
        rows = run_random(env, episodes)
        ckpt = metrics_dir(cfg) / "random.done"
        ckpt.write_text("random policy has no checkpoint\n", encoding="utf-8")
    elif agent == "a2c":
        rows, ckpt = train_a2c(env, cfg, episodes, variant)
    elif agent == "actor_critic":
        rows, ckpt = train_vanilla_actor_critic(env, cfg, episodes, variant)
    elif agent == "dqn":
        rows, ckpt = train_dqn(env, cfg, episodes)
    else:
        raise ValueError(f"Unsupported agent: {agent}")
    suffix = f"_seed{cfg['project']['seed']}" if seed is not None else ""
    out_metrics = metrics_dir(cfg) / f"{variant}{suffix}.csv"
    write_metrics_csv(out_metrics, rows)
    print(f"Wrote metrics to {out_metrics}")
    return Path(ckpt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RL handover agents.")
    parser.add_argument("--config", default="configs/paper_result.yaml")
    parser.add_argument("--agent", choices=["a2c", "dqn", "actor_critic", "random"], required=True)
    parser.add_argument("--horizon", type=int, choices=[5, 25])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    out = train_rl(args.config, args.agent, fast=args.fast, horizon=args.horizon, seed=args.seed)
    print(f"Wrote checkpoint/status to {out}")


if __name__ == "__main__":
    main()

