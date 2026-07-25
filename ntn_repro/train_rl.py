from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from .config import checkpoint_dir, load_config, metrics_dir
from .data import load_dataset
from .deps import require_numpy, require_torch
from .env import HandoverEnv
from .models import ActorCriticNet, DQNNet, TransformerTrajectoryModel
from .utils import set_seed, write_metrics_csv


def masked_argmax(np, values, mask):
    masked = values.copy()
    masked[~mask] = -1e30
    if not mask.any():
        return 0
    return int(masked.argmax())


def load_transformer_predictions(cfg: dict[str, Any], horizon: int, fast: bool = False):
    np = require_numpy()
    torch = require_torch()
    ckpt = checkpoint_dir(cfg) / f"transformer_h{horizon}.pt"
    if not ckpt.exists():
        print(f"Transformer checkpoint {ckpt} not found; using oracle future positions for prediction features.")
        return None
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
    model.load_state_dict(state["model_state"])
    model.eval()
    pred = plane_pos.copy()
    batch = []
    index = []
    stride = 2 if fast else 1
    with torch.no_grad():
        for t in range(history, plane_pos.shape[0], stride):
            for k in range(plane_pos.shape[1]):
                batch.append(norm[t - history : t, k])
                index.append((t, k))
                if len(batch) >= 1024:
                    _flush_predictions(torch, np, model, batch, index, pred)
                    batch, index = [], []
        if batch:
            _flush_predictions(torch, np, model, batch, index, pred)
    return pred.astype(np.float32)


def _flush_predictions(torch, np, model, batch, index, pred):
    xb = torch.tensor(np.stack(batch), dtype=torch.float32)
    y = model(xb)[:, -1, :].cpu().numpy()
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


def train_actor_critic(agent: str, env: HandoverEnv, cfg: dict[str, Any], episodes: int):
    np = require_numpy()
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCriticNet(env.state_dim, env.action_dim, cfg["rl"]["hidden_sizes"]).module.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["rl"]["learning_rate"]))
    gamma = float(cfg["rl"]["gamma"])
    entropy_coef = float(cfg["rl"]["entropy_coef"]) if agent == "a2c" else 0.0
    value_coef = float(cfg["rl"]["value_coef"])
    rows = []
    started = time.perf_counter()
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        entropies = []
        values = []
        rewards = []
        handovers = 0
        invalid = 0
        satisfaction = []
        done = False
        while not done:
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
            action, log_prob, entropy, value = net.act(st, mask)
            result = env.step(int(action.item()))
            log_probs.append(log_prob.squeeze(0))
            entropies.append(entropy.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(float(result.reward))
            handovers += int(result.info["handover"])
            invalid += int(result.info["invalid"])
            satisfaction.append(float(result.info["satisfaction"]))
            state = result.state
            done = result.done

        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + gamma * g
            returns.append(g)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        advantages = returns_t - values_t
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = -entropies_t.mean()
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()

        total_reward = sum(rewards)
        rows.append(_episode_row(agent, episode, total_reward, handovers, satisfaction, invalid))
        rows[-1]["loss"] = float(loss.detach().cpu())
        print(f"{agent} episode={episode} reward={total_reward:.3f} handovers={handovers} sat={rows[-1]['avg_satisfaction']:.3f}")

    out = checkpoint_dir(cfg) / f"{agent}.pt"
    torch.save({"model_state": net.state_dict(), "agent": agent, "elapsed_s": time.perf_counter() - started}, out)
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
            replay.append((state, action, result.reward, result.state, result.done, mask, next_mask))
            state = result.state
            total_reward += result.reward
            handovers += int(result.info["handover"])
            invalid += int(result.info["invalid"])
            satisfaction.append(float(result.info["satisfaction"]))
            done = result.done
            if len(replay) >= batch_size:
                losses.append(_dqn_update(torch, np, policy, target, optimizer, replay, batch_size, gamma, device))
        if (episode + 1) % int(cfg["rl"]["dqn_target_update"]) == 0:
            target.load_state_dict(policy.state_dict())
        rows.append(_episode_row("dqn", episode, total_reward, handovers, satisfaction, invalid))
        rows[-1]["epsilon"] = eps
        rows[-1]["loss"] = float(np.mean(losses)) if losses else 0.0
        print(f"dqn episode={episode} reward={total_reward:.3f} handovers={handovers} eps={eps:.3f}")
    out = checkpoint_dir(cfg) / "dqn.pt"
    torch.save({"model_state": policy.state_dict(), "agent": "dqn", "elapsed_s": time.perf_counter() - started}, out)
    return rows, out


def _dqn_update(torch, np, policy, target, optimizer, replay, batch_size, gamma, device):
    batch = random.sample(replay, batch_size)
    states, actions, rewards, next_states, dones, _masks, next_masks = zip(*batch)
    states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    next_masks_t = torch.tensor(np.stack(next_masks), dtype=torch.bool, device=device)
    q = policy(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_q = target(next_t).masked_fill(~next_masks_t, -1e9).max(dim=1).values
        next_q = torch.where(next_q < -1e8, torch.zeros_like(next_q), next_q)
        target_q = rewards_t + gamma * (1.0 - dones_t) * next_q
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
    data = load_dataset(cfg)
    episodes = int(cfg["rl"]["fast_episodes"] if fast else cfg["rl"]["train_episodes"])
    horizon = int(horizon or cfg["data"]["default_horizon"])
    predictions = None
    use_predictions = agent in {"a2c", "actor_critic"}
    if use_predictions:
        predictions = load_transformer_predictions(cfg, horizon, fast=fast)
    env = HandoverEnv(data, cfg, predicted_positions=predictions, use_predictions=use_predictions, seed=int(cfg["project"]["seed"]))
    if agent == "random":
        rows = run_random(env, episodes)
        ckpt = metrics_dir(cfg) / "random.done"
        ckpt.write_text("random policy has no checkpoint\n", encoding="utf-8")
    elif agent in {"a2c", "actor_critic"}:
        rows, ckpt = train_actor_critic(agent, env, cfg, episodes)
    elif agent == "dqn":
        rows, ckpt = train_dqn(env, cfg, episodes)
    else:
        raise ValueError(f"Unsupported agent: {agent}")
    suffix = f"_seed{cfg['project']['seed']}" if seed is not None else ""
    out_metrics = metrics_dir(cfg) / f"{agent}{suffix}.csv"
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

