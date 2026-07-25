from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev

from .config import load_config, metrics_dir
from .utils import read_metrics_csv, write_json


AGENTS = ["a2c", "dqn", "random", "actor_critic"]


def summarize_agent(path: Path) -> dict[str, float | int | str]:
    rows = read_metrics_csv(path)
    if not rows:
        return {"agent": path.stem, "episodes": 0}
    tail = rows[-50:] if len(rows) > 50 else rows
    rewards = [float(r["total_reward"]) for r in tail]
    handovers = [float(r["handovers"]) for r in tail]
    satisfaction = [float(r["avg_satisfaction"]) for r in tail]
    return {
        "agent": str(rows[0].get("agent", path.stem)),
        "episodes": len(rows),
        "reward_mean_tail": mean(rewards),
        "reward_std_tail": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "handovers_mean_tail": mean(handovers),
        "handovers_std_tail": pstdev(handovers) if len(handovers) > 1 else 0.0,
        "satisfaction_mean_tail": mean(satisfaction),
        "satisfaction_std_tail": pstdev(satisfaction) if len(satisfaction) > 1 else 0.0,
    }


def evaluate(config: str | Path, runs: int = 5) -> Path:
    cfg = load_config(config)
    mdir = metrics_dir(cfg)
    summaries: list[dict[str, float | int | str]] = []
    for agent in AGENTS:
        candidates = sorted(mdir.glob(f"{agent}*.csv"))
        if not candidates:
            continue
        seed_summaries = [summarize_agent(path) for path in candidates[:runs]]
        reward_means = [float(s["reward_mean_tail"]) for s in seed_summaries if "reward_mean_tail" in s]
        handover_means = [float(s["handovers_mean_tail"]) for s in seed_summaries if "handovers_mean_tail" in s]
        satisfaction_means = [float(s["satisfaction_mean_tail"]) for s in seed_summaries if "satisfaction_mean_tail" in s]
        summaries.append(
            {
                "agent": agent,
                "runs_found": len(seed_summaries),
                "reward_mean": mean(reward_means) if reward_means else 0.0,
                "reward_std": pstdev(reward_means) if len(reward_means) > 1 else 0.0,
                "handovers_mean": mean(handover_means) if handover_means else 0.0,
                "handovers_std": pstdev(handover_means) if len(handover_means) > 1 else 0.0,
                "satisfaction_mean": mean(satisfaction_means) if satisfaction_means else 0.0,
                "satisfaction_std": pstdev(satisfaction_means) if len(satisfaction_means) > 1 else 0.0,
            }
        )
    out = mdir / "evaluation_summary.json"
    write_json(out, {"runs_requested": runs, "summaries": summaries})
    print(f"Wrote evaluation summary to {out}")
    for summary in summaries:
        print(
            f"{summary['agent']}: reward={summary['reward_mean']:.3f} "
            f"handovers={summary['handovers_mean']:.3f} satisfaction={summary['satisfaction_mean']:.3f}"
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize reproduction metrics.")
    parser.add_argument("--config", default="configs/paper_result.yaml")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)
    evaluate(args.config, runs=args.runs)


if __name__ == "__main__":
    main()

