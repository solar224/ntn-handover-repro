from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_dataset
from .evaluate import evaluate
from .plot import plot_paper_figures
from .train_rl import train_rl
from .train_transformer import train_transformer
from .validate_scenario import validate_scenario


def reproduce(config: str | Path, fast: bool) -> None:
    build_dataset(config, fast=fast, prefer_celestrak=not fast)
    validate_scenario(config, episode_steps=16 if fast else 64)
    train_transformer(config, horizon=5, fast=fast)
    train_transformer(config, horizon=25, fast=fast)
    train_rl(config, agent="a2c", fast=fast, horizon=5)
    train_rl(config, agent="a2c", fast=fast, horizon=25)
    train_rl(config, agent="dqn", fast=fast)
    train_rl(config, agent="actor_critic", fast=fast, horizon=5)
    train_rl(config, agent="random", fast=fast)
    evaluate(config, runs=1 if fast else 5)
    plot_paper_figures(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full reproduction pipeline.")
    parser.add_argument("--config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fast", action="store_true", help="Run a smoke-sized pipeline.")
    group.add_argument("--full", action="store_true", help="Run the paper-sized pipeline.")
    args = parser.parse_args(argv)
    config = args.config or ("configs/synthetic_smoke.yaml" if args.fast else "configs/paper_result.yaml")
    reproduce(config, fast=args.fast)


if __name__ == "__main__":
    main()

