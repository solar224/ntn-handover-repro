from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_dataset
from .evaluate import evaluate
from .plot import plot_paper_figures
from .train_rl import train_rl
from .train_transformer import train_transformer


def reproduce(config: str | Path, fast: bool) -> None:
    build_dataset(config, fast=fast, prefer_celestrak=not fast)
    train_transformer(config, horizon=5, fast=fast)
    if not fast:
        train_transformer(config, horizon=25, fast=False)
    for agent in ["a2c", "dqn", "actor_critic", "random"]:
        train_rl(config, agent=agent, fast=fast, horizon=5)
    evaluate(config, runs=1 if fast else 5)
    plot_paper_figures(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full reproduction pipeline.")
    parser.add_argument("--config", default="configs/paper_result.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fast", action="store_true", help="Run a smoke-sized pipeline.")
    group.add_argument("--full", action="store_true", help="Run the paper-sized pipeline.")
    args = parser.parse_args(argv)
    reproduce(args.config, fast=args.fast)


if __name__ == "__main__":
    main()

