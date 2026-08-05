from __future__ import annotations

import argparse
from pathlib import Path

from .config import figures_dir, load_config, metrics_dir
from .deps import require_matplotlib
from .utils import read_metrics_csv


STYLE = {
    "a2c_h5": {"label": "Our Framework", "color": "#1b9e77"},
    "a2c_h25": {"label": "Our Framework (Horizon 25)", "color": "#377eb8"},
    "dqn": {"label": "DQN Approach", "color": "#d95f02"},
    "random": {"label": "Random Approach", "color": "#7570b3"},
    "actor_critic_h5": {"label": "Actor-Critic", "color": "#e7298a"},
}


def _load_variant_metrics(mdir: Path, variant: str):
    exact = mdir / f"{variant}.csv"
    if exact.exists():
        return read_metrics_csv(exact)
    paths = sorted(mdir.glob(f"{variant}_seed*.csv"))
    return read_metrics_csv(paths[0]) if paths else []


def plot_paper_figures(config: str | Path) -> list[Path]:
    plt = require_matplotlib()
    cfg = load_config(config)
    mdir = metrics_dir(cfg)
    fdir = figures_dir(cfg)
    outputs: list[Path] = []

    comparison_variants = ["a2c_h5", "dqn", "random"]
    rows_by_agent = {variant: _load_variant_metrics(mdir, variant) for variant in comparison_variants}
    missing = [variant for variant, rows in rows_by_agent.items() if not rows]
    ablation_agents = ["a2c_h5", "a2c_h25", "actor_critic_h5"]
    missing.extend(variant for variant in ablation_agents if not _load_variant_metrics(mdir, variant))
    if missing:
        raise FileNotFoundError(
            "Missing paper-figure metrics for: "
            + ", ".join(sorted(set(missing)))
            + ". Retrain the paper-aligned variants before plotting."
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent, rows in rows_by_agent.items():
        if not rows:
            continue
        ax.plot([r["episode"] for r in rows], [r["handovers"] for r in rows], label=STYLE[agent]["label"], color=STYLE[agent]["color"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Handovers")
    ax.set_title("Handovers Across Different Approaches")
    ax.legend()
    fig.tight_layout()
    out = fdir / "fig2a_handovers.png"
    fig.savefig(out, dpi=180)
    outputs.append(out)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent, rows in rows_by_agent.items():
        if not rows:
            continue
        ax.plot([r["episode"] for r in rows], [r["total_reward"] for r in rows], label=STYLE[agent]["label"], color=STYLE[agent]["color"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Total Reward Across Different Approaches")
    ax.legend()
    fig.tight_layout()
    out = fdir / "fig2b_total_reward.png"
    fig.savefig(out, dpi=180)
    outputs.append(out)
    plt.close(fig)

    labels = []
    values = []
    colors = []
    for agent in comparison_variants:
        rows = rows_by_agent[agent]
        if not rows:
            continue
        tail = rows[-50:] if len(rows) > 50 else rows
        labels.append(STYLE[agent]["label"])
        values.append(sum(float(r["avg_satisfaction"]) for r in tail) / len(tail))
        colors.append(STYLE[agent]["color"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average Demand Satisfaction")
    ax.set_title("Average Demand Satisfaction Across Different Approaches")
    fig.tight_layout()
    out = fdir / "fig2c_demand_satisfaction.png"
    fig.savefig(out, dpi=180)
    outputs.append(out)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent in ablation_agents:
        rows = _load_variant_metrics(mdir, agent)
        if not rows:
            continue
        ax.plot([r["episode"] for r in rows], [r["total_reward"] for r in rows], label=STYLE[agent]["label"], color=STYLE[agent]["color"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Impact of Prediction Horizon and RL Model")
    ax.legend()
    fig.tight_layout()
    out = fdir / "fig3_ablation.png"
    fig.savefig(out, dpi=180)
    outputs.append(out)
    plt.close(fig)

    for path in outputs:
        print(f"Wrote {path}")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style reproduction figures.")
    parser.add_argument("--config", default="configs/paper_result.yaml")
    parser.add_argument("--paper-figures", action="store_true", help="Generate Figure 2/Figure 3 style outputs.")
    args = parser.parse_args(argv)
    if args.paper_figures:
        plot_paper_figures(args.config)
    else:
        parser.error("Use --paper-figures")


if __name__ == "__main__":
    main()

