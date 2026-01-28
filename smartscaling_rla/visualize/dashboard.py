"""
Visualization dashboard for smartscaling_RLA runs using matplotlib only.

This script loads a metrics.npz file (saved by train_qlearning.py)
and plots:
  - CPU utilization over time
  - Number of servers over time
  - Reward per step over time
  - Actions taken by the agent

Usage (from project root):

    python -m smartscaling_rla.visualize.dashboard --run-dir runs/20260128_151547
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: str) -> Dict[str, np.ndarray]:
    """
    Load metrics from a given run directory.

    Expects:
        <run_dir>/metrics.npz

    with keys (as saved by train_qlearning.py):
      - step_episode
      - step_t
      - step_reward
      - step_instances
      - step_cpu_util
      - step_action
      - episode_return
      - episode_avg_cpu
      - episode_avg_instances
    """
    path = os.path.join(run_dir, "metrics.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find metrics.npz in {run_dir!r}")

    data = np.load(path)
    return {k: data[k] for k in data.files}


def moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute a simple moving average for smoothing noisy curves."""
    if len(values) == 0 or len(values) < window:
        return values

    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, ma[0])
    return np.concatenate([pad, ma])


def plot_dashboard(metrics: Dict[str, np.ndarray], title: str | None = None) -> None:
    """
    Create a 4-row matplotlib dashboard:

    1) CPU utilization over time
    2) Number of servers over time
    3) Reward per step (with optional smoothing)
    4) Actions taken by the agent (0=down, 1=stay, 2=up)
    """
    # Extract arrays
    cpu = metrics["step_cpu_util"]
    instances = metrics["step_instances"]
    rewards = metrics["step_reward"]
    actions = metrics["step_action"]
    step_episode = metrics["step_episode"]

    # Global time index
    t_global = np.arange(len(cpu))
    rewards_smooth = moving_average(rewards, window=100)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    if title is None:
        title = "smartscaling_RLA Training Dashboard"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1) CPU utilization
    ax = axes[0]
    ax.plot(t_global, cpu, color="tab:blue", linewidth=1.0, label="CPU utilization")
    ax.set_ylabel("CPU Utilization")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    ax.set_title("CPU utilization over time")

    # 2) Number of servers
    ax = axes[1]
    ax.step(
        t_global,
        instances,
        where="post",
        color="tab:orange",
        linewidth=1.0,
        label="Number of servers",
    )
    ax.set_ylabel("Servers")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    ax.set_title("Number of servers over time")

    # 3) Reward per step
    ax = axes[2]
    ax.plot(t_global, rewards, color="tab:green", alpha=0.4, linewidth=0.8, label="Reward (raw)")
    ax.plot(
        t_global,
        rewards_smooth,
        color="tab:green",
        linewidth=1.5,
        label="Reward (smoothed, window=100)",
    )
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    ax.set_title("Reward per step")

    # 4) Actions taken
    ax = axes[3]
    action_y = actions
    colors = np.array(["tab:red", "tab:gray", "tab:blue"])
    action_colors = colors[actions.clip(0, 2)]

    ax.scatter(
        t_global,
        action_y,
        c=action_colors,
        s=8,
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_ylabel("Action")
    ax.set_xlabel("Global step index")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["0 = scale down", "1 = stay", "2 = scale up"])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("Actions taken by the agent")

    # Light vertical lines between episodes for context
    episode_changes = np.where(np.diff(step_episode) != 0)[0]
    for idx in episode_changes:
        for a in axes:
            a.axvline(idx + 0.5, color="k", alpha=0.05, linewidth=0.5)

    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matplotlib dashboard for smartscaling_RLA runs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a run directory containing metrics.npz "
        "(e.g., runs/20260128_151547).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title for the dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = load_metrics(args.run_dir)
    plot_dashboard(metrics, title=args.title)


if __name__ == "__main__":
    main()

