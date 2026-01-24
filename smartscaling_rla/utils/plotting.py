"""
Plotting utilities using matplotlib.

We keep these functions simple and well-commented so that
beginners can see how to make basic plots of learning curves.
"""

from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: Iterable[float], window: int = 10) -> np.ndarray:
    """
    Compute a simple moving average over the input values.

    This is useful for smoothing the noisy reward curve.
    """
    values = np.array(list(values), dtype=np.float32)
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    # Pad the beginning so that output has same length as input
    padding = np.full(window - 1, ma[0])
    return np.concatenate([padding, ma])


def plot_episode_rewards(episode_rewards: List[float], window: int = 10) -> None:
    """
    Plot the total reward obtained in each episode, together with
    a smoothed moving average curve.
    """
    episodes = np.arange(len(episode_rewards))
    rewards = np.array(episode_rewards, dtype=np.float32)
    smoothed = moving_average(rewards, window=window)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, label="Episode reward", alpha=0.4)
    plt.plot(episodes, smoothed, label=f"{window}-episode moving average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning training in AutoScalingEnv")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

