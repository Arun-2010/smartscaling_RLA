"""
Entry point script to train a Q-learning agent in the AutoScalingEnv.

Usage (from project root):

    python -m smartscaling_rla.simulation.run_simulation
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from smartscaling_rla.agents import QLearningAgent
from smartscaling_rla.config import (
    TrainingConfig,
    env_config,
    q_learning_config,
    training_config,
)
from smartscaling_rla.envs import AutoScalingEnv
from smartscaling_rla.utils.plotting import plot_episode_rewards


def train(
    env: AutoScalingEnv,
    agent: QLearningAgent,
    config: TrainingConfig,
) -> List[float]:
    """
    Run a basic training loop for the given agent in the environment.

    Returns a list of total reward per episode.
    """
    episode_rewards: List[float] = []

    for _ in tqdm(range(config.num_episodes), desc="Training episodes"):
        # Tell the agent a new episode is starting (for epsilon decay)
        agent.begin_episode()

        state, _ = env.reset(seed=config.seed)
        total_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _info = env.step(action)

            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

    return episode_rewards


def main() -> Tuple[np.ndarray, np.ndarray]:
    """Create environment and agent, train, and plot rewards."""
    # Create environment
    env = AutoScalingEnv(config=env_config)

    # Create Q-learning agent. We pass in:
    #   - num_instances: max instances (since instances ∈ [1, max_instances])
    #   - max_load_bucket: must match the environment's discretization
    #   - num_actions: from env.action_space.n
    agent = QLearningAgent(
        num_instances=env.config.max_instances,
        max_load_bucket=env.max_load_bucket,
        num_actions=env.action_space.n,
        config=q_learning_config,
    )

    # Optional: set NumPy seed for reproducibility
    if training_config.seed is not None:
        np.random.seed(training_config.seed)

    # Train
    rewards = train(env, agent, training_config)

    # Basic plot of episode rewards
    plot_episode_rewards(rewards)

    # Return arrays so this function is also useful from notebooks/tests
    return np.arange(len(rewards)), np.array(rewards)


if __name__ == "__main__":
    main()

