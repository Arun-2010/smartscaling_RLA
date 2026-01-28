"""
Training script: connect AutoScalingEnv with QLearningAgent and collect metrics.

This file is intentionally verbose and well-commented for beginners.

Run from project root:

    python -m smartscaling_rla.training.train_qlearning
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from smartscaling_rla.agents import QLearningAgent
from smartscaling_rla.config import env_config, q_learning_config, training_config
from smartscaling_rla.envs import AutoScalingEnv


def compute_cpu_utilization(load: float, instances: int, base_capacity_per_instance: float) -> float:
    """
    In this toy simulator, "CPU utilization" is approximated as:

        utilization = load / (instances * base_capacity_per_instance)

    - load: incoming work rate
    - instances * base_capacity_per_instance: total capacity

    We clamp instances/capacity to avoid division by zero.
    """
    capacity = max(1.0, float(instances) * float(base_capacity_per_instance))
    return float(load) / capacity


def make_run_dir(root: str = "runs") -> str:
    """Create a timestamped run directory to save metrics."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train_q_learning(
    env: AutoScalingEnv,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps: int,
    debug_every_episodes: int = 50,
    debug_every_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Major steps:
      1) Loop over episodes
      2) Reset env at episode start
      3) Loop over steps:
         - choose action (ε-greedy)
         - env.step(action)
         - update Q-table
         - log metrics (cpu util, server count, reward)
      4) Return metrics arrays for saving/plotting
    """

    # Per-step logs (flattened across episodes).
    step_episode: List[int] = []
    step_t: List[int] = []
    step_reward: List[float] = []
    step_instances: List[int] = []
    step_cpu_util: List[float] = []
    step_action: List[int] = []

    # Per-episode summaries (useful for plotting learning curves).
    episode_return: List[float] = []
    episode_avg_cpu: List[float] = []
    episode_avg_instances: List[float] = []

    # Training loop
    for ep in tqdm(range(num_episodes), desc="Training episodes"):
        # Tell the agent a new episode is starting (updates epsilon schedule)
        agent.begin_episode()

        state, _ = env.reset(seed=training_config.seed)

        total_reward = 0.0
        cpu_sum = 0.0
        inst_sum = 0.0
        steps = 0

        for t in range(max_steps):
            # --- Choose action from current policy (ε-greedy) ---
            action = agent.choose_action(state)

            # --- Step environment ---
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # --- Q-learning update ---
            agent.update(state, action, reward, next_state, done)

            # --- Extract metrics from env/info ---
            instances = int(info.get("instances", int(next_state[0])))
            load = float(info.get("load", 0.0))
            cpu_util = compute_cpu_utilization(load, instances, env.config.base_load)

            # --- Log per-step metrics ---
            step_episode.append(ep)
            step_t.append(t)
            step_reward.append(float(reward))
            step_instances.append(instances)
            step_cpu_util.append(cpu_util)
            step_action.append(int(action))

            # --- Accumulate for per-episode metrics ---
            total_reward += float(reward)
            cpu_sum += cpu_util
            inst_sum += instances
            steps += 1

            # --- Occasional debug prints ---
            if (ep % debug_every_episodes == 0) and (t % debug_every_steps == 0):
                print(
                    f"[debug] ep={ep:4d} t={t:3d} "
                    f"action={action} instances={instances} load={load:5.1f} "
                    f"cpu_util={cpu_util:0.2f} reward={reward:0.2f}"
                )

            # Move to next state
            state = next_state

            if done:
                break

        # Episode summaries
        episode_return.append(total_reward)
        episode_avg_cpu.append(cpu_sum / max(1, steps))
        episode_avg_instances.append(inst_sum / max(1, steps))

    # Convert to numpy arrays so saving/plotting is easy and consistent.
    metrics = {
        "step_episode": np.array(step_episode, dtype=np.int32),
        "step_t": np.array(step_t, dtype=np.int32),
        "step_reward": np.array(step_reward, dtype=np.float32),
        "step_instances": np.array(step_instances, dtype=np.int32),
        "step_cpu_util": np.array(step_cpu_util, dtype=np.float32),
        "step_action": np.array(step_action, dtype=np.int32),
        "episode_return": np.array(episode_return, dtype=np.float32),
        "episode_avg_cpu": np.array(episode_avg_cpu, dtype=np.float32),
        "episode_avg_instances": np.array(episode_avg_instances, dtype=np.float32),
    }
    return metrics


def save_metrics(run_dir: str, metrics: Dict[str, np.ndarray]) -> None:
    """
    Save metrics in a single compressed NumPy file.
    This is lightweight, fast, and easy to load later for visualization.
    """
    out_path = os.path.join(run_dir, "metrics.npz")
    np.savez_compressed(out_path, **metrics)


def save_run_config(run_dir: str) -> None:
    """Save the configs used for the run so results are reproducible."""
    cfg = {
        "env_config": asdict(env_config),
        "q_learning_config": asdict(q_learning_config),
        "training_config": asdict(training_config),
    }
    out_path = os.path.join(run_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main() -> str:
    """
    Create env/agent, train for >= 500 episodes, and save metrics.
    Returns the run directory path.
    """
    # --- Create environment ---
    env = AutoScalingEnv(config=env_config)

    # --- Create agent ---
    agent = QLearningAgent(
        num_instances=env.config.max_instances,
        max_load_bucket=env.max_load_bucket,
        num_actions=env.action_space.n,
        config=q_learning_config,
    )

    # Ensure the requirement "run at least 500 episodes" is met.
    num_episodes = max(500, int(training_config.num_episodes))
    max_steps = int(training_config.max_steps_per_episode)

    # --- Train and collect metrics ---
    metrics = train_q_learning(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    # --- Save metrics for later visualization ---
    run_dir = make_run_dir()
    save_run_config(run_dir)
    save_metrics(run_dir, metrics)

    print(f"Saved run artifacts to: {run_dir}")
    print("Saved metrics file: metrics.npz")
    print("Saved config file:  run_config.json")
    return run_dir


if __name__ == "__main__":
    main()

