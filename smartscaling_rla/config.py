"""
Central configuration for the smartscaling_RLA project.

The goal is to keep all important hyperparameters and
environment settings in one place so that beginners can
easily experiment with them.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the auto-scaling environment."""

    max_instances: int = 10
    min_instances: int = 1
    target_utilization: float = 0.7
    max_queue: int = 50
    episode_length: int = 200  # time steps per episode

    # Traffic parameters (very simple synthetic model)
    base_load: float = 5.0
    load_amplitude: float = 5.0

    # Cost / reward shaping
    cost_per_instance: float = 0.1
    penalty_sla_violation: float = 1.0


@dataclass
class QLearningConfig:
    """Hyperparameters for the tabular Q-learning agent."""

    learning_rate: float = 0.1      # α
    discount_factor: float = 0.95   # γ
    epsilon_start: float = 1.0      # initial exploration rate
    epsilon_end: float = 0.05       # final exploration rate
    epsilon_decay_episodes: int = 300  # how quickly we decay ε


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    num_episodes: int = 500
    max_steps_per_episode: int = 200
    seed: int | None = 42


env_config = EnvConfig()
q_learning_config = QLearningConfig()
training_config = TrainingConfig()

