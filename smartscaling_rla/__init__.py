"""
smartscaling_rla
================

A small, beginner-friendly project that demonstrates how
tabular Q-learning can be used to train an auto-scaling
policy in a simulated cloud environment.
"""

from .envs.autoscaling_env import AutoScalingEnv  # noqa: F401
from .agents.q_learning_agent import QLearningAgent  # noqa: F401

