"""
Simple tabular Q-learning agent.

This agent works with any environment that exposes:
    - action_space.n  (number of discrete actions)
    - a discrete state that we can map to an integer index

In our case, we map the 2D state [instances, load_bucket] to a
single integer index using a small helper method.
"""

from __future__ import annotations


from dataclasses import dataclass, field

from typing import Tuple

import numpy as np

from smartscaling_rla.config import QLearningConfig, q_learning_config


@dataclass
class QLearningAgent:
    """
    A basic tabular Q-learning agent with ε-greedy exploration.

    The Q-table has shape [num_states, num_actions]. For simplicity we let
    the agent manage the mapping from environment states (e.g., 2D vectors)
    to a single integer state index.
    """

    num_instances: int
    max_load_bucket: int
    num_actions: int
    config: QLearningConfig = field(default_factory=lambda: q_learning_config)

    def __post_init__(self):
        # Number of distinct instance values (e.g., 1..10 -> 10 values)
        self.num_instance_states = self.num_instances

        # Total number of possible discrete states
        self.num_states = self.num_instance_states * (self.max_load_bucket + 1)

        # Initialize Q-table to zeros
        self.Q = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        # Internal episode counter for epsilon decay
        self.episode = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def begin_episode(self):
        """
        Call this at the start of each episode so that we can update
        the exploration rate (ε).
        """
        self.episode += 1

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action using an ε-greedy policy.

        With probability ε we pick a random action (exploration);
        otherwise we pick the greedy action from the Q-table.
        """
        epsilon = self._current_epsilon()
        state_idx = self._state_to_index(state)

        if np.random.rand() < epsilon:
            # Explore
            return int(np.random.randint(self.num_actions))

        # Exploit (greedy action)
        q_values = self.Q[state_idx]
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Perform the tabular Q-learning update:

        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
        """
        alpha = self.config.learning_rate
        gamma = self.config.discount_factor

        s = self._state_to_index(state)
        s_next = self._state_to_index(next_state)

        q_sa = self.Q[s, action]
        if done:
            target = reward  # no future value if episode ended
        else:
            target = reward + gamma * np.max(self.Q[s_next])

        self.Q[s, action] = q_sa + alpha * (target - q_sa)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _state_to_index(self, state: np.ndarray) -> int:
        """
        Convert a 2D state [instances, load_bucket] into a single integer.

        We assume:
            - instances ∈ [1, num_instances]
            - load_bucket ∈ [0, max_load_bucket]
        """
        instances, load_bucket = int(state[0]), int(state[1])
        instances_clipped = np.clip(instances, 1, self.num_instances)
        load_clipped = np.clip(load_bucket, 0, self.max_load_bucket)

        # Map (instances, load_bucket) -> single index
        # We shift instances so that they start from 0
        idx = (instances_clipped - 1) * (self.max_load_bucket + 1) + load_clipped
        return int(idx)

    def _current_epsilon(self) -> float:
        """
        Linearly decay ε from epsilon_start to epsilon_end over
        epsilon_decay_episodes episodes, then keep it at epsilon_end.
        """
        c = self.config
        if self.episode >= c.epsilon_decay_episodes:
            return c.epsilon_end

        frac = self.episode / max(1, c.epsilon_decay_episodes)
        return float(c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start))

