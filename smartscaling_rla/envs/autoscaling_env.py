"""
Gym-style environment that simulates a simple cloud auto-scaling scenario.

The environment is intentionally simplified and heavily commented so that
beginners can focus on understanding the RL loop and Q-learning updates.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartscaling_rla.config import EnvConfig, env_config


class AutoScalingEnv(gym.Env):
    """
    A tiny auto-scaling environment with a discrete state and action space.

    State (discrete):
        - current number of instances (bounded)
        - discretized load level bucket (0, 1, 2, ...)

    Actions (discrete):
        0: scale down (remove one instance, if possible)
        1: do nothing
        2: scale up (add one instance, if possible)

    The reward encourages:
        - keeping enough capacity so that utilization stays near target
        - avoiding having too many instances (wasting money)
        - avoiding SLA violations when load exceeds capacity
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config: EnvConfig = config or env_config
        self.render_mode = render_mode

        # Discrete actions: scale down, stay, scale up
        self.action_space = spaces.Discrete(3)

        # We keep the state small and discrete:
        # - instances: [min_instances, max_instances]
        # - load_bucket: 0..max_load_bucket
        self.max_load_bucket = 10
        low = np.array([self.config.min_instances, 0], dtype=np.int32)
        high = np.array([self.config.max_instances, self.max_load_bucket], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.current_step: int = 0
        self.instances: int = self.config.min_instances
        self.load: float = self.config.base_load

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to the beginning of an episode."""
        super().reset(seed=seed)

        self.current_step = 0
        self.instances = self.config.min_instances
        self.load = self._sample_load(self.current_step)

        state = self._get_state()
        info: Dict = {}
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply an action, advance the simulation by one time step, and
        return (state, reward, terminated, truncated, info).
        """
        # 1) Apply the scaling action
        if action == 0:  # scale down
            self.instances = max(self.config.min_instances, self.instances - 1)
        elif action == 2:  # scale up
            self.instances = min(self.config.max_instances, self.instances + 1)
        # action == 1 means "do nothing"

        # 2) Advance time and sample new load
        self.current_step += 1
        self.load = self._sample_load(self.current_step)

        # 3) Compute reward based on new state
        reward = self._compute_reward()

        # 4) Episode termination condition
        terminated = False
        truncated = self.current_step >= self.config.episode_length

        state = self._get_state()
        info: Dict = {
            "instances": self.instances,
            "load": self.load,
        }
        return state, float(reward), terminated, truncated, info

    def render(self):
        """
        Simple text render. In a real project, you might generate
        more detailed plots instead (we do that in utils.plotting).
        """
        utilization = self._utilization()
        print(
            f"Step={self.current_step}, instances={self.instances}, "
            f"load={self.load:.1f}, utilization={utilization:.2f}"
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _sample_load(self, t: int) -> float:
        """
        Very simple periodic traffic model plus a bit of noise.

        In a real system, you would plug in historical traffic data or
        a more realistic stochastic process.
        """
        base = self.config.base_load
        amp = self.config.load_amplitude
        # Simple sine wave pattern between base-amp and base+amp
        seasonal = base + amp * math.sin(2 * math.pi * t / self.config.episode_length)
        noise = self.np_random.normal(loc=0.0, scale=1.0)
        return max(0.0, seasonal + noise)

    def _utilization(self) -> float:
        """
        Approximate per-instance utilization.

        We assume each instance can handle `base_load` units comfortably;
        this is purely for demonstration and not meant to be realistic.
        """
        capacity = max(1.0, self.instances * self.config.base_load)
        return float(self.load / capacity)

    def _compute_reward(self) -> float:
        """
        Reward is a combination of:
          - penalty for being far from the target utilization
          - penalty for using many instances (cost)
          - large penalty for SLA violations (utilization >> 1)
        """
        utilization = self._utilization()

        # Penalty for deviation from target utilization (quadratic)
        deviation = utilization - self.config.target_utilization
        penalty_deviation = deviation**2

        # Cost grows linearly with number of instances
        cost = self.instances * self.config.cost_per_instance

        # SLA violation penalty if utilization is too high
        sla_violation = max(0.0, utilization - 1.0)
        penalty_sla = sla_violation * self.config.penalty_sla_violation

        # We NEGATE because we want to maximize reward, but all components
        # we computed are "bad".
        reward = -penalty_deviation - cost - penalty_sla
        return float(reward)

    def _get_state(self) -> np.ndarray:
        """
        Convert the internal (instances, load) variables into
        a discrete observation vector.
        """
        load_bucket = int(
            np.clip(
                round(self.load),
                0,
                self.max_load_bucket,
            )
        )
        return np.array([self.instances, load_bucket], dtype=np.int32)

