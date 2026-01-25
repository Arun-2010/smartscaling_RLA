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
        current_instances = self.instances

        # 1) Apply the scaling action
        if action == 0:  # scale down
            self.instances = max(self.config.min_instances, self.instances - 1)
        elif action == 2:  # scale up
            self.instances = min(self.config.max_instances, self.instances + 1)
        # action == 1 means "do nothing"

        # 2) Advance time and sample new load
        self.current_step += 1
        self.load = self._sample_load(self.current_step)

        # 3) Compute reward based on new state and action taken
        reward = self._compute_reward(action, current_instances)

        # 4) Episode termination condition
        terminated = False
        truncated = self.current_step >= self.config.episode_length

        state = self._get_state()
        info: Dict = {
            "instances": self.instances,
            "load": self.load,
        }
        return state, float(reward), terminated, truncated, info

    def _compute_reward(self, action: int, previous_instances: int) -> float:
        """
        Redesigned reward function to address:
        1. Idle servers penalty
        2. Action differentiation
        3. Long-term cost optimization
        """
        utilization = self._utilization()

        # --- COMPONENT 1: PERFORMANCE (Goldilocks Zone) ---
        # Reward being in the optimal range (e.g., 50-80%)
        # Sigmoid-like or Gaussian bell curve is often better, but step function is simple/clear
        if 0.5 <= utilization <= 0.8:
            perf_reward = 20.0  # Big bonus for being perfect
        else:
            # Penalize distance from target (0.7)
            # Higher weight (-20) to ensure bad states are clearly distinguished
            dist = abs(utilization - 0.7)
            perf_reward = -20.0 * dist 

        # --- COMPONENT 2: COST (Linear + Per-Step) ---
        # Stronger penalty per server to minimize usage
        # This addresses "ignores infrastructure cost"
        cost_penalty = -2.0 * self.instances

        # --- COMPONENT 3: SLA VIOLATION (Critical) ---
        if utilization > 1.0:
            # Huge penalty to prevent crashing
            sla_penalty = -100.0 * (utilization - 1.0)
        else:
            sla_penalty = 0.0

        # --- COMPONENT 4: IDLE RESOURCE PENALTY ---
        # Explicitly address "fails to scale down when idle"
        idle_penalty = 0.0
        if utilization < 0.2 and self.instances > self.config.min_instances:
            # If we are barely using the server but have more than minimum, punish heavily
            idle_penalty = -10.0

        # --- COMPONENT 5: ACTION PENALTIES ---
        action_penalty = 0.0
        
        # A. Penalize "DO NOTHING" when action is needed (Crisis or Waste)
        is_crisis = utilization > 0.9
        is_waste = utilization < 0.3 and self.instances > self.config.min_instances
        if action == 1 and (is_crisis or is_waste):
            action_penalty -= 5.0 # Nudge the agent to ACT
            
        # B. Penalize Invalid/Ineffective Actions
        # Trying to scale down below min
        if action == 0 and previous_instances <= self.config.min_instances:
            action_penalty -= 5.0
        # Trying to scale up above max
        if action == 2 and previous_instances >= self.config.max_instances:
            action_penalty -= 5.0

        # Total Reward = Sum of components
        # We normalize slightly by checking the magnitude, but these values (10-100)
        # provide good separation for Q-values.
        # Total Base Reward
        reward = perf_reward + cost_penalty + sla_penalty + idle_penalty + action_penalty
        
        # --- COMPONENT 6: FORCE SCALE DOWN (Immediate Incentive) ---
        # REQUIRED FIX: Give immediate positive reward for scaling down when idle.
        # We calculate "previous utilization" to see if the state we acted on was wasteful.
        
        # Use current load as proxy (since load is stochastic, this is close enough)
        prev_capacity = max(1.0, previous_instances * self.config.base_load)
        prev_util = self.load / prev_capacity

        scaling_incentive = 0.0
        
        # Condition: We were idle (low util) and had servers to spare
        if prev_util < 0.2 and previous_instances > self.config.min_instances:
            if action == 0:
                # IMMEDIATE REWARD for taking the right action
                scaling_incentive = +15.0 
            elif action == 1:
                # EXPLICIT PENALTY for laziness
                scaling_incentive = -15.0

        # Final Total Reward
        reward += scaling_incentive
        return float(reward)

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

