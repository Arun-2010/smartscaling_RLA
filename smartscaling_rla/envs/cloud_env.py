import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CloudEnv(gym.Env):
    """
    Fake Cloud Environment for Reinforcement Learning based auto-scaling.

    This environment simulates:
    - Incoming traffic
    - CPU utilization
    - Latency
    - Number of servers

    The agent learns when to scale up/down to balance
    performance (CPU, latency) and cost (servers).
    """

    def __init__(self):
        super(CloudEnv, self).__init__()

        # ----------------------------
        # Environment Configuration
        # ----------------------------
        self.min_servers = 1
        self.max_servers = 10

        self.max_steps = 200
        self.current_step = 0

        # Initial state
        self.servers = 3
        self.traffic = 50  # requests per second
        self.cpu_util = 50.0
        self.latency = 100.0

        # ----------------------------
        # Action Space
        # 0: Scale Down
        # 1: Do Nothing
        # 2: Scale Up
        # ----------------------------
        self.action_space = spaces.Discrete(3)

        # ----------------------------
        # Observation Space (Discrete)
        # CPU level: 0 (low), 1 (medium), 2 (high)
        # Traffic level: 0 (low), 1 (medium), 2 (high)
        # Server count: 1 to 10
        # ----------------------------
        self.observation_space = spaces.MultiDiscrete([
            3,  # CPU level
            3,  # Traffic level
            self.max_servers
        ])

    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------
    def _get_cpu_level(self):
        if self.cpu_util < 40:
            return 0
        elif self.cpu_util < 70:
            return 1
        else:
            return 2

    def _get_traffic_level(self):
        if self.traffic < 40:
            return 0
        elif self.traffic < 70:
            return 1
        else:
            return 2

    def _get_state(self):
        return (
            self._get_cpu_level(),
            self._get_traffic_level(),
            self.servers - 1
        )

    # --------------------------------------------------
    # Core Gym Functions
    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.servers = 3
        self.traffic = np.random.randint(40, 60)

        self._update_metrics()

        return self._get_state(), {}

    def step(self, action):
        self.current_step += 1

        # ----------------------------
        # Apply Scaling Action
        # ----------------------------
        if action == 0:  # Scale Down
            self.servers = max(self.min_servers, self.servers - 1)
        elif action == 2:  # Scale Up
            self.servers = min(self.max_servers, self.servers + 1)

        # ----------------------------
        # Simulate Traffic Change
        # ----------------------------
        traffic_change = np.random.randint(-10, 11)
        self.traffic = max(10, self.traffic + traffic_change)

        # ----------------------------
        # Update Metrics
        # ----------------------------
        self._update_metrics()

        # ----------------------------
        # Reward Function
        # ----------------------------
        reward = 0.0

        # Ideal CPU range
        if 40 <= self.cpu_util <= 70:
            reward += 2.0
        else:
            reward -= 1.0

        # SLA violation penalty
        if self.cpu_util > 80:
            reward -= 3.0

        # Cost penalty for more servers
        reward -= 0.2 * self.servers

        # ----------------------------
        # Termination Condition
        # ----------------------------
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._get_state(), reward, terminated, truncated, {}

    # --------------------------------------------------
    # Metric Simulation Logic
    # --------------------------------------------------
    def _update_metrics(self):
        """
        CPU increases with traffic and decreases with more servers.
        Latency increases sharply when CPU is high.
        """

        load_per_server = self.traffic / self.servers
        self.cpu_util = np.clip(load_per_server * 1.2, 5, 100)

        if self.cpu_util < 70:
            self.latency = 100 + self.cpu_util
        else:
            self.latency = 200 + (self.cpu_util - 70) * 3

    def render(self):
        print(
            f"Step: {self.current_step} | "
            f"Servers: {self.servers} | "
            f"Traffic: {self.traffic} | "
            f"CPU: {self.cpu_util:.2f}% | "
            f"Latency: {self.latency:.2f}ms"
        )

if __name__ == "__main__":
    env = CloudEnv()
    state, _ = env.reset()

    for _ in range(5):
        action = env.action_space.sample()
        state, reward, _, _, _ = env.step(action)
        env.render()
