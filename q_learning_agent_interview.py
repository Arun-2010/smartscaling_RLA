"""
TABULAR Q-LEARNING AGENT FOR CLOUD AUTO-SCALING
================================================

This is a simple, interview-friendly implementation of Q-learning
for the cloud auto-scaling problem.

PROBLEM:
--------
We have a cloud environment with:
- Dynamic traffic (requests coming in)
- Servers that can be scaled up/down
- Goal: Balance performance (low CPU) vs cost (fewer servers)

SOLUTION:
---------
Use Q-learning to learn the optimal scaling policy!

Q-LEARNING BASICS:
------------------
Q-learning is a model-free reinforcement learning algorithm that learns
the value of taking action 'a' in state 's'.

The Q-value Q(s,a) represents: "How good is it to take action 'a' in state 's'?"

UPDATE RULE:
    Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
                         └─────┬─────┘   └──┬──┘
                           TD Target    Current Q
    Where:
    - α (alpha) = learning rate (how fast we update)
    - γ (gamma) = discount factor (how much we value future rewards)
    - r = immediate reward
    - s' = next state
    - max Q(s',a') = best possible future value

EXPLORATION vs EXPLOITATION:
----------------------------
We use ε-greedy policy:
- With probability ε: explore (random action)
- With probability 1-ε: exploit (best known action)
- ε decays over time (explore early, exploit later)
"""

import numpy as np
from typing import Tuple, Optional


class QLearningAgent:
    """
    Tabular Q-Learning Agent for Cloud Auto-Scaling.
    
    This agent maintains a Q-table (2D array) where:
    - Rows = States (different cloud configurations)
    - Columns = Actions (scale down, do nothing, scale up)
    - Values = Expected future reward for taking that action in that state
    """
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 300
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            num_states: Total number of possible states
            num_actions: Total number of possible actions (usually 3: down, stay, up)
            learning_rate (α): How much to update Q-values (0-1)
                              - 0 = never learn
                              - 1 = only consider latest experience
            discount_factor (γ): How much to value future rewards (0-1)
                                - 0 = only care about immediate reward
                                - 1 = value all future rewards equally
            epsilon_start: Initial exploration rate (usually 1.0 = 100% random)
            epsilon_end: Final exploration rate (usually 0.05 = 5% random)
            epsilon_decay_episodes: How many episodes to decay epsilon over
        """
        # Store hyperparameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        # Initialize Q-table to zeros
        # Shape: [num_states, num_actions]
        # All Q-values start at 0 (we have no knowledge yet)
        self.Q = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # Track current episode for epsilon decay
        self.episode = 0
        
        print(f"[OK] Q-Learning Agent Initialized!")
        print(f"   - Q-table shape: {self.Q.shape}")
        print(f"   - Learning rate (alpha): {self.alpha}")
        print(f"   - Discount factor (gamma): {self.gamma}")
        print(f"   - Epsilon: {self.epsilon_start} -> {self.epsilon_end}")
    
    
    def begin_episode(self) -> None:
        """
        Call this at the start of each new episode.
        
        This increments the episode counter, which is used for
        epsilon decay (reducing exploration over time).
        """
        self.episode += 1
    
    
    def choose_action(self, state: int) -> int:
        """
        Choose an action using ε-greedy policy.
        
        EPSILON-GREEDY POLICY:
        ----------------------
        This balances exploration (trying new things) and exploitation
        (using what we know works).
        
        With probability ε:
            - Choose RANDOM action (exploration)
            - This helps us discover new strategies
        
        With probability 1-ε:
            - Choose BEST action from Q-table (exploitation)
            - This uses our learned knowledge
        
        Args:
            state: Current state index (integer)
        
        Returns:
            action: Chosen action index (0=scale down, 1=do nothing, 2=scale up)
        """
        # Get current epsilon value (decays over time)
        epsilon = self._get_epsilon()
        
        # EXPLORATION: Random action
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return int(action)
        
        # EXPLOITATION: Best action from Q-table
        # Get all Q-values for this state
        q_values = self.Q[state]
        
        # Choose action with highest Q-value
        # argmax returns the index of the maximum value
        action = np.argmax(q_values)
        return int(action)
    
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """
        Update Q-table using the Q-learning update rule.
        
        Q-LEARNING UPDATE RULE:
        -----------------------
        Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
                             └─────┬─────┘   └──┬──┘
                               TD Target    Current Q
        
        INTUITION:
        ----------
        1. We took action 'a' in state 's' and got reward 'r'
        2. We ended up in new state 's''
        3. We estimate total value as: r + γ * (best future value)
        4. We update our Q-value to move closer to this estimate
        
        TEMPORAL DIFFERENCE (TD) ERROR:
        -------------------------------
        TD Error = [r + γ max Q(s',a')] - Q(s,a)
                   └──────┬──────┘       └──┬──┘
                    What we got      What we expected
        
        If TD Error > 0: We got more than expected → increase Q-value
        If TD Error < 0: We got less than expected → decrease Q-value
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward received
            next_state: State we transitioned to
            done: Whether episode ended (terminal state)
        """
        # Get current Q-value for this (state, action) pair
        current_q = self.Q[state, action]
        
        # Calculate TD Target
        if done:
            # If episode ended, there's no future value
            td_target = reward
        else:
            # Otherwise, add discounted future value
            # max Q(s',a') = best possible value from next state
            max_next_q = np.max(self.Q[next_state])
            td_target = reward + self.gamma * max_next_q
        
        # Calculate TD Error
        td_error = td_target - current_q
        
        # Update Q-value
        # Move current Q-value towards TD target by learning rate α
        self.Q[state, action] = current_q + self.alpha * td_error
    
    
    def decay_epsilon(self) -> float:
        """
        Manually decay epsilon (alternative to automatic decay in begin_episode).
        
        Returns:
            Current epsilon value
        """
        return self._get_epsilon()
    
    
    def _get_epsilon(self) -> float:
        """
        Calculate current epsilon value with linear decay.
        
        EPSILON DECAY SCHEDULE:
        -----------------------
        Episode 0:                ε = epsilon_start (e.g., 1.0 = 100% exploration)
        Episode decay_episodes:   ε = epsilon_end (e.g., 0.05 = 5% exploration)
        Episode > decay_episodes: ε = epsilon_end (stays constant)
        
        This implements: "Explore early, exploit later"
        
        Returns:
            Current epsilon value
        """
        # If we've passed decay period, use minimum epsilon
        if self.episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        
        # Linear interpolation between start and end
        # fraction = how far through decay period we are (0 to 1)
        fraction = self.episode / max(1, self.epsilon_decay_episodes)
        
        # Linearly interpolate
        epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        return float(epsilon)
    
    
    def get_policy(self, state: int) -> int:
        """
        Get the greedy policy action (best action) for a given state.
        
        This is useful for:
        - Evaluating learned policy
        - Deploying agent in production (no exploration)
        
        Args:
            state: State to get action for
        
        Returns:
            Best action according to Q-table
        """
        return int(np.argmax(self.Q[state]))
    
    
    def get_q_values(self, state: int) -> np.ndarray:
        """
        Get all Q-values for a given state.
        
        Useful for debugging and visualization.
        
        Args:
            state: State to get Q-values for
        
        Returns:
            Array of Q-values for each action
        """
        return self.Q[state].copy()
    
    
    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to disk."""
        np.save(filepath, self.Q)
        print(f"[SAVED] Q-table saved to {filepath}")
    
    
    def load_q_table(self, filepath: str) -> None:
        """Load Q-table from disk."""
        self.Q = np.load(filepath)
        print(f"[LOADED] Q-table loaded from {filepath}")


# =============================================================================
# EXAMPLE USAGE FOR CLOUD AUTO-SCALING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Q-LEARNING AGENT DEMO FOR CLOUD AUTO-SCALING")
    print("=" * 70)
    
    # Define state space
    # State = (CPU level, Traffic level, Server count)
    # CPU levels: 3 (low, medium, high)
    # Traffic levels: 3 (low, medium, high)
    # Server counts: 10 (1-10 servers)
    num_states = 3 * 3 * 10  # 90 total states
    
    # Define action space
    # Actions: 0=scale down, 1=do nothing, 2=scale up
    num_actions = 3
    
    # Create agent
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=300
    )
    
    print("\n" + "=" * 70)
    print("SIMULATING TRAINING EPISODES")
    print("=" * 70)
    
    # Simulate a few training steps
    for episode in range(5):
        agent.begin_episode()
        
        # Simulate random state
        state = np.random.randint(0, num_states)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  State: {state}")
        
        # Choose action
        action = agent.choose_action(state)
        action_names = ["Scale Down", "Do Nothing", "Scale Up"]
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  Epsilon: {agent._get_epsilon():.3f}")
        
        # Simulate reward and next state
        reward = np.random.uniform(-2, 2)
        next_state = np.random.randint(0, num_states)
        done = False
        
        print(f"  Reward: {reward:+.2f}")
        print(f"  Next State: {next_state}")
        
        # Update Q-table
        agent.update(state, action, reward, next_state, done)
        
        # Show Q-values for this state
        q_vals = agent.get_q_values(state)
        print(f"  Q-values: {q_vals}")
    
    print("\n" + "=" * 70)
    print(f"[OK] DEMO COMPLETE!")
    print("=" * 70)
    print("\nKEY TAKEAWAYS FOR INTERVIEWS:")
    print("  1. Q-learning learns value of (state, action) pairs")
    print("  2. Uses TD learning: update based on difference between")
    print("     expected and actual returns")
    print("  3. Epsilon-greedy balances exploration vs exploitation")
    print("  4. No model needed - learns directly from experience")
    print("  5. Converges to optimal policy given enough exploration")
    print()
