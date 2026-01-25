# Q-LEARNING INTERVIEW CHEAT SHEET
## Quick Reference for Technical Interviews

---

## 🎯 ELEVATOR PITCH (30 seconds)

"Q-learning is a model-free reinforcement learning algorithm that learns the value of taking actions in different states. It maintains a Q-table where Q(s,a) represents the expected future reward of taking action 'a' in state 's'. The agent uses epsilon-greedy exploration to balance trying new actions versus using known good actions, and updates Q-values using temporal difference learning. It's proven to converge to the optimal policy and works well for discrete state-action spaces like cloud auto-scaling."

---

## 📐 THE FORMULA (MEMORIZE THIS!)

```
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
```

**In words**: "New Q-value equals old Q-value plus learning rate times the TD error"

**TD Error** = `[r + γ max Q(s',a') - Q(s,a)]`
- What we got minus what we expected

---

## 🔑 KEY CONCEPTS

### 1. Q-Value
**Q(s,a)** = Expected total future reward starting from state s, taking action a, then following optimal policy

### 2. Epsilon-Greedy
```
if random() < ε:
    return random_action()  # Explore
else:
    return argmax(Q[s])     # Exploit
```

### 3. Temporal Difference (TD) Learning
- Learn from difference between prediction and reality
- Don't wait for episode to end
- Update immediately after each step

### 4. Off-Policy
- Learn optimal policy while following exploratory policy
- Can learn from any experience (even mistakes!)

---

## 💬 COMMON INTERVIEW QUESTIONS

### Q1: "What is Q-learning?"
**A**: "Q-learning is a model-free, off-policy RL algorithm that learns action-value functions. It estimates Q(s,a) - the expected return of taking action a in state s - and uses these estimates to derive an optimal policy."

### Q2: "How does Q-learning differ from SARSA?"
**A**: "Q-learning is off-policy (learns optimal policy while exploring), SARSA is on-policy (learns the policy it's following). Q-learning updates use max Q(s',a'), SARSA uses the actual next action taken."

### Q3: "What is the exploration-exploitation dilemma?"
**A**: "It's the trade-off between trying new actions to discover better strategies (exploration) versus using known good actions to maximize reward (exploitation). We solve it with epsilon-greedy: explore with probability ε, exploit otherwise."

### Q4: "How do you know Q-learning has converged?"
**A**: "When Q-values stop changing significantly and episode rewards stabilize. Formally, convergence is guaranteed if all state-action pairs are visited infinitely often and learning rate decays appropriately."

### Q5: "What are the limitations of tabular Q-learning?"
**A**: 
- Doesn't scale to large state spaces (table grows exponentially)
- No generalization to unseen states
- Requires discrete states/actions
- Solution: Use function approximation (Deep Q-Networks)

### Q6: "Explain the update rule intuitively"
**A**: "We take an action and observe the reward and next state. We calculate what the total value should be (reward + discounted future value) and compare it to our current estimate. We then update our estimate to move toward this target, controlled by the learning rate."

### Q7: "Why do we need a discount factor?"
**A**: "The discount factor (γ) determines how much we value future rewards. γ=0 means only care about immediate reward (myopic). γ=1 means value all future rewards equally. γ=0.95 is typical - we prefer sooner rewards but still consider the future."

### Q8: "How would you apply Q-learning to cloud auto-scaling?"
**A**: "States would be (CPU, traffic, server count), actions would be (scale up/down/maintain). Rewards would balance performance (keep CPU in optimal range) versus cost (minimize servers). The agent learns when to scale based on current conditions to optimize this trade-off."

---

## 📊 ALGORITHM PSEUDOCODE

```python
# Initialize
Q[s,a] = 0 for all s, a
epsilon = 1.0

# Training
for episode in episodes:
    s = env.reset()
    
    for step in steps:
        # Choose action (epsilon-greedy)
        if random() < epsilon:
            a = random_action()
        else:
            a = argmax(Q[s])
        
        # Take action
        s', r, done = env.step(a)
        
        # Update Q-value
        if done:
            target = r
        else:
            target = r + gamma * max(Q[s'])
        
        Q[s,a] = Q[s,a] + alpha * (target - Q[s,a])
        
        s = s'
        if done: break
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * decay_rate)
```

---

## 🎓 WHITEBOARD EXAMPLE

**Problem**: Robot learning to navigate a 3x3 grid to goal

```
Grid:        Q-Table (initially zeros):
S . .        State 0: [0, 0, 0, 0]  (up, down, left, right)
. . .        State 1: [0, 0, 0, 0]
. . G        ...
             State 8: [0, 0, 0, 0]

After training:
State 0: [0, 0.5, 0, 0.8]  → Best: right (0.8)
State 1: [0, 0.6, 0.5, 0.9] → Best: right (0.9)
...
State 7: [0.9, 0, 0.8, 0.95] → Best: right (0.95) to goal!
```

**Walk through one update**:
1. State s=0, choose action a=right (random exploration)
2. Get reward r=0, next state s'=1
3. Current Q[0,right] = 0
4. Target = 0 + 0.9 × max(Q[1]) = 0 + 0.9 × 0 = 0
5. Q[0,right] = 0 + 0.1 × (0 - 0) = 0 (no change yet)

After many episodes, values propagate back from goal!

---

## 🔢 HYPERPARAMETERS

| Parameter | Typical | Too Low | Too High |
|-----------|---------|---------|----------|
| α (learning rate) | 0.1 | Learns slowly | Unstable, oscillates |
| γ (discount) | 0.95 | Myopic | Slow convergence |
| ε start | 1.0 | Insufficient exploration | Wastes time |
| ε end | 0.05 | No adaptation | Too random |
| ε decay | 300 eps | Premature convergence | Slow learning |

---

## ⚡ QUICK FACTS

- **Invented**: 1989 by Chris Watkins
- **Type**: Model-free, off-policy, value-based
- **Complexity**: O(|S| × |A|) space, O(1) update time
- **Convergence**: Guaranteed with proper conditions
- **Best for**: Discrete, small-medium state spaces
- **Not for**: Continuous states/actions, huge spaces

---

## 🎯 CODE SNIPPET (MEMORIZE)

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
    
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s,a] += self.alpha * (target - self.Q[s,a])
```

---

## 🚨 COMMON MISTAKES

1. **Forgetting to decay epsilon** → Never stops exploring
2. **Using same action in update as next action** → That's SARSA, not Q-learning!
3. **Not handling terminal states** → Q-value should be just reward
4. **Learning rate too high** → Unstable learning
5. **Insufficient exploration** → Gets stuck in local optimum

---

## 💡 PRO TIPS FOR INTERVIEWS

1. **Draw the Q-table** - Visual helps explain
2. **Use concrete example** - Grid world or your project
3. **Mention limitations** - Shows depth of understanding
4. **Compare to alternatives** - SARSA, DQN, Policy Gradients
5. **Discuss convergence** - Shows theoretical knowledge
6. **Explain epsilon decay** - Shows practical experience

---

## 🎤 CLOSING STATEMENT

"Q-learning is elegant because it's simple yet powerful. With just a table and one update rule, it can learn optimal policies for complex problems. While it has limitations in large state spaces, it's perfect for problems like cloud auto-scaling where states are discrete and interpretability matters. For larger problems, we can extend it to Deep Q-Networks using neural networks as function approximators."

---

## 📝 PRACTICE QUESTIONS

1. Derive the Q-learning update from Bellman equation
2. Prove Q-learning is off-policy
3. Compare Q-learning vs Monte Carlo methods
4. Design Q-learning for traffic light control
5. Explain why we need epsilon-greedy vs pure greedy
6. Calculate Q-value update given: Q(s,a)=0.5, r=1, γ=0.9, max Q(s')=0.8, α=0.1

**Answer to #6**:
```
target = 1 + 0.9 × 0.8 = 1.72
error = 1.72 - 0.5 = 1.22
new Q = 0.5 + 0.1 × 1.22 = 0.622
```

---

**Print this before interviews!** 🎯

**Last Updated**: 2026-01-25  
**Status**: Interview Ready ✓
