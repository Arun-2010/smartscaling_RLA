smartscaling_RLA
<small>
smartscaling_RLA 📌 Project Overview This project demonstrates how Reinforcement Learning (RL) can be used to 
perform automatic cloud resource scaling in a simulated environment. Instead of using fixed rule-based autoscaling 
(like CPU > 80%), we train an RL agent that learns optimal scaling decisions based on system performance and cost. 
⚠️ Note: This is a simulation, not a real cloud system like AWS or Kubernetes. The goal is to prove intelligence 
and learning, not infrastructure deployment. 🎯 Problem Statement Traditional autoscaling systems use static rules 
such as: Scale up if CPU > 80% Scale down if CPU < 40% These rules: Do not adapt to changing workloads Can cause 
over-provisioning or SLA violations This project solves the problem by using a Reinforcement Learning agent that: 
Observes system metrics Takes scaling actions Learns from rewards and penalties 🧠 Solution Approach The system is 
divided into three simple components: Fake Cloud Environment Simulates traffic, CPU usage, latency, and servers RL 
Agent (Brain) Learns when to scale up or down Optimizes performance and cost Dashboard (Visualization) Shows CPU 
trends, server count, and agent actions
</small>
================

A beginner-friendly Reinforcement Learning project that learns how to auto-scale cloud servers using **tabular Q-learning** (not deep RL).

This repo is designed to be **simple**, **interview-friendly**, and easy to explain: it includes a **fake cloud simulator**, a **Q-learning agent**, a **training loop**, and a **Matplotlib dashboard** for screenshots.

---

## Problem statement

In cloud systems, we need to decide **when to add or remove servers** as traffic changes.

- Scale up too late → high latency / SLA violations  
- Scale down too slowly → wasted compute / higher cost

The key challenge is balancing **performance** and **cost** under **uncertain, changing workloads**.

---

## Why rule-based autoscaling fails

Traditional autoscaling uses fixed rules (thresholds + cooldowns), for example:

- “Scale up if CPU > 80%”
- “Scale down if CPU < 40%”

These approaches often fail because they are:

- **Brittle**: thresholds tuned for one workload can break for another.
- **Reactive**: they respond late and can’t learn from past mistakes.
- **Hard to tune**: small parameter changes can cause oscillation (thrashing).
- **Not goal-aware**: they don’t naturally optimize long-term cost vs. SLA trade-offs.

---

## Reinforcement Learning solution (Q-learning)

This project trains an RL agent that learns an autoscaling policy by interacting with a simulated environment:

- **State**: a small discrete representation of the system (instances + load bucket)
- **Actions**: `scale down`, `do nothing`, `scale up`
- **Reward**: combines performance (utilization / SLA safety) and cost (server usage)

The agent uses **tabular Q-learning** with ε-greedy exploration:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big)
\]

---

## Architecture overview

- **Fake Cloud Environment (Gym-style)**: `smartscaling_rla/envs/autoscaling_env.py`
  - Simulates time-varying traffic, server capacity, and utilization.
  - Exposes `reset()` / `step()` like OpenAI Gym / Gymnasium.

- **Reinforcement Learning Agent**: `smartscaling_rla/agents/q_learning_agent.py`
  - Tabular Q-table, ε-greedy action selection, standard Q-learning update.

- **Training + Metrics Logging**: `smartscaling_rla/training/train_qlearning.py`
  - Runs **at least 500 episodes**
  - Logs per-step metrics and saves them for visualization.

- **Dashboard / Visualization (Matplotlib only)**: `smartscaling_rla/visualize/dashboard.py`
  - CPU utilization, server count, reward over time, and actions taken.
  - Designed to produce clean plots suitable for README screenshots.

---

## Reward design (high level)

The reward function encourages “good autoscaling behavior”:

- **Performance / utilization shaping**: prefer a “healthy” utilization band and penalize deviation.
- **Infrastructure cost penalty**: penalize using many servers to reduce spend.
- **SLA violation penalty**: strong penalty when overloaded (utilization > 1.0).
- **Idle resource penalty**: penalize over-provisioning when utilization is very low.
- **Action shaping**: discourage doing nothing when action is clearly needed, discourage invalid actions, and provide incentives for correct scale-down under idle conditions.

This reward design helps the agent learn both **reactive safety** (avoid overload) and **cost efficiency** (scale down when idle).

---

## How to run the project

### Setup

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### Train (recommended: saves metrics)

```bash
python -m smartscaling_rla.training.train_qlearning
```

This creates a run folder like:

```text
runs/
  20260128_151547/
    metrics.npz
    run_config.json
```

### Visualize (Matplotlib dashboard)

Replace the run directory with the one printed by training:

```bash
python -m smartscaling_rla.visualize.dashboard --run-dir runs\20260128_151547
```

### Optional: simple demo run (no saved metrics)

```bash
python -m smartscaling_rla.simulation.run_simulation
```

---

## Future improvements

- **More realistic simulator**
  - instance warm-up delays, request queues, latency modeling
  - traffic traces from real workloads
- **Baselines**
  - implement classic threshold/cooldown autoscalers and compare cost/SLA metrics
- **Richer state**
  - include latency, queue length, error rate, trend features
- **More advanced RL**
  - DQN / PPO, or offline RL using logged traces
- **Experiment tracking**
  - structured run summaries, hyperparameter sweeps, reproducibility tooling


