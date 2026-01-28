smartscaling_RLA
================

Beginner-friendly reinforcement learning project that uses **tabular Q-learning**
to learn an auto-scaling policy in a simple **OpenAI Gym-style** environment.

## What you can do with this repo

- Train a Q-learning agent to **scale up / scale down / do nothing**
- Track and save **CPU utilization**, **server count**, and **reward per step**
- Plot learning curves and analyze runs

## Install

From the project root:

```bash
pip install -e .
```

## Run (simple demo)

This runs training and shows a Matplotlib plot of episode rewards:

```bash
python -m smartscaling_rla.simulation.run_simulation
```

## Run (training + metrics logging + saving) ✅

This run collects and saves the metrics you asked for:

- CPU utilization
- Server count
- Reward per step
- Occasional debug prints

Run:

```bash
python -m smartscaling_rla.training.train_qlearning
```

Outputs are saved under `runs/<timestamp>/`:

- `metrics.npz` (NumPy compressed arrays)
- `run_config.json` (configs used for reproducibility)

## Project structure (key files)

- **`smartscaling_rla/envs/autoscaling_env.py`**: Gym-style environment (state/action/reward)
- **`smartscaling_rla/agents/q_learning_agent.py`**: tabular Q-learning agent
- **`smartscaling_rla/training/train_qlearning.py`**: training + per-step logging + saving metrics
- **`smartscaling_rla/utils/plotting.py`**: Matplotlib plotting helpers

