"""
Inspector script for the Smart Scaling RL Agent.
This script trains the agent completely and then audits its behavior.
"""
import numpy as np
import time
from smartscaling_rla.envs import AutoScalingEnv
from smartscaling_rla.agents import QLearningAgent
from smartscaling_rla.config import env_config, q_learning_config
import matplotlib.pyplot as plt

def inspect_agent():
    print("="*60)
    print("[1/3] TRAINING AGENT (500 episodes)...")
    print("="*60)
    
    # Setup environment and agent
    env = AutoScalingEnv(config=env_config)
    
    # Agent parameters must match environment state/action space
    # State: [instances, load_bucket]
    # We call QLearningAgent to handle the Q-Table
    agent = QLearningAgent(
        num_instances=env.config.max_instances,
        max_load_bucket=env.max_load_bucket,
        num_actions=env.action_space.n,
        config=q_learning_config,
    )
    
    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    start_time = time.time()
    rewards_history = []
    
    for episode in range(500):
        agent.begin_episode()
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(200):
            # Choose action (epsilon-greedy)
            action = agent.choose_action(state)
            
            # Step env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Learn
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"    Episode {episode+1}/500 | Avg Reward: {np.mean(rewards_history[-100:]):.2f} | Epsilon: {agent._current_epsilon():.3f}")

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")
    
    # ---------------------------------------------------------
    # INSPECTION / AUDIT
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("[2/3] AUDITING LEARNED POLICY")
    print("="*60)
    print("Checking how the agent responds to critical situations:\n")
    
    actions_map = {0: "SCALE DOWN (Remove Server)", 1: "DO NOTHING", 2: "SCALE UP (Add Server)"}
    
    # AUDIT 1: HIGH LOAD, FEW SERVERS -> SHOULD SCALE UP
    # State = [Instances=1, Load=10 (Max)]
    audit_state_1 = np.array([1, 10]) 
    action_1_idx = agent.choose_action(audit_state_1) # Note: epsilon is low now
    # Force greedy for audit
    q_values_1 = agent.Q[agent._state_to_index(audit_state_1)]
    best_action_1 = np.argmax(q_values_1)
    
    print(f"SCENARIO 1: CRITICAL OVERLOAD")
    print(f"  Conditions: 1 Server handling MAX Load (10)")
    print(f"  Agent sees state: {audit_state_1}")
    print(f"  Agent Q-Values: {dict(zip(actions_map.values(), np.round(q_values_1, 2)))}")
    print(f"  DECISION: {actions_map[best_action_1]}")
    if best_action_1 == 2:
        print("  RESULT: PASS (Agent correctly chose to Scale Up) [OK]")
    else:
        print("  RESULT: FAIL (Agent did NOT Scale Up) [FAIL]")
        
    print("-" * 40)

    # AUDIT 2: LOW LOAD, MANY SERVERS -> SHOULD SCALE DOWN
    # State = [Instances=10, Load=0 (Min)]
    audit_state_2 = np.array([10, 0])
    q_values_2 = agent.Q[agent._state_to_index(audit_state_2)]
    best_action_2 = np.argmax(q_values_2)
    
    print(f"SCENARIO 2: RESOURCE WASTE")
    print(f"  Conditions: 10 Servers handling NO Load (0)")
    print(f"  Agent sees state: {audit_state_2}")
    print(f"  Agent Q-Values: {dict(zip(actions_map.values(), np.round(q_values_2, 2)))}")
    print(f"  DECISION: {actions_map[best_action_2]}")
    if best_action_2 == 0:
        print("  RESULT: PASS (Agent correctly chose to Scale Down) [OK]")
    else:
        print("  RESULT: FAIL (Agent did NOT Scale Down) [FAIL]")

    # ---------------------------------------------------------
    # VISUAL DEMO
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("[3/3] LIVE DEMO (10 Steps)")
    print("="*60)
    
    state, info = env.reset()
    print(f"Initial State: {info} | State Vector: {state}")
    
    for i in range(10):
        # Force greedy action for demo
        q_idx = agent._state_to_index(state)
        action = np.argmax(agent.Q[q_idx])
        
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Action={actions_map[action]} -> Load={info['load']:.1f}, Instances={info['instances']}")
        state = next_state
        time.sleep(0.1)

    print("\n" + "="*60)
    print("DONE. Training plot saved to 'training_results.png'")
    print("="*60)

if __name__ == "__main__":
    inspect_agent()
