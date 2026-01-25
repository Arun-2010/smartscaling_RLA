"""
Demo script to showcase the Fake Cloud Environment.

This demonstrates how the cloud simulator works:
- Traffic changes dynamically
- CPU utilization responds to traffic and server count
- Latency increases when CPU is high
- Agent can scale up/down servers
"""

from smartscaling_rla.envs.cloud_env import CloudEnv
import time


def demo_fake_cloud():
    """Run a demonstration of the fake cloud environment."""
    
    print("=" * 70)
    print("FAKE CLOUD ENVIRONMENT DEMONSTRATION")
    print("=" * 70)
    print("\nThis simulates a cloud infrastructure with:")
    print("  - Dynamic traffic patterns")
    print("  - CPU utilization based on load")
    print("  - Latency that increases with high CPU")
    print("  - Ability to scale servers up/down")
    print("\n" + "=" * 70)
    
    # Create the fake cloud environment
    env = CloudEnv()
    state, _ = env.reset()
    
    print("\nInitial State:")
    env.render()
    
    print("\n" + "-" * 70)
    print("Running 10 random scaling actions...")
    print("-" * 70)
    
    actions_map = {0: "Scale Down", 1: "Do Nothing", 2: "Scale Up"}
    
    total_reward = 0.0
    
    for step in range(10):
        # Take a random action
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nAction: {actions_map[action]}")
        print(f"Reward: {reward:+.2f} | Total Reward: {total_reward:+.2f}")
        env.render()
        
        if terminated or truncated:
            print("\nEpisode ended!")
            break
        
        time.sleep(0.2)  # Small delay for readability
    
    print("\n" + "=" * 70)
    print(f"Demo Complete! Total Reward: {total_reward:+.2f}")
    print("=" * 70)
    
    print("\nKey Observations:")
    print("  - When CPU is 40-70%: Good performance (positive reward)")
    print("  - When CPU > 80%: SLA violation (penalty)")
    print("  - More servers = Higher cost (penalty)")
    print("  - The RL agent learns to balance performance vs cost!")
    print()


if __name__ == "__main__":
    demo_fake_cloud()
