import numpy as np

from smartscaling_rla.agents import QLearningAgent


def test_q_learning_agent_basic_update():
    num_instances = 5
    max_load_bucket = 3
    num_actions = 3
    agent = QLearningAgent(
        num_instances=num_instances,
        max_load_bucket=max_load_bucket,
        num_actions=num_actions,
    )

    state = np.array([3, 2], dtype=np.int32)
    next_state = np.array([4, 2], dtype=np.int32)
    action = 1
    reward = 1.0

    # Q-value before update
    s_idx = agent._state_to_index(state)
    before = float(agent.Q[s_idx, action])

    agent.begin_episode()
    agent.update(state, action, reward, next_state, done=False)

    after = float(agent.Q[s_idx, action])
    assert after != before

