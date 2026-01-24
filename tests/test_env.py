import numpy as np

from smartscaling_rla.envs import AutoScalingEnv


def test_env_reset_and_step():
    env = AutoScalingEnv()
    state, info = env.reset(seed=123)

    # Basic shape / type checks
    assert isinstance(state, np.ndarray)
    assert "instances" not in info or isinstance(info.get("instances", 1), (int, float))

    # One environment step should not crash and should respect spaces
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(next_state)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

