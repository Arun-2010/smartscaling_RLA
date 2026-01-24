from smartscaling_rla.simulation.run_simulation import main


def test_simulation_runs_few_episodes(monkeypatch):
    """
    Run a very small training loop to ensure everything wires up correctly.
    """
    from smartscaling_rla import config

    # Temporarily reduce number of episodes for a quick test
    original_num_episodes = config.training_config.num_episodes
    config.training_config.num_episodes = 5

    try:
        episodes, rewards = main()
        assert len(episodes) == len(rewards) == 5
    finally:
        # Restore original value
        config.training_config.num_episodes = original_num_episodes

