"""Read-only checks of the completed 121-angle experiment."""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv


def test_saved_checkpoint_and_original_action_spaces():
    model = PPO.load("models/ppo_121_angles/checkpoint_100352.zip", device="cpu")
    assert model.num_timesteps == 100352 and model.action_space.n == 121
    assert model.observation_space.shape == (155,)
    env = PeggleEnv(RLConfig(goal="all", free_balls=True, n_actions=121))
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (155,) and env.action_space.n == 121
        np.testing.assert_array_equal(env.config.angles_degrees, np.arange(-60, 61))
        assert RLConfig().n_actions == 21
    finally:
        env.close()


def test_saved_probe_and_validation_evidence():
    root = "results/ppo_121_angles"
    summary = pd.read_csv(root + "/diagnostics/isolated_peg32_summary.csv")
    attempts = pd.read_csv(root + "/diagnostics/isolated_peg32_angles.csv")
    assert len(summary) == 6 and summary.angles_tested.eq(121).all()
    assert len(attempts) == 726 and not attempts.hit_peg32.any()
    scores = pd.read_csv(root + "/121/validation_checkpoints.csv")
    assert len(scores) == 10 and scores.games.eq(50).all()
    assert scores.loc[scores.is_best, "timesteps"].tolist() == [100352]
