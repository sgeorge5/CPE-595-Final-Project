"""Compatibility helpers for preserved notebook results and regression checks."""
from stable_baselines3 import PPO
from levels.config_rl import PPO_V4_MODEL_PATH, PPO_V4_SETTINGS, RLConfig, SEED

MODELS = {"original": PPO_V4_MODEL_PATH, "A": "models/ppo_v4_improved_A.zip",
          "B": "models/ppo_v4_improved_B.zip"}




def experiment_config(experiment):
    return RLConfig(goal="all", free_balls=True, endgame_bonus=experiment == "B")


def verify_original_settings():
    model = PPO.load(PPO_V4_MODEL_PATH, device="cpu")
    assert model.num_timesteps == 100_352 and model.seed == SEED
    assert model.observation_space.shape == (155,) and model.action_space.n == 21
    for key, value in PPO_V4_SETTINGS.items():
        assert getattr(model, key) == value, key
    assert type(model.policy).__name__ == "ActorCriticPolicy"
    return model
