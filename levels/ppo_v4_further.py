"""Compatibility helpers for preserved notebook results and regression checks."""
from stable_baselines3 import PPO
from levels.config_rl import PPO_V4_SETTINGS, RLConfig, SEED
from levels.ppo_v4_improved import MODELS as SAVED_MODELS

START = 125_952
MODELS = {
    "A": SAVED_MODELS["A"],
    "B": SAVED_MODELS["B"],
    "C": "models/ppo_v4_improved_C_best.zip",
    "D": "models/ppo_v4_improved_D_best.zip",
}


def config_for(experiment):
    return RLConfig(goal="all", free_balls=True, endgame_bonus=experiment != "A",
                    ppo_completion_bonus=300 if experiment == "C" else 100)


def settings_for(experiment):
    settings = dict(PPO_V4_SETTINGS)
    if experiment == "D":
        settings["ent_coef"] = 0.005
    return settings


def verify_source():
    model = PPO.load(MODELS["B"], device="cpu")
    assert model.num_timesteps == START and model.seed == SEED
    assert model.action_space.n == 21 and model.observation_space.shape == (155,)
    for key, value in PPO_V4_SETTINGS.items():
        assert getattr(model, key) == value, key
    return model
