"""Compatibility helpers for preserved notebook results and regression checks."""
from stable_baselines3 import PPO
from levels.config_rl import RLConfig, SEED
from levels.ppo_v4_further import MODELS as PREVIOUS_MODELS, settings_for
from levels.train_ppo import validation_score

START = 166_912
MENU_MEAN_GAIN = 0.25
MODELS = {
    "D": PREVIOUS_MODELS["D"],
    "control": "models/ppo_v4_improved_F_control.zip",
    "F": "models/ppo_v4_improved_F_best.zip",
}


def config_for(experiment):
    return RLConfig(goal="all", free_balls=True, endgame_bonus=True,
                    early_peg_bonus=experiment == "F")


def verify_source():
    model = PPO.load(MODELS["D"], device="cpu")
    assert model.num_timesteps == START and model.seed == SEED
    assert model.observation_space.shape == (155,) and model.action_space.n == 21
    for key, value in settings_for("D").items():
        assert getattr(model, key) == value, key
    return model


def checkpoint_rank(row):
    return validation_score(row) + (-row["peg32_left_games"],)


def promote_f(baseline, candidate):
    # Peg 32 is a checkpoint tie-breaker, not a reason by itself to change the menu.
    old, new = validation_score(baseline), validation_score(candidate)
    if new[:2] != old[:2]:
        return new[:2] > old[:2]
    return new[2] - old[2] >= MENU_MEAN_GAIN
