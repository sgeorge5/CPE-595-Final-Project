"""Compatibility helpers for preserved notebook results and regression checks."""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from levels.config_rl import PPO_V4_RESULTS_DIR, RLConfig
from levels.visualize_data import record_gameplay




def record_version4_demo():
    selection = pd.read_csv(os.path.join(PPO_V4_RESULTS_DIR, "selected_model.csv")).iloc[0]
    games = pd.read_csv(os.path.join(selection.results_dir, "evaluation_episodes.csv"))
    games = games[games.strategy.isin(["PPO", "PPO sampled"])]
    example = games.sort_values(["success", "pegs_cleared", "shots_used", "seed"],
                                ascending=[False, False, True, True]).iloc[0]
    logged = pd.read_csv(os.path.join(selection.results_dir, "evaluation_shots.csv"))
    logged = logged[(logged.strategy == example.strategy) & (logged.seed == example.seed)]
    config = RLConfig(goal="all", free_balls=True)
    model = PPO.load(selection.menu_model_path, device="cpu")
    replay, table = record_gameplay(model, int(example.seed), config, actions=logged.action.tolist())
    for key in ("pegs_cleared", "shot_score", "shot_free_balls", "balls_remaining", "free_balls_earned"):
        np.testing.assert_array_equal(table[key], logged[key])
    return replay, table, example
