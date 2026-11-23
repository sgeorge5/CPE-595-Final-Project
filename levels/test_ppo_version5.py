import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from levels.config_rl import PPO_V4_MODEL_PATH, PPO_V5_MODEL_PATH, PPO_V5_RESULTS_DIR, RLConfig
from levels.peggle_env import PeggleEnv
from levels.ppo_version5 import version5_config, performance
from play import load_game_mode


def test_version5_action_mapping_and_shared_rules():
    v4 = PeggleEnv(RLConfig(goal="all", free_balls=True), record_trajectory=True)
    v5 = PeggleEnv(version5_config(), record_trajectory=True)
    np.testing.assert_array_equal(v4.config.coordinates, v5.config.coordinates)
    assert v5.config.n_pegs == 77 and v5.action_space.n == 41 and v4.action_space.n == 21
    np.testing.assert_array_equal(v5.config.angles_degrees, np.arange(-60, 61, 3))
    np.testing.assert_array_equal(v5.config.angles_degrees[::2], v4.config.angles_degrees)
    np.testing.assert_array_equal(v4.reset(seed=50042)[0], v5.reset(seed=50042)[0])
    assert v5.observation_space.shape == (155,)
    earned_ball = False
    for a in (0, -36, -30, -60, 60):
        action4 = (a + 60) // 6
        obs4, reward4, done4, truncated4, info4 = v4.step(action4)
        obs5, reward5, done5, truncated5, info5 = v5.step(2 * action4)
        np.testing.assert_array_equal(obs4, obs5)
        assert (reward4, done4, truncated4, info4) == (reward5, done5, truncated5, info5)
        assert v4.trajectory == v5.trajectory
        earned_ball = earned_ball or info5["shot_free_balls"] > 0
    assert earned_ball
    for action, angle in ((0, -60), (20, 0), (40, 60)):
        v5.reset(seed=50042)
        assert v5.step(action)[-1]["angle_degrees"] == angle
    check_env(PeggleEnv(version5_config()))


def test_exact_orange_arrangement_can_be_replayed():
    env = PeggleEnv(version5_config())
    observation, _ = env.reset(seed=50042)
    orange_ids = np.flatnonzero(observation[:77]).tolist()
    replay, _ = env.reset(seed=42, options={"orange_peg_ids": orange_ids})
    np.testing.assert_array_equal(observation, replay)


def test_saved_action_spaces_and_menu_selection():
    model, config = load_game_mode("1")
    assert model.action_space.n == config.n_actions == 21
    assert model.observation_space.shape == (155,) and config.max_shots == 10 and not config.free_balls
    for version in range(1, 4):
        suffix = "_best" if version == 3 else ""
        saved = PPO.load(f"models/ppo_version{version}{suffix}.zip", device="cpu")
        assert saved.action_space.n == 21 and saved.observation_space.shape == (78,)
    v4 = PPO.load(PPO_V4_MODEL_PATH, device="cpu")
    assert v4.action_space.n == 21 and v4.observation_space.shape == (155,)
    selection_path = os.path.join(PPO_V5_RESULTS_DIR, "selection.csv")
    if os.path.isfile(selection_path):
        selection = pd.read_csv(selection_path).iloc[0]
        v5 = PPO.load(PPO_V5_MODEL_PATH, device="cpu")
        assert v5.action_space.n == 41 and v5.num_timesteps == selection.selected_timestep
        ppo, config = load_game_mode("2")
        assert ppo.action_space.n == config.n_actions == (41 if selection.menu_version == 5 else 21)
        assert config.free_balls and ppo.observation_space.shape == (155,)


def test_selection_prefers_completion_and_sampled_win_rate():
    loss = dict(wins=0, completion_pct=0, avg_shots_success=float("nan"), avg_pegs_cleared=76)
    win = dict(wins=1, completion_pct=1, avg_shots_success=12, avg_pegs_cleared=70)
    assert performance(False, 10, win) > performance(True, 9, loss) > performance(False, 10, loss)
    faster = dict(win, avg_shots_success=11)
    assert performance(False, 10, faster) > performance(False, 10, win)
