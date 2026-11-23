"""Scoring tests without training or rewriting any existing saved results."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check

from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv, fever_multiplier, peggle_shot_score, free_balls_for_score


def test_fever_boundaries():
    for remaining, multiplier in [(25, 1), (16, 1), (15, 2), (11, 2), (10, 3), (7, 3),
                                  (6, 5), (4, 5), (3, 10), (1, 10), (0, 10)]:
        assert fever_multiplier(remaining) == multiplier


def test_free_ball_thresholds():
    for score, balls in [(0, 0), (24999, 0), (25000, 1), (74999, 1),
                         (75000, 2), (124999, 2), (125000, 3), (200000, 3)]:
        assert free_balls_for_score(score) == balls


def test_points_and_order():
    assert peggle_shot_score(["blue"], 25) == 10
    assert peggle_shot_score(["orange"], 25) == 100
    assert peggle_shot_score(["blue"] * 5 + ["orange"] * 2, 25) == 1750
    # At 16 remaining, the orange crosses the 2x threshold before it scores.
    assert peggle_shot_score(["blue", "orange"], 16) == (10 + 200) * 2
    assert peggle_shot_score(["orange", "blue"], 16) == (200 + 20) * 2
    assert peggle_shot_score(["blue"], 0) == 100
    assert peggle_shot_score([], 0) == 0


def test_version4_observation_and_old_defaults():
    for config, size in [(RLConfig(observation="compact"), 155),
                         (RLConfig(goal="all"), 78),
                         (RLConfig(goal="all", free_balls=True), 155)]:
        env = PeggleEnv(config)
        obs, info = env.reset(seed=42)
        assert config.n_pegs == 77 and env.action_space.n == 21
        assert obs.shape == (size,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        if config.free_balls:
            assert info["starting_balls"] == info["balls_remaining"] == 10
            assert obs[:77].sum() == 25 and obs[77:154].sum() == 0
            assert obs[-1] == np.float32(10 / 241)
        else:
            assert "free_balls_earned" not in info and "shot_score" not in info
    with pytest.raises(ValueError, match="only available for PPO"):
        RLConfig(free_balls=True)
    check_env(PeggleEnv(RLConfig(goal="all", free_balls=True)), skip_render_check=True)
    sb3_check(PeggleEnv(RLConfig(goal="all", free_balls=True)))


def test_earned_balls_and_shot_count(monkeypatch):
    env = PeggleEnv(RLConfig(goal="all", free_balls=True))
    env.reset(seed=42)

    def hit_oranges(angle):
        env.shot_hit_colors = []
        for peg in env.pegs:
            if peg.isOrange and not peg.isHit:
                env.shot_hit_colors.append(peg.color)
                peg.isHit = True
                peg.isVisible = False
        return 1, False

    monkeypatch.setattr(env, "_simulate_shot", hit_oranges)
    obs, reward, done, _, info = env.step(10)
    assert info["orange_remaining"] == 0 and info["pegs_remaining"] == 52
    assert info["shot_score"] == 215000 and info["shot_free_balls"] == 3
    assert info["free_balls_earned"] == 3 and info["shots_earning_free_balls"] == 1
    assert info["balls_remaining"] == 12 and info["shots_used"] == 1
    assert obs[-1] == np.float32(12 / 241) and env.observation_space.contains(obs)
    assert reward == 123 and not done  # RL reward is separate from 215,000 points.
    _, reward, done, _, second = env.step(10)
    assert second["shot_score"] == second["shot_free_balls"] == 0
    assert second["total_score"] == 215000 and second["free_balls_earned"] == 3
    assert second["balls_remaining"] == 11 and second["shots_used"] == 2
    assert reward == -2 and not done
    env.reset(seed=42)
    env.shots_remaining, env.shots_fired = 1, 9
    _, _, done, _, info = env.step(10)
    assert not done and info["balls_remaining"] == 3 and info["shots_used"] == 10


def test_real_contacts_and_legacy_physics_match():
    old = PeggleEnv(RLConfig(goal="all"), record_trajectory=True)
    new = PeggleEnv(RLConfig(goal="all", free_balls=True), record_trajectory=True)
    old.reset(seed=42)
    new.reset(seed=42)
    for action in (10, 0, 10):
        old.step(action)
        _, _, _, _, info = new.step(action)
        assert len(old.trajectory) == len(new.trajectory)
        for a, b in zip(old.trajectory, new.trajectory):
            assert a == {key: b[key] for key in a}
        assert len(new.shot_hit_colors) == info["new_pegs_cleared"]
    assert len(old.pegs) == len(new.pegs) == 77


def test_no_balls_and_all_pegs_termination():
    env = PeggleEnv(RLConfig(goal="all", free_balls=True))
    env.reset(seed=42)
    env.shots_remaining = 1
    env.shots_fired = 9
    # One last unreachable peg: an empty final shot must end the game.
    for peg in env.pegs:
        peg.isHit = True
    env.pegs[0].isHit = False
    env.pegs[0].pos = env.Vector(100, 300)
    _, reward, done, truncated, info = env.step(10)
    assert done and not truncated and not info["success"]
    assert info["balls_remaining"] == 0 and info["shots_used"] == 10 and reward == -2
    env.reset(seed=42)
    for peg in env.pegs:
        peg.isHit = True
    env.pegs[0].isHit = False
    env.pegs[0].pos = env.Vector(600, 300)
    _, reward, done, _, info = env.step(10)
    assert done and info["success"] and info["pegs_cleared"] == 77 and reward == 103


def test_dqn_has_original_ten_shot_budget():
    env = PeggleEnv(RLConfig(observation="compact"))
    env.reset(seed=42)
    done = False
    while not done:
        _, _, done, _, info = env.step(17)
        assert "shot_score" not in info and "free_balls_earned" not in info
    assert info["shots_used"] == 10 and info["shots_remaining"] == 0


def test_recorded_version4_state_matches_scoring_info():
    from levels.visualize_data import plot_gameplay
    import matplotlib.pyplot as plt

    env = PeggleEnv(RLConfig(goal="all", free_balls=True), record_trajectory=True)
    env.reset(seed=50042)
    board = env.board_records()
    _, _, _, _, info = env.step(10)
    frame = env.trajectory[-1]
    for key in ("shot_score", "shot_free_balls", "free_balls_earned", "balls_remaining", "shots_used"):
        assert frame[key] == info[key]
    figure = plot_gameplay(board, env.trajectory, len(env.trajectory)-1, 0, 1, env.config)
    assert f"Shot score: {info['shot_score']:,}" in figure.axes[0].get_title()
    assert f"Balls left: {info['balls_remaining']}" in figure.axes[0].get_title()
    plt.close(figure)


def test_saved_version4_menu_and_logged_game():
    import os
    import pandas as pd
    from levels.config_rl import PPO_V4_RESULTS_DIR
    from levels.train_ppo import play_episode
    from play import load_game_mode

    path = os.path.join(PPO_V4_RESULTS_DIR, "selected_model.csv")
    if not os.path.isfile(path):
        pytest.skip("No verified Version 4 model has been selected yet.")
    selected = pd.read_csv(path).iloc[0]
    model, config = load_game_mode("2", ppo_path=selected.menu_model_path, ppo_results=selected.results_dir)
    assert config.free_balls and config.goal == "all"
    assert model.observation_space.shape == (155,) and model.action_space.n == 21
    episode, trace = play_episode(config, "PPO", model, seed=50042)
    logged = pd.read_csv(os.path.join(selected.results_dir, "evaluation_shots.csv"))
    logged = logged[logged.strategy == "PPO"]
    for key in ("action", "pegs_cleared", "shot_score", "shot_free_balls", "balls_remaining"):
        np.testing.assert_array_equal(logged[key], [shot[key] for shot in trace])
    assert episode["shots_used"] == 10 + episode["free_balls_earned"] - episode["balls_remaining"]


def test_missing_version4_falls_back_to_previous_selection(tmp_path, monkeypatch):
    import os
    import pandas as pd
    import play
    from levels.config_rl import PPO_VERSIONS_DIR

    pd.DataFrame([dict(menu_model_path="missing_version4.zip", results_dir="missing_results")]).to_csv(
        os.path.join(tmp_path, "selected_model.csv"), index=False)
    monkeypatch.setattr(play, "PPO_V4_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(play, "PPO_V5_RESULTS_DIR", os.path.join(tmp_path, "missing_version5"))
    monkeypatch.setattr(play, "PPO_V4_IMPROVED_DIR", os.path.join(tmp_path, "missing_improved"))
    monkeypatch.setattr(play, "PPO_V4_FURTHER_DIR", os.path.join(tmp_path, "missing_further"))
    monkeypatch.setattr(play, "PPO_V4_SHOT_ORDER_DIR", os.path.join(tmp_path, "missing_shot_order"))
    previous = pd.read_csv(os.path.join(PPO_VERSIONS_DIR, "selected_model.csv")).iloc[0]
    assert play.saved_ppo_files() == (previous.menu_model_path, previous.results_dir)
