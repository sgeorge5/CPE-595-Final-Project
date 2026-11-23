import hashlib
import os

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DQN

from levels import dqn_all_pegs as run
from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv


def test_same_observation_and_rules_as_ppo_v3():
    config = run.all_pegs_config()
    original = RLConfig(goal="all")
    np.testing.assert_array_equal(config.coordinates, original.coordinates)
    assert config.goal == "all" and config.observation == "hits" and len(config.coordinates) == 77
    assert not config.free_balls and not config.endgame_bonus and not config.early_peg_bonus
    assert config.max_shots == 10 and config.peg_reward == 5 and config.shot_penalty == 2
    assert config.completion_bonus == 100 and config.empty_shot_penalty == 0
    for name in ("dt", "ball_radius", "peg_radius", "gravity", "launch_force", "collision_samples",
                 "max_frames_per_shot", "stuck_frames"):
        assert getattr(config, name) == getattr(original, name)
    env = PeggleEnv(config)
    try:
        check_env(env, skip_render_check=True)
        assert env.action_space.n == 21 and env.observation_space.shape == (78,)
        np.testing.assert_array_equal(config.angles_degrees, np.arange(-60, 61, 6))
        assert tuple(config.angles_degrees[[0, 10, 20]]) == (-60, 0, 60)
        obs, info = env.reset(seed=42)
        np.testing.assert_array_equal(obs, np.array([0] * 77 + [1], dtype=np.float32))
        assert obs.dtype == np.float32 and info["orange_initial"] == 25
        assert sum(peg.color == "blue" for peg in env.pegs) == 52
    finally:
        env.close()


def test_colors_do_not_change_dynamics_or_reward_and_shots_end_at_ten():
    class FixedAction:
        def predict(self, observation, deterministic):
            assert deterministic and observation.dtype == np.float32
            return 0, None

    model = FixedAction()
    one, shots1 = run.play_game(model, "Greedy", 14)
    two, shots2 = run.play_game(model, "Greedy", 15)
    assert one["orange_peg_ids"] != two["orange_peg_ids"]
    assert one["shots_used"] == two["shots_used"] == 10
    assert not one["success"] and not two["success"]
    fields = ["action", "reward", "pegs_cleared", "new_pegs_cleared", "shots_remaining"]
    left, right = pd.DataFrame(shots1), pd.DataFrame(shots2)
    pd.testing.assert_frame_equal(left[fields], right[fields])
    assert left.shots_remaining.tolist() == list(range(9, -1, -1))
    assert left.reward.eq(5 * left.new_pegs_cleared - 2).all()
    assert -2 in left.reward.values and left.new_pegs_cleared.max() > 0


def test_only_full_clear_earns_completion_bonus(monkeypatch):
    env = PeggleEnv(run.all_pegs_config())
    try:
        env.reset(seed=42)
        orange = [peg for peg in env.pegs if peg.isOrange]
        blue = [peg for peg in env.pegs if not peg.isOrange]

        def simulated_hit(peg):
            def hit(_angle):
                peg.isHit = True
                return 1, False
            return hit

        for peg in orange[:-1]:
            peg.isHit = True
        monkeypatch.setattr(env, "_simulate_shot", simulated_hit(orange[-1]))
        _, reward, done, truncated, info = env.step(10)
        assert (reward, done, truncated, info["orange_remaining"]) == (3, False, False, 0)
        monkeypatch.setattr(env, "_simulate_shot", simulated_hit(blue[0]))
        _, reward, done, _, info = env.step(10)
        assert (reward, done, info["shots_remaining"]) == (3, False, 8)
        for peg in blue[1:-1]:
            peg.isHit = True
        monkeypatch.setattr(env, "_simulate_shot", simulated_hit(blue[-1]))
        _, reward, done, truncated, info = env.step(10)
        assert (reward, done, truncated, info["pegs_cleared"]) == (103, True, False, 77)
        assert info["shots_remaining"] == 7 and info["success"]
        assert "free_balls_earned" not in info and "shot_score" not in info
    finally:
        env.close()


def test_resume_restores_replay_and_keeps_prior_checkpoint(tmp_path, monkeypatch):
    root = str(tmp_path)
    model_dir = os.path.join(root, "models")
    results_dir = os.path.join(root, "results")
    monkeypatch.setattr(run, "MODEL_DIR", model_dir)
    monkeypatch.setattr(run, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(run, "SETTINGS", dict(run.SETTINGS, learning_starts=16, buffer_size=512))
    monkeypatch.setattr(run, "VALIDATION_GAMES", 2)
    with run.run_lock():
        assert run.train(128) == 128
    stem = os.path.join(model_dir, "checkpoint_128")
    path = stem + ".zip"
    with open(path, "rb") as file:
        initial = hashlib.file_digest(file, "sha256").hexdigest()
    model = DQN.load(path, device="cpu")
    assert model.policy.optimizer.state_dict()["state"]
    model.load_replay_buffer(stem + ".pkl")
    assert model.replay_buffer.size() == 128

    broken = os.path.join(model_dir, "checkpoint_256.zip")
    with open(broken, "wb") as file:
        file.write(b"interrupted")
    os.remove(os.path.join(results_dir, "validation", "checkpoint_128.csv"))
    with run.run_lock():
        assert run.train(256) == 256
    with open(path, "rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == initial
    with open(broken, "rb") as file:
        assert file.read() == b"interrupted"
    new, new_path = run.latest_model()
    new.load_replay_buffer(os.path.splitext(new_path)[0] + ".pkl")
    assert new.num_timesteps == 256 and new.replay_buffer.size() == 256
    scores = pd.read_csv(os.path.join(results_dir, "checkpoint_evaluation.csv"))
    assert scores.timesteps.tolist() == [128, 256]
    assert pd.read_csv(os.path.join(results_dir, "latest_checkpoint.csv")).actual_timesteps.iloc[0] == 256


def test_checkpoint_preference_uses_validation_wins_before_clearance():
    loss = dict(greedy_win=False, wins=0, greedy_pegs=76, avg_pegs_cleared=74,
                avg_shots_success=float("nan"), greedy_shots=10)
    sampled_win = dict(loss, wins=1, greedy_pegs=73, avg_shots_success=10)
    greedy_win = dict(loss, greedy_win=True, greedy_pegs=77, greedy_shots=9)
    assert run.validation_rank(greedy_win) > run.validation_rank(sampled_win) > run.validation_rank(loss)
