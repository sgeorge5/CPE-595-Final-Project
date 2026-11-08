import sys

import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from levels.config_rl import (
    CONFIG, FIXED_BOARD, ROOT, RLConfig, SMALL_CONFIG, load_level_coordinates
)
from levels.peggle_env import PeggleEnv
from levels.visualize_data import original_level_data, level_inventory, plot_board, plot_gameplay, record_gameplay
from levels.train_dqn import train_agent, load_training_model, checkpoint_key
from levels.evaluate import evaluate_agents, angle_diagnostic, greedy_diagnostic, run_policy


def test_reset_and_preprocessing():
    env = PeggleEnv()
    initial, info = env.reset(seed=42)
    assert initial.shape == (309,) and initial.dtype == np.float32
    assert env.observation_space.contains(initial)
    assert info["orange_initial"] == info["orange_remaining"] == 25
    assert info["shots_remaining"] == 10 and env.action_space.n == 21
    features = initial[:-1].reshape(77, 4)
    np.testing.assert_allclose(features[:, :2], CONFIG.coordinates / [1200, 900])
    assert features[:, 2].sum() == 25 and features[:, 3].sum() == 0
    assert initial[-1] == 1.0
    env.step(10)
    reset, reset_info = env.reset(seed=42)
    np.testing.assert_array_equal(initial, reset)
    assert reset_info == info


def test_full_shot_and_reward_accounting():
    for action in range(21):
        env = PeggleEnv()
        initial, _ = env.reset(seed=123)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert obs[-1] == np.float32(0.9)
        assert info["shots_used"] == 1 and info["simulation_frames"] > 1
        assert not truncated
        assert not info["shot_timed_out"]
        expected = 10 * info["new_orange_cleared"] - 1 + (25 if info["success"] else 0)
        assert reward == expected
        assert terminated == info["success"]
        np.testing.assert_array_equal(obs[:-1].reshape(77, 4)[:, :3], initial[:-1].reshape(77, 4)[:, :3])


def test_physical_hit_not_rewarded_twice():
    env = PeggleEnv(SMALL_CONFIG)
    env.reset(seed=42)
    _, first_reward, _, _, first = env.step(10)
    assert first["new_orange_cleared"] >= 1 and first_reward >= 9
    _, second_reward, _, _, second = env.step(10)
    assert second["new_orange_cleared"] == 0 and second_reward == -1
    assert second["orange_cleared"] == first["orange_cleared"]


def test_success_bonus_and_terminal_guard():
    env = PeggleEnv(SMALL_CONFIG)
    env.reset(seed=42)
    env.pegs[0].isHit = env.pegs[2].isHit = True
    _, reward, terminated, truncated, info = env.step(10)
    assert terminated and not truncated and info["success"]
    assert info["new_orange_cleared"] == 1 and reward == 34
    assert any(not p.isHit and p.color == "blue" for p in env.pegs)
    with pytest.raises(RuntimeError):
        env.step(10)


def test_budget_exhaustion():
    env = PeggleEnv(RLConfig(max_shots=1))
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and not info["success"]
    assert obs[-1] == 0 and info["terminal_reason"] == "shot_budget"
    assert reward == 10 * info["new_orange_cleared"] - 1


def test_invalid_actions_and_step_before_reset():
    env = PeggleEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    for action in (-1, 21, 0.5, [10]):
        with pytest.raises(ValueError):
            env.step(action)


def test_deterministic_sequences_and_isolated_instances():
    left, right = PeggleEnv(), PeggleEnv()
    left.reset(seed=1)
    right.reset(seed=1)
    for action in (3, 10, 17, 0, 7):
        a, b = left.step(action), right.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        if a[2]:
            break
    before = right._observation().copy()
    left.reset()
    np.testing.assert_array_equal(before, right._observation())


def test_gymnasium_and_sb3_contract():
    gym_check_env(PeggleEnv(), skip_render_check=True)
    sb3_check_env(PeggleEnv(), warn=True)


def test_physics_settings_stay_separate():
    normal_env = PeggleEnv(RLConfig())
    stronger_gravity = RLConfig()
    stronger_gravity.gravity = 0.09
    stronger_env = PeggleEnv(stronger_gravity)

    normal_ball = normal_env.Ball(600, 36)
    stronger_ball = stronger_env.Ball(600, 36)
    normal_ball.update()
    stronger_ball.update()
    assert normal_ball.vel.y == pytest.approx(0.045)
    assert stronger_ball.vel.y == pytest.approx(0.09)


def test_upstream_data_and_no_graphical_side_effects():
    data = original_level_data()
    assert len(data) == 77 and len(level_inventory()) == 10
    assert not data.color_stored_in_file.any()
    assert not data.initial_is_hit.any()
    assert "local.config" not in sys.modules
    assert "pygame" not in sys.modules
    assert not (ROOT / "user.cfg").exists()


def test_dqn_updates_weights_and_save_load(tmp_path):
    torch.set_num_threads(1)
    env = PeggleEnv()
    model = DQN("MlpPolicy", env, seed=42, device="cpu", learning_starts=16,
                buffer_size=128, batch_size=16, train_freq=4,
                policy_kwargs={"net_arch": [64, 64]})
    before = {k: v.clone() for k, v in model.q_net.state_dict().items()}
    model.learn(total_timesteps=64)
    assert model._n_updates > 0
    assert any(not torch.equal(before[k], v) for k, v in model.q_net.state_dict().items())
    model.save(tmp_path / "compatibility.zip")
    restored = DQN.load(tmp_path / "compatibility.zip", device="cpu")
    assert restored.observation_space.shape == (309,)
    assert restored.action_space.n == 21
    obs, _ = env.reset(seed=42)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
    env.close()


def test_training_and_evaluation_artifacts(tmp_path):
    path = tmp_path / "dqn_level1_random.zip"
    model, training_log = train_agent(32, model_path=path, results_dir=tmp_path, verbose=0)
    assert path.exists() and len(training_log) > 0
    assert model.num_timesteps == 32
    summary, episodes = evaluate_agents(path, episodes=3, results_dir=tmp_path)
    assert len(episodes) == 6 and list(summary.agent) == ["DQN", "Random"]
    saved = pd.read_csv(tmp_path / "evaluation_episodes.csv")
    assert saved.orange_initial.eq(25).all()
    assert saved.orange_cleared.between(0, 25).all()
    assert saved.shots_used.between(1, 10).all()
    for row in summary.itertuples():
        actual = saved[saved.agent == row.agent]
        assert row.success_rate_pct == pytest.approx(actual.success.mean() * 100)
        assert row.orange_cleared_pct == pytest.approx(actual.orange_cleared.mean() / 25 * 100)
        assert row.avg_orange_remaining == pytest.approx(actual.orange_remaining.mean())
        successful = actual[actual.success]
        assert (np.isnan(row.avg_shots_success) if successful.empty else
                row.avg_shots_success == pytest.approx(successful.shots_used.mean()))
    # Evaluation also accepts a trained Python object, without reading a file.
    in_memory, _ = evaluate_agents(tmp_path / "missing.zip", episodes=3,
                                    results_dir=tmp_path, model=model)
    pd.testing.assert_frame_equal(summary, in_memory)
    with pytest.raises(ValueError, match="Model input size"):
        evaluate_agents(model=model, config=SMALL_CONFIG, episodes=1, results_dir=tmp_path)
    pairs = pd.read_csv(tmp_path / "evaluation_arrangements.csv").pivot(
        index="episode", columns="agent", values="orange_peg_ids")
    assert pairs.DQN.equals(pairs.Random)
    assert pairs.DQN.nunique() == 3
    assert not list(tmp_path.glob("*.json"))
    assert not (tmp_path / "training.monitor.csv").exists()
    for name in ("training_rewards.png", "dqn_vs_random.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_level1_coordinates_match_original_game_translation():
    raw = load_level_coordinates(center=False)
    expected = raw + [600, 450] - (raw.min(axis=0) + raw.max(axis=0)) / 2
    assert raw.shape == (77, 2)
    np.testing.assert_array_equal(CONFIG.coordinates, expected)
    np.testing.assert_array_equal(CONFIG.coordinates[0], [99.5, 394])
    env = PeggleEnv()
    for seed in (42, 43, 44):
        env.reset(seed=seed)
        np.testing.assert_array_equal([(p.pos.x, p.pos.y) for p in env.pegs], expected)


def test_random_arrangements_and_seeded_sequence():
    left, right = PeggleEnv(), PeggleEnv()
    arrangements = []
    for episode in range(6):
        seed = 42 if episode == 0 else None
        a, _ = left.reset(seed=seed)
        b, _ = right.reset(seed=seed)
        np.testing.assert_array_equal(a, b)
        colors = a[:-1].reshape(77, 4)[:, 2]
        assert colors.sum() == 25
        assert len(np.flatnonzero(colors)) == 25
        arrangements.append(tuple(colors))
    assert len(set(arrangements)) == 6
    other, _ = left.reset(seed=43)
    assert tuple(other[:-1].reshape(77, 4)[:, 2]) != arrangements[0]


def test_colors_stay_fixed_within_game_and_reset_restores_pegs():
    env = PeggleEnv()
    initial, _ = env.reset(seed=42)
    colors = [peg.color for peg in env.pegs]
    total_hits = 0
    for action in (10, 0, 20):
        obs, reward, done, _, info = env.step(action)
        assert [peg.color for peg in env.pegs] == colors
        np.testing.assert_array_equal(obs[:-1].reshape(77, 4)[:, 2],
                                      initial[:-1].reshape(77, 4)[:, 2])
        total_hits += info["new_orange_cleared"]
        assert total_hits == info["orange_cleared"] <= 25
        assert reward == 10 * info["new_orange_cleared"] - 1 + (25 if done and info["success"] else 0)
        if done:
            break
    assert any(peg.isHit for peg in env.pegs)
    env.reset()
    assert all(not peg.isHit and peg.isVisible for peg in env.pegs)
    assert env._info()["orange_remaining"] == 25


def test_full_level_success_bonus_and_no_double_reward():
    # First find the first orange contact, then leave just that target uncleared.
    probe = PeggleEnv(record_trajectory=True)
    probe.reset(seed=42)
    probe.step(10)
    target = None
    for frame in probe.trajectory:
        for i, peg in enumerate(probe.pegs):
            if peg.isOrange and frame["is_hit"][i]:
                target = i
                break
        if target is not None:
            break
    assert target is not None
    env = PeggleEnv()
    env.reset(seed=42)
    for i, peg in enumerate(env.pegs):
        if peg.isOrange and i != target:
            peg.isHit = True
    _, reward, terminated, truncated, info = env.step(10)
    assert info["new_orange_cleared"] == 1
    assert info["orange_cleared"] == 25 and info["success"]
    assert reward == 34 and terminated and not truncated
    with pytest.raises(RuntimeError):
        env.step(10)


def test_optional_small_board_keeps_original_task():
    env = PeggleEnv(SMALL_CONFIG)
    first, info = env.reset(seed=42)
    second, _ = env.reset(seed=99)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (41,) and info["orange_initial"] == 3
    assert [(p.pos.x, p.pos.y, p.color) for p in env.pegs] == list(FIXED_BOARD)
    _, reward, done, _, info = env.step(9)
    assert done and info["success"] and reward == 54


def test_trajectory_and_visualizations_use_current_states():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    plain, recorded = PeggleEnv(), PeggleEnv(record_trajectory=True)
    plain.reset(seed=42)
    recorded.reset(seed=42)
    board = recorded.board_records()
    fig = plot_board(board)
    orange, _, blue, _ = fig.axes[0].collections[:4]
    assert len(orange.get_offsets()) == 25 and len(blue.get_offsets()) == 52
    np.testing.assert_allclose(orange.get_offsets(),
                               [(p["x"], p["y"]) for p in board if p["color"] == "orange"])
    plt.close(fig)
    a, b = plain.step(10), recorded.step(10)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:] and plain.trajectory == []
    assert recorded.trajectory[-1]["is_visible"] == [p.isVisible for p in recorded.pegs]
    snapshot = recorded.trajectory[100]
    fig = plot_gameplay(board, recorded.trajectory, 100, 0, 1)
    circles = fig.axes[0].patches
    visible = [p for i, p in enumerate(board) if snapshot["is_visible"][i]]
    for circle, peg in zip(circles, visible):
        expected_color = "#ed8b22" if peg["color"] == "orange" else "#2786c4"
        assert circle.get_facecolor() == to_rgba(expected_color)
        assert circle.center == (peg["x"], peg["y"])
    plt.close(fig)
    recorded.reset()
    assert recorded.trajectory == []


def test_only_resolved_contact_is_marked_hit():
    # One swept segment detects two pegs, but the first bounce prevents the second hit.
    config = RLConfig(board="small")
    config.gravity = 0
    config.launch_force = config.max_ball_velocity = 100
    config.max_frames_per_shot = 1
    env = PeggleEnv(config)
    env.reset(seed=42)
    env.pegs = env.pegs[:2]
    env.pegs[0].pos = env.Vector(600, 100)
    env.pegs[1].pos = env.Vector(600, 130)
    env._simulate_shot(0)
    assert env.pegs[0].isHit
    assert not env.pegs[1].isHit


def test_compact_observation_preserves_changing_state():
    env = PeggleEnv(RLConfig(observation="compact"))
    obs, _ = env.reset(seed=42)
    assert obs.shape == (155,) and obs.dtype == np.float32
    for action in (10, 4, 16):
        full = env._observation("full")
        np.testing.assert_array_equal(obs[:-1], full[:-1].reshape(77, 4)[:, 2:].flatten())
        assert obs[-1] == full[-1]
        obs, _, done, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if done:
            break
    gym_check_env(PeggleEnv(RLConfig(observation="compact")), skip_render_check=True)


def test_empty_shot_penalty_and_full_reward_formula():
    for penalty in (0, 5):
        cfg = RLConfig(board="small", empty_shot_penalty=penalty)
        env = PeggleEnv(cfg)
        env.reset(seed=42)
        _, reward, _, _, info = env.step(0)
        assert info["new_orange_cleared"] == 0 and reward == -1 - penalty
        env.reset(seed=42)
        _, reward, done, _, info = env.step(9)
        assert done and info["success"] and reward == 54  # No empty-shot penalty on success.


def test_checkpoint_resume_retains_weights_optimizer_replay_and_history(tmp_path):
    from stable_baselines3.common.monitor import Monitor
    from levels.config_rl import DQN_SETTINGS

    cfg = RLConfig(observation="compact")
    settings = dict(DQN_SETTINGS)
    settings.update(learning_starts=16, learning_rate=0.0003)
    path = tmp_path / "last_model.zip"
    model, history = train_agent(64, config=cfg, dqn_settings=settings,
        model_path=path, results_dir=tmp_path, checkpoint_interval=32,
        validation_episodes=1, verbose=0)
    old_weights = {k: v.clone() for k, v in model.q_net.state_dict().items()}
    restored = load_training_model(path, Monitor(PeggleEnv(cfg)))
    assert restored.num_timesteps == 64 and restored.learning_rate == 0.0003
    assert restored.replay_buffer.size() == model.replay_buffer.size() == 64
    for key, value in restored.q_net.state_dict().items():
        assert torch.equal(value, old_weights[key])
    assert restored.policy.optimizer.state_dict()["state"]  # Adam's learned moments restored.
    restored.get_env().close()
    saved_checkpoint = (tmp_path / "checkpoint_64.zip").read_bytes()
    continued, full_history = train_agent(64, config=cfg, resume_from=path,
        model_path=path, results_dir=tmp_path, checkpoint_interval=32,
        validation_episodes=1, verbose=0)
    assert continued.num_timesteps == 128 and continued._n_updates > model._n_updates
    assert continued.replay_buffer.size() == 128
    assert (tmp_path / "checkpoint_64.zip").read_bytes() == saved_checkpoint
    pd.testing.assert_frame_equal(history, full_history.iloc[:len(history)].reset_index(drop=True))
    assert full_history.timesteps.is_monotonic_increasing
    scores = pd.read_csv(tmp_path / "checkpoint_evaluation.csv")
    selected = scores[scores.is_best].iloc[0]
    assert checkpoint_key(selected) == max(checkpoint_key(row) for _, row in scores.iterrows())
    best = DQN.load(tmp_path / "best_model.zip", device="cpu")
    checkpoint = DQN.load(selected.checkpoint, device="cpu")
    for key, value in best.q_net.state_dict().items():
        assert torch.equal(value, checkpoint.q_net.state_dict()[key])
    with pytest.raises(FileExistsError):
        train_agent(32, config=cfg, model_path=path, results_dir=tmp_path)


def test_all_baselines_share_boards_and_current_reward_rules(tmp_path):
    compact = RLConfig(observation="compact", empty_shot_penalty=5)
    model = DQN("MlpPolicy", PeggleEnv(compact), seed=42, device="cpu", buffer_size=128)
    old = DQN("MlpPolicy", PeggleEnv(), seed=42, device="cpu", buffer_size=128)
    old.save(tmp_path / "old.zip")
    summary, episodes = evaluate_agents(model=model, episodes=2, config=compact,
        results_dir=tmp_path, include_baselines=True, original_model_path=tmp_path / "old.zip")
    assert set(summary.agent) == {"DQN", "Random", "Straight down", "Original DQN"}
    arrangements = pd.read_csv(tmp_path / "evaluation_arrangements.csv").pivot(
        index="episode", columns="agent", values="orange_peg_ids")
    for agent in arrangements:
        assert arrangements[agent].equals(arrangements.DQN)
    shots = pd.read_csv(tmp_path / "evaluation_shots.csv")
    expected = 10 * shots.new_orange_cleared - 1 - 5 * shots.new_orange_cleared.eq(0)
    expected += 25 * shots.orange_remaining.eq(0)
    np.testing.assert_array_equal(shots.reward, expected)
    assert shots[shots.agent == "Straight down"].angle_degrees.eq(0).all()
    for row in episodes.itertuples():
        actual = shots[(shots.agent == row.agent) & (shots.episode == row.episode)]
        assert row.episode_reward == actual.reward.sum()


def test_angle_diagnostic_and_greedy_are_separate_measured_probes(tmp_path):
    data = angle_diagnostic(seeds=(42, 43), results_dir=tmp_path)
    assert len(data) == 42
    for seed, rows in data.groupby("seed"):
        assert rows.x_at_frame_10.nunique() == 21
        assert rows.orange_cleared.nunique() > 1
        np.testing.assert_array_equal(rows.reward, 10 * rows.orange_cleared - 1)
    greedy = greedy_diagnostic(seeds=(42,), config=RLConfig(max_shots=1), results_dir=tmp_path)
    assert greedy.iloc[0].orange_cleared == data[data.seed == 42].orange_cleared.max()
    assert greedy.iloc[0].shots_used == 1


def test_upstream_free_flight_and_isolated_collision_equations():
    # Independent arithmetic checks against the bundled Ball.update integration order.
    env = PeggleEnv()
    for angle in (-30, 30):
        ball = env.Ball(600, 36)
        ball.isAlive = True
        velocity = 5 * np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
        ball.applyForce(env.Vector(*velocity))
        position = np.array([600.0, 36.0])
        for frame in range(15):
            velocity += [0, 0.045]
            position += velocity
            ball.update(1)
            np.testing.assert_allclose([ball.pos.x, ball.pos.y], position, atol=1e-10)
    env.reset(seed=42)
    peg = env.pegs[0]
    peg.pos = env.Vector(600, 300)
    ball = env.Ball(600, 260)
    ball.prevPos, ball.pos, ball.vel = env.Vector(600, 260), env.Vector(600, 280), env.Vector(0, 5)
    assert env.collision.isBallTouchingPeg(ball, peg, 1)
    ball = env.collision.resolveCollision(ball, [peg], 1)
    assert ball.pos.x == pytest.approx(600) and ball.pos.y == pytest.approx(263)
    assert ball.vel.y == pytest.approx(-4.5)  # Original restitution 0.90 at this speed.


def test_recorded_gameplay_matches_policy_evaluation():
    cfg = RLConfig(observation="compact", empty_shot_penalty=5)
    model = DQN("MlpPolicy", PeggleEnv(cfg), seed=42, device="cpu", buffer_size=128)
    _, episodes, measured_shots, _ = run_policy(model, episodes=1, seed=42, config=cfg)
    trajectories, shown_shots = record_gameplay(model, 42, cfg)
    for column in ("angle_degrees", "new_orange_cleared", "orange_remaining", "shots_remaining", "reward"):
        np.testing.assert_array_equal(shown_shots[column], measured_shots[column])
    assert shown_shots.iloc[-1].orange_remaining == episodes.iloc[0].orange_remaining
    for shot in trajectories:
        assert len(shot["trajectory"]) > 1
        final = shot["trajectory"][-1]
        assert not final["ball_alive"]
        assert len(final["is_hit"]) == 77


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
