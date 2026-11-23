"""Check isolated rewards and demonstrated transitions without retraining old models."""
import numpy as np
from levels import dqn_eight_shot_experiment as run
from levels.peggle_env import PeggleEnv
from levels.config_rl import RLConfig
from gymnasium.utils.env_checker import check_env
import os
import torch
from stable_baselines3 import DQN


def test_bonus_only_changes_actual_success_reward():
    base = PeggleEnv(run.all_pegs_config())
    bonus = run.CompletionReward()
    actions = [(angle + 60) // 6 for angle in run.DEMONSTRATION_ANGLES]
    a, _ = base.reset(seed=50042)
    b, _ = bonus.reset(seed=50042)
    np.testing.assert_array_equal(a, b)
    try:
        for action in actions:
            a, reward_a, done_a, truncated_a, info_a = base.step(int(action))
            b, reward_b, done_b, truncated_b, info_b = bonus.step(int(action))
            np.testing.assert_array_equal(a, b)
            assert (done_a, truncated_a, info_a) == (done_b, truncated_b, info_b)
            assert reward_b - reward_a == (20 * info_a["shots_remaining"] if info_a["success"] else 0)
        assert done_a and info_a["pegs_cleared"] == 77 and info_a["shots_used"] == 8
        assert info_a["shots_remaining"] == 2 and len(a) == 78
        assert "shot_score" not in info_a and "free_balls_earned" not in info_a
    finally:
        base.close()
        bonus.close()


def test_demo_transitions_use_true_observations_and_mapping():
    transitions = run.collect_demonstration()
    assert len(transitions) == 8 and sum(int(t[4]) for t in transitions) == 1
    assert [int(t[1]) for t in transitions] == [9, 12, 5, 0, 1, 10, 3, 3]
    assert [int(t[3][:77].sum()) for t in transitions] == [8, 38, 51, 64, 68, 71, 75, 77]
    assert transitions[-1][2] == 148
    for i, (obs, action, reward, after, done) in enumerate(transitions):
        assert obs.shape == after.shape == (78,) and obs.dtype == after.dtype == np.float32
        assert 0 <= action < 21
        assert float(after[-1]) == float(np.float32((9 - i) / 10))
        if i < 7:
            np.testing.assert_array_equal(after, transitions[i + 1][0])


def test_validation_prioritizes_completion_then_winning_shots():
    base = dict(wins=30, average_winning_shots=10, greedy_win=True, greedy_shots=10, mean_pegs=75)
    faster = dict(base, average_winning_shots=8, greedy_shots=8)
    more = dict(base, wins=31, mean_pegs=74)
    assert run.rank(more) > run.rank(faster) > run.rank(base)


def test_all_peg_observation_actions_and_physics():
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


def test_fresh_training_runs_without_pretrained_files(tmp_path, monkeypatch):
    import play
    from levels import train_dqn
    original_load = DQN.load
    original_one = train_dqn.train_model_one
    original_two = train_dqn.train_model_two
    root = os.path.realpath(str(tmp_path))

    def load_generated(cls, path, *args, **kwargs):
        resolved = os.path.realpath(str(path))
        assert os.path.commonpath([root, resolved]) == root
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(DQN, "load", classmethod(load_generated))
    monkeypatch.setattr(train_dqn, "train_model_one", lambda folder: original_one(folder, steps=32))
    monkeypatch.setattr(train_dqn, "train_model_two", lambda folder: original_two(
        folder, base_steps=1104, additional_steps=1104, cloning_updates=8,
        validation_games=2, interval=552))
    orange, config, first_run = play.train_fresh("1", root)
    assert orange.observation_space.shape == (155,) and orange.action_space.n == 21
    again, _, second_run = play.train_fresh("1", root)
    assert first_run != second_run and os.path.isdir(first_run) and os.path.isdir(second_run)
    assert orange.num_timesteps == again.num_timesteps == 32
    model, config, all_run = play.train_fresh("2", root)
    assert model.observation_space.shape == (78,) and model.action_space.n == 21
    assert not config.free_balls and model.num_timesteps > 1104
    last = DQN.load(os.path.join(all_run, "demonstration_stage", "checkpoint_2208.zip"), device="cpu")
    assert last._n_updates > 0 and last.policy.optimizer.state_dict()["state"]
    # The selected checkpoint reloads and plays using its actual learned predictions.
    restored = DQN.load(os.path.join(all_run, "best_model.zip"), device="cpu")
    actual, _ = run.play(restored)
    assert 0 <= actual["pegs_cleared"] <= 77 and actual["shots_used"] <= 10
    summary = play.evaluate_new_model(restored, config, all_run)
    assert summary["games"] == 100
    assert os.path.isfile(os.path.join(all_run, "evaluation_summary.csv"))


def test_saved_final_policy_uses_q_values():
    if not os.path.isfile(run.FINAL_DQN_PATH):
        import pytest
        pytest.skip("Optional historical checkpoint is absent in a source-only submission.")
    model = run.load_policy(train_if_missing=False)
    game, trace = run.play(model)
    assert game["success"] and game["shots_used"] == 8 and game["balls_remaining"] == 2
    env = PeggleEnv(run.all_pegs_config())
    obs, _ = env.reset(seed=50042)
    try:
        for shot in trace:
            with torch.no_grad():
                predicted = int(model.q_net(torch.as_tensor(obs[None], device=model.device)).argmax(dim=1))
            assert predicted == shot["action"]
            obs, _, _, _, _ = env.step(predicted)
    finally:
        env.close()


def test_pretrained_menu_routes_both_models_without_training(monkeypatch, capsys):
    import play
    marker = object()
    def no_training(*args, **kwargs):
        raise AssertionError("Pretrained playback started training")
    monkeypatch.setattr(play, "train_fresh", no_training)
    for choice in ("1", "2"):
        answers = iter([choice, "1"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
        config = RLConfig(observation="compact") if choice == "1" else run.all_pegs_config()
        monkeypatch.setattr(play, "load_game_mode", lambda selected: (marker, config))
        shown = []
        monkeypatch.setattr(play, "demonstrate", lambda model, config, label: shown.append((model, label)))
        play.main()
        assert shown == [(marker, "DQN Model " + choice)]
    output = capsys.readouterr().out
    assert "Use existing pretrained model" in output and "Train a new model from scratch" in output
    assert "Back to model selection" in output


def test_missing_and_incompatible_pretrained_checkpoints(monkeypatch):
    import play
    import pytest
    monkeypatch.setattr(play.os.path, "isfile", lambda path: False)
    for choice in ("1", "2"):
        with pytest.raises(FileNotFoundError, match="Pretrained checkpoint missing"):
            play.load_game_mode(choice)
    monkeypatch.setattr(play.os.path, "isfile", lambda path: True)
    fake = type("Policy", (), {"observation_space": type("Space", (), {"shape": (1,)})(),
                               "action_space": type("Space", (), {"n": 21})()})()
    monkeypatch.setattr(play.DQN, "load", lambda *args, **kwargs: fake)
    for choice in ("1", "2"):
        with pytest.raises(ValueError, match="Incompatible checkpoint"):
            play.load_game_mode(choice)


def test_back_menu_and_fresh_model_demonstration(monkeypatch, capsys):
    import play
    answers = iter(["1", "3", "2", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    model = object()
    events = []
    monkeypatch.setattr(play, "train_fresh", lambda choice: (model, run.all_pegs_config(), "new_run"))
    monkeypatch.setattr(play, "evaluate_new_model", lambda m, c, folder: events.append(("evaluate", m, folder)))
    monkeypatch.setattr(play, "demonstrate", lambda m, c, label: events.append(("play", m, label)))
    play.main()
    assert events == [("evaluate", model, "new_run"), ("play", model, "DQN Model 2")]
    assert capsys.readouterr().out.count("PEGGLE REINFORCEMENT LEARNING") == 2
