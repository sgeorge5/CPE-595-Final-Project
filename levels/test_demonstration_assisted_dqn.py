import numpy as np
from levels import demonstation_assisted_dqn as run
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
    original_standard = train_dqn.train_standard_all_pegs
    original_two = train_dqn.train_model_two
    root = os.path.realpath(str(tmp_path))

    def load_generated(cls, path, *args, **kwargs):
        resolved = os.path.realpath(str(path))
        assert os.path.commonpath([root, resolved]) == root
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(DQN, "load", classmethod(load_generated))
    monkeypatch.setattr(train_dqn, "train_model_one", lambda folder, **kwargs: original_one(folder, steps=208, **kwargs))
    monkeypatch.setattr(train_dqn, "train_standard_all_pegs", lambda folder, **kwargs: original_standard(
        folder, steps=1104, validation_games=2, **kwargs))
    monkeypatch.setattr(train_dqn, "train_model_two", lambda folder, **kwargs: original_two(
        folder, base_steps=1104, additional_steps=1104, cloning_updates=8,
        validation_games=2, interval=552, **kwargs))
    orange, config, first_run = play.train_fresh("1", root)
    assert orange.observation_space.shape == (155,) and orange.action_space.n == 21
    again, _, second_run = play.train_fresh("1", root)
    assert first_run != second_run and os.path.isdir(first_run) and os.path.isdir(second_run)
    assert orange.num_timesteps == again.num_timesteps == 208
    assert orange._n_updates > 0 and orange.policy.optimizer.state_dict()["state"]
    standard, config, standard_run = play.train_fresh("2", root)
    assert standard.observation_space.shape == (78,) and standard.action_space.n == 21
    assert not config.free_balls and standard.num_timesteps == 1104
    assert standard._n_updates > 0 and standard.policy.optimizer.state_dict()["state"]
    assert not os.path.isdir(os.path.join(standard_run, "demonstration_stage"))
    model, config, all_run = play.train_fresh("3", root)
    assert model.observation_space.shape == (78,) and model.action_space.n == 21
    assert not config.free_balls and model.num_timesteps > 1104
    last = DQN.load(os.path.join(all_run, "demonstration_stage", "checkpoint_2208.zip"), device="cpu")
    assert last._n_updates > 0 and last.policy.optimizer.state_dict()["state"]
    restored = DQN.load(os.path.join(all_run, "best_model.zip"), device="cpu")
    actual, _ = run.play(restored)
    assert 0 <= actual["pegs_cleared"] <= 77 and actual["shots_used"] <= 10
    summary = play.evaluate_new_model(restored, config, all_run, assisted=True)
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


def test_pretrained_menu_routes_three_models_without_training(monkeypatch, capsys):
    import play
    marker = object()
    def no_training(*args, **kwargs):
        raise AssertionError("Pretrained playback started training")
    monkeypatch.setattr(play, "train_fresh", no_training)
    for choice in ("1", "2", "3"):
        answers = iter([choice, "1", "2"] + (["1"] if choice == "1" else []))
        monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
        config = RLConfig(observation="compact") if choice == "1" else run.all_pegs_config()
        monkeypatch.setattr(play, "load_game_mode", lambda selected: (marker, config))
        shown = []
        monkeypatch.setattr(play, "play_policy", lambda model, config, label, **kwargs: shown.append((model, label)))
        play.main()
        assert shown == [(marker, play.MODEL_LABELS[choice])]
    output = capsys.readouterr().out
    assert "Use existing pretrained model" in output and "Train a new model from scratch" in output
    assert "Back to model selection" in output


def test_missing_and_incompatible_pretrained_checkpoints(monkeypatch):
    import play
    import pytest
    monkeypatch.setattr(play.os.path, "isfile", lambda path: False)
    for choice in ("1", "2", "3"):
        with pytest.raises(FileNotFoundError, match="Pretrained checkpoint missing"):
            play.load_game_mode(choice)
    monkeypatch.setattr(play.os.path, "isfile", lambda path: True)
    fake = type("Policy", (), {"observation_space": type("Space", (), {"shape": (1,)})(),
                               "action_space": type("Space", (), {"n": 21})()})()
    monkeypatch.setattr(play.DQN, "load", lambda *args, **kwargs: fake)
    for choice in ("1", "2", "3"):
        with pytest.raises(ValueError, match="Incompatible checkpoint"):
            play.load_game_mode(choice)


def test_back_menu_and_fresh_model_demonstration(monkeypatch, capsys):
    import play
    answers = iter(["1", "3", "2", "2", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    model = object()
    events = []
    monkeypatch.setattr(play, "train_fresh", lambda choice, **kwargs: (model, run.all_pegs_config(), "new_run"))
    monkeypatch.setattr(play, "evaluate_new_model", lambda m, c, folder, **kwargs: events.append(("evaluate", m, folder)))
    monkeypatch.setattr(play, "play_policy", lambda m, c, label, **kwargs: events.append(("play", m, label)))
    play.main()
    assert events == [("evaluate", model, "new_run"), ("play", model, play.MODEL_LABELS["2"])]
    assert capsys.readouterr().out.count("PEGGLE REINFORCEMENT LEARNING") == 2


class TestViewer:
    """Exercise real environment recording/callbacks without a desktop window."""
    __test__ = False

    def __init__(self):
        self.active = True
        self.shots, self.stages, self.panels = [], [], []

    def animate(self, shot, config, stats):
        final = shot["trajectory"][-1]
        assert not final["ball_alive"] and len(final["is_hit"]) == 77
        if config.goal == "all":
            assert sum(final["is_hit"]) == shot["info"]["pegs_cleared"]
        self.shots.append(shot)
        self.panels.append(dict(stats))

    def stage(self, name, detail=""):
        self.stages.append(name)

    def panel(self, stats):
        self.panels.append(dict(stats))

    def finish(self):
        pass

    def close(self):
        self.active = False


def test_all_twelve_menu_combinations_and_navigation(monkeypatch, capsys):
    import play
    for choice in ("1", "2", "3"):
        for source in ("1", "2"):
            for mode in ("1", "2"):
                answers = iter([choice, source, mode] + (["1"] if choice == "1" else []))
                monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
                viewer, model, events = TestViewer(), object(), []
                config = RLConfig(observation="compact") if choice == "1" else run.all_pegs_config()
                monkeypatch.setattr(play, "LiveGame", lambda: viewer)
                monkeypatch.setattr(play, "load_game_mode", lambda c: (events.append(("load", c)) or (model, config)))
                monkeypatch.setattr(play, "train_fresh", lambda c, **kw: (
                    events.append(("train", c, kw["viewer"])) or (model, config, "isolated")))
                monkeypatch.setattr(play, "evaluate_new_model", lambda m, c, f, **kw:
                    events.append(("evaluate", kw["assisted"])))
                monkeypatch.setattr(play, "play_policy", lambda m, c, label, **kw:
                    events.append(("play", m, c, kw["viewer"])))
                play.main()
                selected_viewer = viewer if mode == "1" else None
                assert events[-1] == ("play", model, config, selected_viewer)
                if source == "1":
                    assert events[:-1] == [("load", choice)]
                else:
                    assert events[:-1] == [("train", choice, selected_viewer), ("evaluate", choice == "3")]
    # Invalid values at all levels; mode Back returns to source, source Back to models.
    answers = iter(["bad", "1", "bad", "1", "bad", "3", "3", "2", "1", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    play.main()
    output = capsys.readouterr().out
    assert output.count("Please enter 1, 2, or 3.") == 3
    assert "SELECT TRAINING MODE" in output and "SELECT PLAYBACK MODE" in output


def test_missing_graphics_can_fall_back_to_headless(monkeypatch, capsys):
    import play
    def unavailable():
        raise RuntimeError("Live graphics are unavailable. Choose headless execution.")
    answers = iter(["2", "1", "1", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(play, "LiveGame", unavailable)
    monkeypatch.setattr(play, "load_game_mode", lambda choice: (object(), run.all_pegs_config()))
    selected = []
    monkeypatch.setattr(play, "play_policy", lambda *args, **kwargs: selected.append(kwargs["viewer"]))
    play.main()
    assert selected == [None] and "Choose headless" in capsys.readouterr().out


def test_pretrained_visual_and_headless_match_without_files(monkeypatch):
    import play
    import webbrowser
    from levels.visualize_data import play_policy, plt
    figures = plt.get_fignums()
    def no_browser(*args, **kwargs):
        raise AssertionError("Playback opened a browser")
    monkeypatch.setattr(webbrowser, "open", no_browser)
    for choice in ("1", "2", "3"):
        model, config = play.load_game_mode(choice)
        original_step = PeggleEnv.step
        def headless_step(env, action):
            assert not env.record_trajectory
            result = original_step(env, action)
            assert env.trajectory == []
            return result
        with monkeypatch.context() as patch:
            patch.setattr(PeggleEnv, "step", headless_step)
            plain = play_policy(model, config, play.MODEL_LABELS[choice])
        viewer = TestViewer()
        shown = play_policy(model, config, play.MODEL_LABELS[choice],
                            seed=int(plain.iloc[0].seed), viewer=viewer)
        import pandas as pd
        pd.testing.assert_frame_equal(plain, shown)
        assert len(viewer.shots) == len(plain)
        if choice in ("2", "3"):
            assert plain.seed.eq(50042).all()
            assert plain.iloc[-1].pegs_cleared == 77 and plain.iloc[-1].success
            assert len(plain) == (10 if choice == "2" else 8)
        assert plt.get_fignums() == figures


def test_orange_random_arrangements_print_seed_and_keep_actions_greedy(monkeypatch, capsys):
    import play
    import pandas as pd
    from levels import visualize_data as visual
    model, config = play.load_game_mode("1")
    seeds = iter([50042, 50045])
    # Control only playback's new entropy generator, so this check is reproducible.
    rng = type("Seeds", (), {"integers": lambda self, *args: next(seeds)})()
    monkeypatch.setattr(visual.np.random, "default_rng", lambda: rng)
    first_view, second_view = TestViewer(), TestViewer()
    first = visual.play_policy(model, config, "Orange DQN", viewer=first_view)
    second = visual.play_policy(model, config, "Orange DQN", viewer=second_view)
    assert first.seed.eq(50042).all() and second.seed.eq(50045).all()
    first_colors = [p["color"] for p in first_view.shots[0]["board"]]
    second_colors = [p["color"] for p in second_view.shots[0]["board"]]
    assert first_colors != second_colors
    assert first_colors.count("orange") == second_colors.count("orange") == 25
    repeated = visual.play_policy(model, config, "Orange DQN", seed=50045)
    pd.testing.assert_frame_equal(second, repeated)
    env = PeggleEnv(config)
    obs, _ = env.reset(seed=50045)
    for row in second.itertuples():
        with torch.no_grad():
            action = int(model.q_net(torch.as_tensor(obs[None], device=model.device)).argmax(dim=1))
        assert action == row.action
        obs, _, _, _, _ = env.step(action)
    env.close()
    output = capsys.readouterr().out
    assert "Game seed: 50042" in output and "Game seed: 50045" in output


def test_orange_presentation_seed_is_an_actual_historical_win():
    import play
    import pandas as pd
    from levels.visualize_data import play_policy
    model, config = play.load_game_mode("1")
    assert play.ORANGE_PRESENTATION_SEED == 50119
    path = "results/level1_improved/evaluation_episodes.csv"
    for seed, expected_shots, episode in ((50119, 7, 78), (50045, 8, 4)):
        rows = play_policy(model, config, "Orange presentation", seed=seed)
        viewer = TestViewer()
        arrangement = "Selected demonstration case" if seed == 50119 else "Custom seed arrangement"
        shown = play_policy(model, config, "Orange presentation", seed=seed,
                            viewer=viewer, arrangement_label=arrangement)
        pd.testing.assert_frame_equal(rows, shown)
        assert all(panel["Arrangement"] == arrangement for panel in viewer.panels)
        final = rows.iloc[-1]
        assert final.success and final.orange_cleared == 25 and final.shots_used == expected_shots
        if os.path.isfile(path):
            history = pd.read_csv(path)
            game = history[(history.agent == "DQN") & (history.seed == seed)].iloc[0]
            assert game.episode == episode and game.success and game.orange_cleared == 25 and game.shots_used == expected_shots


def test_playback_seed_options_only_apply_to_orange_model(monkeypatch, capsys):
    import play
    selected = []
    monkeypatch.setattr(play, "load_game_mode", lambda choice: (object(), object()))
    monkeypatch.setattr(play, "play_policy", lambda *args, **kwargs: selected.append(kwargs["seed"]))
    for choice in ("1", "2", "3"):
        for seed, winning in ((None, False), (1234, False), (None, True)):
            answers = iter([choice, "1", "2"] + (["1"] if choice == "1" and seed is None and not winning else []))
            monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
            play.main(seed=seed, winning_arrangement=winning)
            expected = (play.ORANGE_PRESENTATION_SEED if winning else seed) if choice == "1" else None
            assert selected[-1] == expected
    output = capsys.readouterr().out
    assert output.count("Previously verified winning arrangement:") == 1
    assert "selected demonstration case, not a randomly generated test game" in output


def test_orange_arrangement_menu_choices_and_invalid_input(monkeypatch, capsys):
    import play
    for answers, expected in ((["1"], ("1", None)), (["2"], ("2", 50119)),
                              (["3", "50045"], ("3", 50045)), (["4"], ("4", None))):
        inputs = iter(answers)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        assert play.select_orange_arrangement() == expected
    inputs = iter(["bad", "3", "bad", "3", "-1", "3", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert play.select_orange_arrangement() == ("3", 0)
    output = capsys.readouterr().out
    assert output.count("Please enter a nonnegative integer seed.") == 2
    assert output.count("Please enter 1, 2, 3, or 4.") == 1


def test_orange_arrangement_options_reach_both_playback_modes(monkeypatch, capsys):
    import play
    marker, selected = object(), []
    monkeypatch.setattr(play, "LiveGame", TestViewer)
    monkeypatch.setattr(play, "load_game_mode", lambda choice: (marker, object()))
    monkeypatch.setattr(play, "play_policy", lambda model, config, label, **kw: selected.append(kw))
    for mode in ("1", "2"):
        for arrangement, expected_seed in (("1", None), ("2", 50119), ("3", 50045)):
            answers = iter(["1", "1", mode, arrangement] + (["50045"] if arrangement == "3" else []))
            monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
            play.main()
            assert selected[-1]["seed"] == expected_seed
            assert (selected[-1]["viewer"] is not None) == (mode == "1")
            label = selected[-1]["arrangement_label"]
            if arrangement == "2":
                assert label == "Selected demonstration case (previously successful)"
            elif arrangement == "1":
                assert label == "Random orange peg arrangement"
            else:
                assert label == "Custom seed arrangement"
    assert capsys.readouterr().out.count("not a randomly generated test game") == 2


def test_orange_arrangement_back_returns_without_loading_or_training(monkeypatch, capsys):
    import play
    # Arrangement Back -> mode Back -> source Back -> select the all-peg model.
    answers = iter(["1", "1", "2", "4", "3", "3", "2", "1", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    loaded, played = [], []
    monkeypatch.setattr(play, "load_game_mode", lambda choice: (loaded.append(choice) or (object(), object())))
    monkeypatch.setattr(play, "play_policy", lambda *args, **kw: played.append(kw))
    def no_training(*args, **kw):
        raise AssertionError("Back navigation started training")
    monkeypatch.setattr(play, "train_fresh", no_training)
    play.main()
    assert loaded == ["2"] and len(played) == 1
    assert played[0]["seed"] is None and played[0]["arrangement_label"] is None
    assert capsys.readouterr().out.count("SELECT PLAYBACK MODE") == 3


def test_display_preserves_training_actions_weights_and_terminal_boards():
    from stable_baselines3.common.monitor import Monitor
    from levels.visualize_data import TrainingDisplay
    torch.set_num_threads(1)
    results = []
    for mode in ("original", "headless", "visual"):
        viewer = TestViewer() if mode == "visual" else None
        display = TrainingDisplay("Test DQN", 64, viewer=viewer, render_every=1)
        env = Monitor(PeggleEnv(run.all_pegs_config()))
        if mode != "original":
            env = display.wrap(env)
        model = DQN("MlpPolicy", env, seed=42, device="cpu", learning_starts=16,
                    buffer_size=128, batch_size=16, train_freq=4)
        model.learn(64, callback=display if mode != "original" else None)
        assert model._n_updates == 12 and model.policy.optimizer.state_dict()["state"]
        assert model.num_timesteps == 64
        if mode != "original":
            assert display.completed_steps == 64 and display.episodes == 6
        buffer = model.replay_buffer
        results.append(([buffer.observations.copy(), buffer.next_observations.copy(),
                         buffer.actions.copy(), buffer.rewards.copy(), buffer.dones.copy()],
                        {key: value.clone() for key, value in model.q_net.state_dict().items()}))
        if mode == "visual":
            assert len(viewer.shots) == 64
            for i, shot in enumerate(viewer.shots):
                assert shot["action"] == int(buffer.actions[i, 0, 0])
                np.testing.assert_array_equal(shot["observation"], buffer.observations[i, 0])
                np.testing.assert_array_equal(shot["next_observation"], buffer.next_observations[i, 0])
                # Terminal images must survive the automatic reset, with the old board's hits.
                if buffer.dones[i, 0]:
                    assert shot["info"]["shots_used"] == 10 and shot["next_observation"][-1] == 0
                    assert shot["trajectory"][-1]["is_hit"] == shot["next_observation"][:-1].astype(bool).tolist()
        else:
            assert not env.unwrapped.record_trajectory and env.unwrapped.trajectory == []
        env.close()
    # Only compare filled replay slots; unused np.empty memory is unspecified.
    for measured in results[1:]:
        for left, right in zip(results[0][0], measured[0]):
            np.testing.assert_array_equal(left[:64], right[:64])
        for key, weights in results[0][1].items():
            assert torch.equal(weights, measured[1][key])


def test_periodic_rendering_and_close_do_not_skip_training():
    from stable_baselines3.common.monitor import Monitor
    from levels.visualize_data import TrainingDisplay
    viewer = TestViewer()
    display = TrainingDisplay("Test", 12, viewer=viewer, render_every=3)
    env = display.wrap(Monitor(PeggleEnv(RLConfig(goal="all", max_shots=1))))
    model = DQN("MlpPolicy", env, seed=42, learning_starts=16, buffer_size=128, device="cpu")
    model.learn(8, callback=display)
    assert [shot["episode"] for shot in viewer.shots] == [1, 3, 6]
    assert display.completed_steps == display.episodes == 8
    viewer.active = False
    model.learn(4, callback=display, reset_num_timesteps=False)
    assert len(viewer.shots) == 3 and display.completed_steps == display.episodes == 12
    assert model.replay_buffer.size() == 12 and env.unwrapped.trajectory == []
    assert not env.unwrapped.record_trajectory
    env.close()


def test_three_visual_training_paths_use_real_updates_and_unique_outputs(tmp_path, monkeypatch):
    import play
    from levels import train_dqn
    one, standard, assisted = train_dqn.train_model_one, train_dqn.train_standard_all_pegs, train_dqn.train_model_two
    monkeypatch.setattr(train_dqn, "train_model_one", lambda folder, **kw: one(folder, steps=208, **kw))
    monkeypatch.setattr(train_dqn, "train_standard_all_pegs", lambda folder, **kw:
        standard(folder, steps=1004, validation_games=1, **kw))
    monkeypatch.setattr(train_dqn, "train_model_two", lambda folder, **kw:
        assisted(folder, base_steps=1004, additional_steps=1004, cloning_updates=4,
                 validation_games=1, interval=1004, **kw))
    folders = []
    for choice in ("1", "2", "3"):
        viewer = TestViewer()
        model, config, folder = play.train_fresh(choice, str(tmp_path), viewer=viewer)
        folders.append(folder)
        assert model._n_updates > 0 and model.policy.optimizer.state_dict()["state"]
        assert os.path.isfile(os.path.join(folder, "best_model.zip")) and viewer.shots
        assert any("Training progress" in stats for stats in viewer.panels)
        if choice == "3":
            assert all(any(name.startswith(f"Stage {stage}:") for name in viewer.stages) for stage in range(1, 6))
            assert any(stats.get("Mode", "").startswith("Demonstration data") for stats in viewer.panels)
            assert len(model.replay_buffer.observations[0, 0]) == 78
        else:
            assert not any("Demonstration" in name for name in viewer.stages)
    assert len(set(folders)) == 3
