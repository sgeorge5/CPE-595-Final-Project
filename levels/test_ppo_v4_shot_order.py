import numpy as np
import pytest

from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv
from levels.test_environment import state_signature
from levels.ppo_v4_shot_order import (config_for, checkpoint_rank, promote_f,
                                      verify_source)
from levels.train_ppo import play_episode


def test_peg32_bonus_once_and_only_before_the_endgame():
    for remaining in (11, 10):
        control = PeggleEnv(config_for("control"))
        experiment = PeggleEnv(config_for("F"))
        np.testing.assert_array_equal(control.config.coordinates, experiment.config.coordinates)
        np.testing.assert_array_equal(control.config.angles_degrees, experiment.config.angles_degrees)
        assert tuple(experiment.config.coordinates[32]) == (887.5, 202.0)
        for env in (control, experiment):
            obs, _ = env.reset(seed=42)
            assert obs.shape == (155,) and env.action_space.n == 21
            for i, peg in enumerate(env.pegs):
                peg.pos = env.Vector(600 if i == 32 else 100, 300)
                peg.isHit = i != 32 and i >= remaining - 1
                peg.isVisible = not peg.isHit
        # An empty shot earns no early-clear bonus.
        assert control.step(20)[1] == experiment.step(20)[1] == -2
        a, b = control.step(10), experiment.step(10)
        assert a[4]["new_pegs_cleared"] == 1 and experiment.pegs[32].isHit
        np.testing.assert_array_equal(a[0], b[0])
        assert a[2:] == b[2:] and state_signature(control) == state_signature(experiment)
        assert b[1] - a[1] == (10 if remaining > 10 else 0)
        assert control.step(10)[1] == experiment.step(10)[1] == -2
        control.close()
        experiment.close()


def test_tracking_preserves_d_gameplay_and_defaults():
    cfg = config_for("D")
    model = verify_source()
    original, trace = play_episode(cfg, "PPO", model, seed=50042)
    tracked, detailed = play_episode(cfg, "PPO", model, seed=50042, track_pegs=True)
    assert original["pegs_cleared"] == tracked["pegs_cleared"] == 76
    assert set(original.items()).issubset(tracked.items())
    assert all(set(a.items()).issubset(b.items()) for a, b in zip(trace, detailed))
    assert sum(len(shot["hit_ids"].split()) for shot in detailed) == 76
    assert not cfg.early_peg_bonus and not RLConfig().early_peg_bonus
    with pytest.raises(ValueError, match="requires"):
        RLConfig(early_peg_bonus=True)


def test_rank_puts_completed_boards_above_peg32_and_menu_uses_primary_metrics():
    a = dict(wins=0, completion_pct=0, avg_shots_success=float("nan"),
             avg_pegs_cleared=75.6, peg32_left_games=50)
    b = dict(a, peg32_left_games=0)
    assert checkpoint_rank(b) > checkpoint_rank(a)
    assert not promote_f(a, b)
    assert promote_f(a, dict(b, avg_pegs_cleared=76.0))
    win = dict(a, wins=1, completion_pct=2, avg_shots_success=11, avg_pegs_cleared=74)
    assert checkpoint_rank(win) > checkpoint_rank(b) and promote_f(a, win)
