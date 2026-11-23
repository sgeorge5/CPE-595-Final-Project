import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv
from levels.ppo_endgame_diagnostics import state_signature
from levels.ppo_v4_further import MODELS, START, config_for, settings_for, verify_source


def test_c_bonus_only_for_all_77_and_d_keeps_b_reward():
    for remaining in (11, 1):
        outcomes = []
        signatures = []
        for experiment in ("B", "C", "D"):
            env = PeggleEnv(config_for(experiment))
            obs, _ = env.reset(seed=42)
            assert obs.shape == (155,) and env.action_space.n == 21
            # Leave one peg in the launcher path, plus any others out of reach.
            for i, peg in enumerate(env.pegs):
                peg.pos = env.Vector(600 if i == 0 else 100, 300)
                peg.isHit = i >= remaining
                peg.isVisible = not peg.isHit
            outcomes.append(env.step(10))
            signatures.append(state_signature(env))
            env.close()
        b, c, d = outcomes
        np.testing.assert_array_equal(b[0], c[0])
        np.testing.assert_array_equal(b[0], d[0])
        assert b[2:] == c[2:] == d[2:]
        assert signatures[0] == signatures[1] == signatures[2]
        assert b[1] == d[1] == (105 if remaining == 1 else 3)
        assert c[1] == b[1] + (200 if remaining == 1 else 0)


def test_d_entropy_is_applied_after_checkpoint_load():
    original = verify_source()
    env = Monitor(PeggleEnv(config_for("D")))
    try:
        new = PPO.load(MODELS["B"], env=env, device="cpu", **settings_for("D"))
        assert original.num_timesteps == new.num_timesteps == START
        assert original.ent_coef == settings_for("C")["ent_coef"] == 0.02
        assert new.ent_coef == 0.005 and new.policy_kwargs == original.policy_kwargs
        assert new.learning_rate == original.learning_rate == 0.0003
        assert config_for("D").completion_bonus == config_for("B").completion_bonus == 100
    finally:
        env.close()


def test_old_configuration_defaults_and_geometry_are_preserved():
    dqn = RLConfig(observation="compact")
    assert dqn.n_actions == 21 and not dqn.free_balls and dqn.max_shots == 10
    assert dqn.completion_bonus == 25 and not dqn.endgame_bonus
    for experiment in ("A", "B", "C", "D"):
        config = config_for(experiment)
        assert config.n_pegs == 77 and config.max_shots == 10 and config.free_balls
        np.testing.assert_array_equal(config.coordinates, dqn.coordinates)
        np.testing.assert_array_equal(config.angles_degrees, dqn.angles_degrees)
    with pytest.raises(ValueError, match="requires"):
        RLConfig(ppo_completion_bonus=300)
