import numpy as np

from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv
from levels.ppo_v4_improved import experiment_config, verify_original_settings
from levels.train_ppo import validation_score


def test_endgame_reward_changes_only_new_pegs_in_last_ten():
    a = PeggleEnv(experiment_config("A"), record_trajectory=True)
    b = PeggleEnv(experiment_config("B"), record_trajectory=True)
    for remaining in (11, 10, 1):
        outcomes = []
        for env in (a, b):
            obs, _ = env.reset(seed=42)
            assert obs.shape == (155,) and env.action_space.n == 21
            assert len(env.pegs) == 77 and env.shots_remaining == 10
            # One reachable peg and any other remaining pegs off the shot path.
            for index, peg in enumerate(env.pegs):
                peg.isHit = index >= remaining
                peg.isVisible = not peg.isHit
                peg.pos = env.Vector(600 if index == 0 else 100, 300)
            outcomes.append(env.step(10))
        obs_a, reward_a, done_a, truncated_a, info_a = outcomes[0]
        obs_b, reward_b, done_b, truncated_b, info_b = outcomes[1]
        np.testing.assert_array_equal(obs_a, obs_b)
        assert (done_a, truncated_a, info_a) == (done_b, truncated_b, info_b)
        assert a.trajectory == b.trajectory and info_a["new_pegs_cleared"] == 1
        assert reward_a == (103 if remaining == 1 else 3)
        assert reward_b == reward_a + (2 if remaining <= 10 else 0)
        if remaining == 10:
            assert a.step(10)[1] == b.step(10)[1] == -2


def test_original_checkpoint_and_default_reward_modes():
    assert verify_original_settings().num_timesteps == 100352
    for config in (RLConfig(observation="compact"), RLConfig(goal="all"),
                   experiment_config("original"), RLConfig(goal="all", free_balls=True, n_actions=41)):
        assert not config.endgame_bonus
    dqn = RLConfig(observation="compact")
    assert dqn.n_pegs * dqn.features_per_peg + 1 == 155
    assert dqn.n_actions == 21 and dqn.max_shots == 10 and not dqn.free_balls


def test_validation_ranking_uses_wins_then_shots_then_clearance():
    loss = dict(wins=0, completion_pct=0, avg_shots_success=float("nan"), avg_pegs_cleared=76)
    win = dict(wins=1, completion_pct=2, avg_shots_success=12, avg_pegs_cleared=72)
    assert validation_score(win) > validation_score(loss)
    assert validation_score(dict(win, wins=2, completion_pct=4)) > validation_score(win)
    assert validation_score(dict(win, avg_shots_success=11)) > validation_score(win)
    assert validation_score(dict(win, avg_pegs_cleared=73)) > validation_score(win)
    assert validation_score(loss) > validation_score(dict(loss, avg_pegs_cleared=75))
