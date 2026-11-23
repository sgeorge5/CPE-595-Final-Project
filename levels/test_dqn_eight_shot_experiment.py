"""Check isolated rewards and demonstrated transitions without retraining old models."""
import numpy as np
import os
import pandas as pd
from levels import dqn_eight_shot_experiment as run
from levels import dqn_all_pegs as original
from levels.peggle_env import PeggleEnv


def test_bonus_only_changes_actual_success_reward():
    base = PeggleEnv(original.all_pegs_config())
    bonus = run.CompletionReward()
    logged = pd.read_csv(os.path.join(run.DEMO_FOLDER, "shots.csv"))
    a, _ = base.reset(seed=50042)
    b, _ = bonus.reset(seed=50042)
    np.testing.assert_array_equal(a, b)
    try:
        for row in logged.itertuples():
            a, reward_a, done_a, truncated_a, info_a = base.step(int(row.action))
            b, reward_b, done_b, truncated_b, info_b = bonus.step(int(row.action))
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
