import os

import numpy as np
import pandas as pd
import pytest

from levels import dqn_all_pegs as original
from levels import dqn_shot_efficiency as run
from levels.peggle_env import PeggleEnv


def test_original_trace_and_fresh_search_reset():
    config = original.all_pegs_config()
    assert len(config.coordinates) == 77 and config.max_shots == 10 and not config.free_balls
    np.testing.assert_array_equal(config.angles_degrees, run.ANGLES)
    env = PeggleEnv(config)
    try:
        game, shots = run.play_sequence(env, (0, 36, 36, -18, 6, -60, -18, -60, -18, -24))
        assert game["success"] and game["shots_used"] == 10 and game["peg32_shot"] == 1
        assert [shot["new_pegs_cleared"] for shot in shots] == [26, 7, 5, 7, 8, 7, 8, 6, 2, 1]
        _, one = run.play_sequence(env, (0,))
        assert one == shots[:1] and one[0]["shots_remaining"] == 9
        with pytest.raises(ValueError):
            run.play_sequence(env, (1,))
    finally:
        env.close()




def test_verified_eight_shot_replay_ends_with_two_shots_left():
    angles = (-6, 12, -30, -60, -54, 0, -42, -42)
    env = PeggleEnv(original.all_pegs_config())
    try:
        game, trace = run.play_sequence(env, angles + (0,))
        assert (game["pegs_cleared"], game["shots_used"], game["peg32_shot"]) == (77, 8, 3)
        assert trace[-1]["shots_remaining"] == 2 and trace[-1]["success"]
        np.testing.assert_array_equal([shot["angle_degrees"] for shot in trace], angles)
    finally:
        env.close()






def test_saved_search_evidence_and_replay():
    root = "results/dqn_shot_efficiency"
    trials = pd.read_csv(os.path.join(root, "search_candidates.csv"))
    assert len(trials) == 20000
    solutions = pd.read_csv(os.path.join(root, "search_solutions.csv"))
    assert len(solutions) == 2 and solutions.shots_used.eq(8).all()
    game = run.playback(show=False)
    assert game["replay_verified"] and game["shots_used"] == 8
