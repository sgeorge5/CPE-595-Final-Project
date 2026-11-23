"""Replay the saved simulator solution through the original Level 1 physics."""
import argparse
import os
from urllib.request import pathname2url
import webbrowser
import pandas as pd
import numpy as np
from levels import dqn_all_pegs as original

ANGLES = tuple(range(-60, 61, 6))
BOARD_SEED = original.DEMO_SEED
SEARCH_LABEL = "Simulator search"


def play_sequence(env, angles):
    """Reset to the same complete board, then stop on success or sequence end."""
    if not 1 <= len(angles) <= 10 or any(angle not in ANGLES for angle in angles):
        raise ValueError("Use one to ten shots drawn from the original 21 angles.")
    obs, _ = env.reset(seed=BOARD_SEED)
    orange_ids = angle_text(i for i, peg in enumerate(env.pegs) if peg.isOrange)
    trace = []
    for angle in angles:
        action = ANGLES.index(angle)
        next_obs, reward, done, truncated, info = env.step(action)
        hits = np.flatnonzero(next_obs[:77] > obs[:77])
        assert len(hits) == info["new_pegs_cleared"] and info["shots_remaining"] == 9 - len(trace)
        assert "free_balls_earned" not in info and "shot_score" not in info
        trace.append(dict(shot=info["shots_used"], shots_used=info["shots_used"], angle_degrees=angle,
                          action=action, hit_ids=angle_text(hits), peg32_hit=32 in hits, reward=reward,
                          **{k: info[k] for k in ("new_pegs_cleared", "pegs_cleared", "pegs_remaining",
                                                 "shots_remaining", "success")}))
        obs = next_obs
        if done or truncated:
            break
    game = dict(model=SEARCH_LABEL, strategy="Recorded angle sequence", seed=BOARD_SEED,
                orange_peg_ids=orange_ids, actions=angle_text(row["action"] for row in trace),
                angle_sequence=angle_text(row["angle_degrees"] for row in trace),
                shots_used=len(trace), pegs_cleared=info["pegs_cleared"], success=info["success"],
                peg32_shot=next((row["shot"] for row in trace if row["peg32_hit"]), None))
    return game, trace


def angle_text(angles):
    return " ".join(map(str, angles))


def parse_angles(text):
    return tuple(int(angle) for angle in str(text).split())


def playback(show=True):
    folder = "results/dqn_shot_efficiency/solutions/candidate_12466"
    game = pd.read_csv(os.path.join(folder, "game.csv")).iloc[0].to_dict()
    trace = pd.read_csv(os.path.join(folder, "shots.csv"))
    original.replay_data(game, trace)
    assert game["shots_used"] == 8 and game["pegs_cleared"] == 77
    print("Simulator Search — Eight-Shot Solution")
    print(f"Angles {game['angle_sequence']}. Cleared 77/77 in eight shots; peg 32 on shot 3.")
    replay = os.path.join(folder, "replay.gif")
    print(replay)
    if show:
        webbrowser.open("file:" + pathname2url(os.path.abspath(replay)))
    return game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["play"])
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    playback(show=not args.no_browser)
