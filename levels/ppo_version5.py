"""Compatibility helpers for preserved notebook results and regression checks."""
from levels.config_rl import RLConfig




def version5_config():
    return RLConfig(goal="all", free_balls=True, n_actions=41)


def performance(deterministic_win, deterministic_shots, sampled):
    """Prefer wins, then sampled win rate and shots in wins, then sampled clearance."""
    any_win = bool(deterministic_win or sampled["wins"])
    winning_shots = sampled["avg_shots_success"] if sampled["wins"] else deterministic_shots
    return (any_win, sampled["completion_pct"], -winning_shots if any_win else 0,
            sampled["avg_pegs_cleared"])
