"""Compatibility helpers for preserved notebook results and regression checks."""





def state_signature(env):
    """State carried between shots; the next ball starts at the normal launcher."""
    return (
        tuple((peg.pos.x, peg.pos.y, peg.color, peg.isHit, peg.isVisible) for peg in env.pegs),
        env.shots_remaining, env.shots_fired, env.shot_score, env.total_score,
        env.shot_free_balls, env.free_balls_earned, env.shots_earning_free_balls,
        env.simulation_timeouts, env.stuck_removals,
    )
