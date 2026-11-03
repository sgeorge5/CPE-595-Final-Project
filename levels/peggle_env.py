"""Headless orange-peg Gymnasium environment using PegglePy's actual physics.

Upstream: https://github.com/Mr0o/PegglePy (GPL-3.0; see upstream LICENSE).
The upstream Vector, Ball and collision function bodies execute unchanged.
Only imports are redirected, in memory, to an isolated package with explicit
physics settings and a lightweight peg record. No game files are modified.
"""

import ast
from dataclasses import dataclass
from functools import lru_cache
import sys
import types
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

if __package__:
    from .config_rl import CONFIG, FIXED_BOARD, N_ORANGE, N_PEGS, PEGGLE_ROOT, RLConfig
else:
    from config_rl import CONFIG, FIXED_BOARD, N_ORANGE, N_PEGS, PEGGLE_ROOT, RLConfig


@dataclass
class HeadlessPeg:
    """PegglePy's physics/state fields, without drawing, timers or animation."""
    pos: Any
    color: str
    radius: float
    mass: float
    isHit: bool = False
    isVisible: bool = True

    @property
    def isOrange(self):
        return self.color == "orange"


@lru_cache(maxsize=None)
def _load_physics(config):
    """Redirect upstream local.* imports without replacing the global local package.

    Importing upstream local.config directly would load user settings, run an
    automatic installer and potentially initialize a display. The private config
    supplies its documented default constants instead. Collision code needs only
    Peg's position and radius, so graphical Peg construction is unnecessary.
    """
    source_dir = PEGGLE_ROOT / "local"
    if not (source_dir / "collision.py").exists():
        raise FileNotFoundError(f"Expected the downloaded PegglePy source at {PEGGLE_ROOT}")
    package_name = f"_peggle_rl_physics_{len(_PHYSICS_PACKAGES)}"
    _PHYSICS_PACKAGES.append(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    sys.modules[package_name] = package

    def module(name):
        result = types.ModuleType(f"{package_name}.{name}")
        sys.modules[result.__name__] = result
        setattr(package, name, result)
        return result

    def load(name):
        target = module(name)
        path = source_dir / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("local."):
                node.module = package_name + node.module[len("local"):]
        target.__file__ = str(path)
        exec(compile(tree, str(path), "exec"), target.__dict__)
        return target

    vectors = load("vectors")
    settings = module("config")
    settings.configs = {"WIDTH": config.width, "HEIGHT": config.height}
    settings.gravity = vectors.Vector(0, config.gravity)
    settings.maxBallVelocity = config.max_ball_velocity
    settings.ballRad = config.ball_radius
    settings.defaultBallMass = config.ball_mass
    settings.collisionSampleSize = config.collision_samples
    module("peg").Peg = HeadlessPeg
    ball = load("ball")
    collision = load("collision")
    return vectors.Vector, ball.Ball, collision


_PHYSICS_PACKAGES = []


class PeggleEnv(gym.Env):
    """One step launches and finishes one shot; actions never use a trajectory oracle.

    Hit pegs remain solid during the shot and are removed at its end, as in
    PegglePy. Low-speed repeated contacts remove an already-hit peg after a
    deterministic 24-tick delay, replacing the game's wall-clock stuck timer.
    Bucket/free-ball/power-up/score effects are deliberately outside this stage.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: RLConfig = CONFIG):
        super().__init__()
        self.config = config
        self.Vector, self.Ball, self.collision = _load_physics(config)
        self.action_space = spaces.Discrete(config.n_actions)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_PEGS * 4 + 1,), dtype=np.float32)
        self.render_mode = None
        self.pegs = []
        self.shots_remaining = config.max_shots
        self._done = True
        self.simulation_timeouts = 0
        self.stuck_removals = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pegs = [HeadlessPeg(self.Vector(x, y), color, self.config.peg_radius,
                                 self.config.peg_mass) for x, y, color in FIXED_BOARD]
        self.shots_remaining = self.config.max_shots
        self.simulation_timeouts = 0
        self.stuck_removals = 0
        self._done = False
        return self._observation(), self._info()

    def _observation(self):
        features = [[peg.pos.x / self.config.width, peg.pos.y / self.config.height,
                     float(peg.isOrange), float(peg.isHit)] for peg in self.pegs]
        return np.concatenate((np.asarray(features, dtype=np.float32).ravel(),
                               [self.shots_remaining / self.config.max_shots])).astype(np.float32)

    def board_records(self):
        """Extract raw coordinates, colors and current hit status in stable ID order."""
        return [{"peg_id": i, "x": float(peg.pos.x), "y": float(peg.pos.y),
                 "color": peg.color, "is_hit": peg.isHit, "is_visible": peg.isVisible,
                 "radius": peg.radius} for i, peg in enumerate(self.pegs)]

    def _info(self):
        remaining = sum(p.isOrange and not p.isHit for p in self.pegs)
        return {"orange_remaining": remaining, "orange_cleared": N_ORANGE - remaining,
                "success": remaining == 0, "shots_remaining": self.shots_remaining,
                "shots_used": self.config.max_shots - self.shots_remaining,
                "simulation_timeouts": self.simulation_timeouts,
                "stuck_removals": self.stuck_removals}

    def _simulate_shot(self, angle_degrees):
        cfg = self.config
        ball = self.Ball(cfg.width / 2, cfg.height / 25)
        ball.isAlive = ball.isLaunch = True
        angle = np.deg2rad(angle_degrees)
        ball.applyForce(self.Vector(cfg.launch_force * np.sin(angle),
                                    cfg.launch_force * np.cos(angle)))
        active = [p for p in self.pegs if not p.isHit]
        slow_contacts = {}
        timed_out = True
        for frame in range(1, cfg.max_frames_per_shot + 1):
            ball.update(cfg.dt)
            if not ball.isAlive:
                timed_out = False
                break
            # Cheap swept bounding-box filter; collision decisions and response
            # still use the upstream functions, including their sampling logic.
            reach = ball.radius + cfg.peg_radius
            xmin, xmax = min(ball.prevPos.x, ball.pos.x) - reach, max(ball.prevPos.x, ball.pos.x) + reach
            ymin, ymax = min(ball.prevPos.y, ball.pos.y) - reach, max(ball.prevPos.y, ball.pos.y) + reach
            nearby = [p for p in active if xmin <= p.pos.x <= xmax and ymin <= p.pos.y <= ymax]
            touched = [p for p in nearby if self.collision.isBallTouchingPeg(ball, p, cfg.dt)]
            if touched:
                ball = self.collision.resolveCollision(ball, touched, cfg.dt)
                for peg in touched:
                    peg.isHit = True
                    key = id(peg)
                    if ball.vel.getMag() <= 1.5:
                        slow_contacts.setdefault(key, frame)
                    else:
                        slow_contacts.pop(key, None)
                    if key in slow_contacts and frame - slow_contacts[key] >= cfg.stuck_frames:
                        active.remove(peg)
                        peg.isVisible = False
                        self.stuck_removals += 1
            # Keep the timer across the tiny gaps between bounces, as upstream
            # does; a subsequent fast contact cancels it for that peg.
        for peg in self.pegs:
            if peg.isHit:
                peg.isVisible = False
        self.simulation_timeouts += int(timed_out)
        return frame, timed_out

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in [0, {self.config.n_actions - 1}].")
        before = self._info()["orange_cleared"]
        angle = float(self.config.angles_degrees[int(action)])
        frames, timeout = self._simulate_shot(angle)
        self.shots_remaining -= 1
        info = self._info()
        newly_cleared = info["orange_cleared"] - before
        reward = self.config.orange_reward * newly_cleared - self.config.shot_penalty
        terminated = bool(info["success"] or self.shots_remaining == 0)
        # The shot budget is part of the finite task and observation, so failure
        # at zero shots is terminal, not a time-limit truncation to bootstrap from.
        if info["success"]:
            reward += self.config.completion_bonus
        self._done = terminated
        info.update(new_orange_cleared=newly_cleared, angle_degrees=angle,
                    simulation_frames=frames, shot_timed_out=timeout,
                    terminal_reason="success" if info["success"] else
                                    "shot_budget" if terminated else None)
        return self._observation(), float(reward), terminated, False, info
