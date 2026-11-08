"""Gymnasium environment: reset a board, choose an angle, simulate a full shot."""

import ast
import sys
import types

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from levels.config_rl import CONFIG, FIXED_BOARD, PEGGLE_ROOT, RLConfig


def random_peg_colors(n_pegs, n_orange, rng):
    """Choose distinct orange pegs once per game; all other pegs are blue."""
    colors = ["blue"] * n_pegs
    for index in rng.choice(n_pegs, size=n_orange, replace=False):
        colors[index] = "orange"
    return colors


class HeadlessPeg:
    """Peg position, color and hit status, without game images or animations."""

    def __init__(self, pos, color, radius, mass):
        self.pos = pos
        self.color = color
        self.radius = radius
        self.mass = mass
        self.isHit = False
        self.isVisible = True
        self.isOrange = color == "orange"


# Remember physics that has already been loaded for a set of settings.
_PHYSICS_CACHE = {}


def _load_physics(config):
    """Load PegglePy's physics without starting its graphical game.

    PegglePy imports its settings and classes as local.* modules. Redirecting
    those imports keeps the original physics functions intact while supplying
    headless settings. The sys/ast/module code is confined to this adapter.
    """
    cache_key = (
        config.width, config.height, config.gravity,
        config.max_ball_velocity, config.ball_radius,
        config.ball_mass, config.collision_samples,
    )
    if cache_key in _PHYSICS_CACHE:
        return _PHYSICS_CACHE[cache_key]

    source_dir = PEGGLE_ROOT / "local"
    if not (source_dir / "collision.py").exists():
        raise FileNotFoundError(f"Expected the downloaded PegglePy source at {PEGGLE_ROOT}")
    package_name = f"_peggle_rl_physics_{len(_PHYSICS_CACHE)}"
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
    physics = (vectors.Vector, ball.Ball, collision)
    _PHYSICS_CACHE[cache_key] = physics
    return physics


class PeggleEnv(gym.Env):
    """DQN observes the board between shots; one step finishes one whole shot."""
    metadata = {"render_modes": []}

    def __init__(self, config: RLConfig = CONFIG, record_trajectory=False):
        super().__init__()
        self.config = config
        self.Vector, self.Ball, self.collision = _load_physics(config)
        self.action_space = spaces.Discrete(config.n_actions)
        size = config.n_pegs * config.features_per_peg + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self.render_mode = None
        self.pegs = []
        self.shots_remaining = config.max_shots
        self._done = True
        self.simulation_timeouts = 0
        self.stuck_removals = 0
        # Optional display data only; training leaves recording switched off.
        self.record_trajectory = record_trajectory
        self.trajectory = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.config.random_oranges:
            colors = random_peg_colors(self.config.n_pegs, self.config.n_orange,
                                      self.np_random)
        else:
            colors = [color for x, y, color in FIXED_BOARD]
        self.pegs = []
        for (x, y), color in zip(self.config.coordinates, colors):
            self.pegs.append(HeadlessPeg(self.Vector(x, y), color,
                                        self.config.peg_radius, self.config.peg_mass))
        self.orange_initial = sum(peg.isOrange for peg in self.pegs)
        self.shots_remaining = self.config.max_shots
        self.simulation_timeouts = 0
        self.stuck_removals = 0
        self._done = False
        self.trajectory = []
        return self._observation(), self._info()

    def _observation(self, mode=None):
        mode = self.config.observation if mode is None else mode
        features = []
        for peg in self.pegs:
            if mode == "full":
                features.extend([peg.pos.x / self.config.width,
                                 peg.pos.y / self.config.height])
            features.extend([float(peg.isOrange), float(peg.isHit)])
        features.append(self.shots_remaining / self.config.max_shots)
        return np.array(features, dtype=np.float32)

    def board_records(self):
        """Extract raw coordinates, colors and current hit status in stable ID order."""
        return [{"peg_id": i, "x": float(peg.pos.x), "y": float(peg.pos.y),
                 "color": peg.color, "is_hit": peg.isHit, "is_visible": peg.isVisible,
                 "radius": peg.radius} for i, peg in enumerate(self.pegs)]

    def _info(self):
        remaining = sum(p.isOrange and not p.isHit for p in self.pegs)
        return {"orange_initial": self.orange_initial,
                "orange_remaining": remaining, "orange_cleared": self.orange_initial - remaining,
                "success": remaining == 0, "shots_remaining": self.shots_remaining,
                "shots_used": self.config.max_shots - self.shots_remaining,
                "simulation_timeouts": self.simulation_timeouts,
                "stuck_removals": self.stuck_removals}

    def _record_frame(self, ball, frame, shot_finished=False):
        if self.record_trajectory:
            # Copy values now so later collisions cannot change earlier frames.
            self.trajectory.append({
                "frame": frame, "x": float(ball.pos.x), "y": float(ball.pos.y),
                "ball_alive": ball.isAlive and not shot_finished,
                "is_hit": [peg.isHit for peg in self.pegs],
                "is_visible": [peg.isVisible for peg in self.pegs],
            })

    def _simulate_shot(self, angle_degrees):
        cfg = self.config
        ball = self.Ball(cfg.width / 2, cfg.height / 25)
        ball.isAlive = ball.isLaunch = True
        angle = np.deg2rad(angle_degrees)
        ball.applyForce(self.Vector(cfg.launch_force * np.sin(angle),
                                    cfg.launch_force * np.cos(angle)))
        self.trajectory = []  # Only keep the current shot.
        self._record_frame(ball, 0)
        active = [p for p in self.pegs if not p.isHit]
        slow_contacts = {}
        timed_out = True
        for frame in range(1, cfg.max_frames_per_shot + 1):
            ball.update(cfg.dt)
            if not ball.isAlive:
                self._record_frame(ball, frame)
                timed_out = False
                break
            reach = ball.radius + cfg.peg_radius
            xmin, xmax = min(ball.prevPos.x, ball.pos.x) - reach, max(ball.prevPos.x, ball.pos.x) + reach
            ymin, ymax = min(ball.prevPos.y, ball.pos.y) - reach, max(ball.prevPos.y, ball.pos.y) + reach
            nearby = [p for p in active if xmin <= p.pos.x <= xmax and ymin <= p.pos.y <= ymax]
            touched = [p for p in nearby if self.collision.isBallTouchingPeg(ball, p, cfg.dt)]
            if touched:
                ball = self.collision.resolveCollision(ball, touched, cfg.dt)
                # The response stops at its first impact. Later candidates on the
                # original swept path were not necessarily reached after that bounce.
                contacts = []
                for peg in touched:
                    distance = np.hypot(ball.pos.x - peg.pos.x, ball.pos.y - peg.pos.y)
                    if abs(distance - ball.radius - peg.radius) < 0.000001:
                        contacts.append(peg)
                touched = contacts
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
            self._record_frame(ball, frame)
        for peg in self.pegs:
            if peg.isHit:
                peg.isVisible = False
        self._record_frame(ball, frame, shot_finished=True)
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
        if newly_cleared == 0:
            reward -= self.config.empty_shot_penalty
        terminated = bool(info["success"] or self.shots_remaining == 0)
        if info["success"]:
            reward += self.config.completion_bonus
        self._done = terminated
        if info["success"]:
            terminal_reason = "success"
        elif terminated:
            terminal_reason = "shot_budget"
        else:
            terminal_reason = None
        info.update(new_orange_cleared=newly_cleared, angle_degrees=angle,
                    simulation_frames=frames, shot_timed_out=timeout,
                    terminal_reason=terminal_reason)
        return self._observation(), float(reward), terminated, False, info
