"""Settings for one small board and one orange-peg DQN agent."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PEGGLE_ROOT = ROOT / "local" / "PegglePy-main"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODELS_DIR / "dqn_orange.zip"
# Edit these values to change training and evaluation when running the scripts.
SEED = 42
DEFAULT_TIMESTEPS = 10_000
DEFAULT_EVAL_EPISODES = 100

# DQN settings are Python values, shared by the script and notebook.
DQN_POLICY = "MlpPolicy"
DQN_SETTINGS = {
    "learning_rate": 0.001,
    "buffer_size": 20_000,  # Maximum number of stored gameplay transitions.
    "learning_starts": 200,
    "batch_size": 64,
    "gamma": 0.99,  # Discount applied to future rewards.
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 500,
    "exploration_fraction": 0.6,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [64, 64]},
}


# Each entry is (x coordinate, y coordinate, color). This is the only RL board.
FIXED_BOARD = (
    (420.0, 300.0, "orange"),
    (600.0, 420.0, "orange"),
    (800.0, 300.0, "orange"),
    (230.0, 440.0, "blue"),
    (350.0, 550.0, "blue"),
    (500.0, 620.0, "blue"),
    (690.0, 600.0, "blue"),
    (900.0, 500.0, "blue"),
    (1000.0, 650.0, "blue"),
    (600.0, 720.0, "blue"),
)
N_PEGS = len(FIXED_BOARD)
N_ORANGE = sum(color == "orange" for _, _, color in FIXED_BOARD)


@dataclass(frozen=True)
class RLConfig:
    """A group of named settings, accessed as CONFIG.width, CONFIG.max_shots, etc."""

    # Board and action settings.
    width: int = 1200
    height: int = 900
    max_shots: int = 10
    n_actions: int = 21
    min_angle: float = -60.0
    max_angle: float = 60.0
    # Physics defaults from PegglePy, used by the headless adapter.
    launch_force: float = 5.0
    gravity: float = 0.045
    max_ball_velocity: float = 10.0
    ball_radius: float = 12.0
    peg_radius: float = 25.0
    ball_mass: float = 4.0
    peg_mass: float = 20.0
    collision_samples: int = 10
    dt: float = 1.0
    max_frames_per_shot: int = 10_000
    stuck_frames: int = 24  # 0.4 seconds at a nominal 60 simulation ticks/s.
    # Rewards for the orange-only objective.
    orange_reward: float = 10.0
    completion_bonus: float = 25.0
    shot_penalty: float = 1.0

    def __post_init__(self):
        if self.max_shots < 1 or self.max_frames_per_shot < 1 or self.stuck_frames < 1:
            raise ValueError("Shot budget and simulation limits must be positive.")
        if self.n_actions < 2 or not -90 < self.min_angle < self.max_angle < 90:
            raise ValueError("Use at least two downward launch angles in (-90, 90).")
        if self.dt <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Simulation step and board dimensions must be positive.")
        if any(not (self.peg_radius < x < self.width - self.peg_radius
                    and self.peg_radius < y < self.height - self.peg_radius)
               for x, y, _ in FIXED_BOARD):
            raise ValueError("Every fixed peg must fit inside the board.")

    @property
    def angles_degrees(self):
        """21 angles from -60 to +60 degrees, measured from downward vertical."""
        return np.linspace(self.min_angle, self.max_angle, self.n_actions)


CONFIG = RLConfig()


def ensure_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
