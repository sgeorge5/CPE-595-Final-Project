"""Settings for one small board and one orange-peg DQN agent."""

from pathlib import Path

import numpy as np

# Run the notebook or scripts from the Final Project folder.
ROOT = Path(".")
PEGGLE_ROOT = Path("local/PegglePy-main")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/dqn_orange.zip")
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


class RLConfig:
    """A group of named settings, accessed as CONFIG.width, CONFIG.max_shots, etc."""

    def __init__(self, max_shots=10):
        # Board and action settings.
        self.width = 1200
        self.height = 900
        self.max_shots = max_shots
        self.n_actions = 21
        self.min_angle = -60.0
        self.max_angle = 60.0

        # Physics defaults from PegglePy, used by the headless adapter.
        self.launch_force = 5.0
        self.gravity = 0.045
        self.max_ball_velocity = 10.0
        self.ball_radius = 12.0
        self.peg_radius = 25.0
        self.ball_mass = 4.0
        self.peg_mass = 20.0
        self.collision_samples = 10
        self.dt = 1.0
        self.max_frames_per_shot = 10_000
        self.stuck_frames = 24  # 0.4 seconds at a nominal 60 simulation ticks/s.

        # Rewards for the orange-only objective.
        self.orange_reward = 10.0
        self.completion_bonus = 25.0
        self.shot_penalty = 1.0

        self.validate()
        self.angles_degrees = np.linspace(
            self.min_angle, self.max_angle, self.n_actions
        )

    def validate(self):
        """Check settings before using them to create an environment."""
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

CONFIG = RLConfig()


def ensure_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
