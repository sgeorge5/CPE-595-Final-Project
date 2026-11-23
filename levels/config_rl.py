"""Level 1 settings for DQN and PPO."""

import os
import pickle

import numpy as np

# Run the notebook or scripts from the Final Project folder.
PEGGLE_ROOT = "local/PegglePy-main"
MODELS_DIR = "models"
RESULTS_DIR = "results/level1_random"
ORIGINAL_LEVEL1_MODEL_PATH = "models/dqn_level1_random.zip"
IMPROVED_MODELS_DIR = "models/level1_improved"
IMPROVED_RESULTS_DIR = "results/level1_improved"
BEST_MODEL_PATH = "models/level1_improved/best_model.zip"
VALIDATION_SEED = 20_042
TEST_SEED = 50_042
CHECKPOINT_INTERVAL = 5_000
VALIDATION_EPISODES = 10
# Edit these values to change training and evaluation when running the scripts.
SEED = 42
DEFAULT_TIMESTEPS = 10_000
DEFAULT_EVAL_EPISODES = 100
PPO_MAX_SHOTS = 10
PPO_TIMESTEPS = 10_000
PPO_MODEL_PATH = "models/ppo_all_pegs_level1.zip"
PPO_RESULTS_DIR = "results/ppo_all_pegs"
PPO_VERSIONS_DIR = "results/ppo_all_pegs/versions"
PPO_V4_RESULTS_DIR = "results/ppo_version4"
PPO_V4_MODEL_PATH = "models/ppo_version4_best.zip"
PPO_V4_IMPROVED_DIR = "results/ppo_v4_improved"
PPO_V4_FURTHER_DIR = "results/ppo_v4_further"
PPO_V4_SHOT_ORDER_DIR = "results/ppo_v4_improved_F"
PPO_V5_RESULTS_DIR = "results/ppo_version5"
PPO_V5_MODEL_PATH = "models/ppo_version5_best.zip"
PPO_SETTINGS = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "ent_coef": 0.01,
    "policy_kwargs": {"net_arch": [64, 64]},
}
PPO_V4_SETTINGS = dict(PPO_SETTINGS, ent_coef=0.02)
PPO_V5_SETTINGS = dict(PPO_V4_SETTINGS)

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


class _CoordinateUnpickler(pickle.Unpickler):
    """The bundled level files should contain coordinates, not executable objects."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError("Level data must contain primitive coordinates only.")


def load_level_coordinates(level_name="Level 1.lvl", center=True, width=1200, height=900):
    """Keep archive order and apply only PegglePy loadData's normal translation."""
    level_dir = os.path.realpath(os.path.join(PEGGLE_ROOT, "levels"))
    path = os.path.realpath(os.path.join(level_dir, level_name))
    if (os.path.normcase(os.path.dirname(path)) != os.path.normcase(level_dir)
            or os.path.splitext(path)[1] != ".lvl"):
        raise ValueError("Choose a bundled .lvl filename.")
    with open(path, "rb") as level_file:
        positions = np.asarray(_CoordinateUnpickler(level_file).load(), dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0
            or not np.isfinite(positions).all()):
        raise ValueError("Expected a nonempty list of finite (x, y) coordinates.")
    if center:
        midpoint = (positions.min(axis=0) + positions.max(axis=0)) / 2
        positions = positions + np.array([width / 2, height / 2]) - midpoint
    return positions


class RLConfig:
    """A group of named settings, accessed as CONFIG.width, CONFIG.max_shots, etc."""

    def __init__(self, max_shots=10, observation="full", empty_shot_penalty=0, goal="orange",
                 free_balls=False, n_actions=21, endgame_bonus=False, ppo_completion_bonus=100,
                 early_peg_bonus=False):
        # Board and action settings.
        self.width = 1200
        self.height = 900
        self.max_shots = max_shots
        self.n_actions = n_actions
        self.min_angle = -60.0
        self.max_angle = 60.0
        if goal not in ("orange", "all"):
            raise ValueError('goal must be "orange" or "all".')
        self.goal = goal
        self.free_balls = free_balls
        self.endgame_bonus = endgame_bonus
        self.early_peg_bonus = early_peg_bonus
        if endgame_bonus and (goal != "all" or not free_balls or n_actions != 21):
            raise ValueError("The endgame bonus is reserved for the PPO V4 experiment.")
        if ppo_completion_bonus != 100 and not endgame_bonus:
            raise ValueError("A custom completion bonus requires the PPO V4 endgame reward.")
        if early_peg_bonus and not endgame_bonus:
            raise ValueError("The early peg bonus requires the PPO V4 endgame reward.")
        if free_balls and goal != "all":
            raise ValueError("Free balls are only available for PPO's all-peg objective.")
        if goal == "all":
            observation = "color_hits" if free_balls else "hits"
        elif observation not in ("full", "compact"):
            raise ValueError('observation must be "full" or "compact".')
        self.observation = observation
        self.features_per_peg = {"full": 4, "compact": 2, "hits": 1, "color_hits": 2}[observation]
        self.coordinates = load_level_coordinates(width=self.width, height=self.height)
        if len(self.coordinates) != 77:
            raise ValueError("Expected all 77 pegs from the original Level 1.")
        self.n_orange = 25  # PegglePy local/peg.py: createPegColors.
        self.n_pegs = len(self.coordinates)
        # Each peg can score once; at most three free balls per scoring shot.
        self.ball_observation_scale = max_shots + 3 * self.n_pegs if free_balls else max_shots

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
        self.empty_shot_penalty = empty_shot_penalty
        if goal == "all":
            self.peg_reward = 5.0
            self.shot_penalty = 2.0
            self.completion_bonus = float(ppo_completion_bonus)

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
        if self.empty_shot_penalty < 0:
            raise ValueError("The empty-shot penalty must be nonnegative.")
        if any(not (self.peg_radius < x < self.width - self.peg_radius
                    and self.peg_radius < y < self.height - self.peg_radius)
               for x, y in self.coordinates):
            raise ValueError("Every fixed peg must fit inside the board.")

CONFIG = RLConfig()
