"""Train the orange-peg agent. Edit the default budget in config_rl.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from levels.config_rl import (
    CONFIG, DEFAULT_TIMESTEPS, DQN_POLICY, DQN_SETTINGS, MODEL_PATH, RESULTS_DIR, SEED
)
from levels.peggle_env import PeggleEnv


class EpisodeLogger(BaseCallback):
    """SB3 calls this helper while learning; collect each finished game's results."""
    columns = ["episode", "timesteps", "episode_reward", "shots_used", "orange_cleared",
               "success", "simulation_timeouts", "stuck_removals"]

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for info in self.locals["infos"]:
            # Monitor adds "episode" only when a game finishes.
            if "episode" in info:
                self.rows.append({
                    "episode": len(self.rows) + 1,
                    "timesteps": self.num_timesteps,
                    "episode_reward": info["episode"]["r"],
                    "shots_used": info["shots_used"],
                    "orange_cleared": info["orange_cleared"],
                    "success": info["success"],
                    "simulation_timeouts": info["simulation_timeouts"],
                    "stuck_removals": info["stuck_removals"],
                })
        return True


def plot_training(log_path=RESULTS_DIR / "training_episodes.csv", output_path=None):
    data = pd.read_csv(log_path)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    if len(data):
        window = min(50, len(data))
        axes[0].plot(data.timesteps, data.episode_reward, alpha=0.25, color="#2786c4", label="Episode return")
        axes[0].plot(data.timesteps, data.episode_reward.rolling(window, min_periods=1).mean(),
                     color="#124c79", label=f"{window}-episode rolling mean")
        axes[1].plot(data.timesteps, data.success.astype(float).rolling(window, min_periods=1).mean() * 100,
                     color="#ed8b22", label="Rolling training success")
        axes[1].set_ylim(-2, 102)
    else:
        axes[0].text(0.5, 0.5, "No completed episodes in this short run", ha="center", transform=axes[0].transAxes)
    axes[0].set(ylabel="Episode return", title="DQN training on the fixed board (includes exploratory actions)")
    axes[1].set(xlabel="Environment timesteps (complete shots)", ylabel="Success (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig


def train_agent(total_timesteps=DEFAULT_TIMESTEPS, seed=SEED, model_path=MODEL_PATH,
                results_dir=RESULTS_DIR, config=CONFIG, verbose=1):
    """Train a fresh DQN; return the model and measured episode DataFrame.

    Settings live in config_rl.py. Only the model ZIP, episode CSV and reward
    plot are saved; no JSON settings, timestamps or package inventories.
    """
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive.")
    model_path, results_dir = Path(model_path), Path(results_dir)
    if model_path.suffix != ".zip":
        raise ValueError("Use a .zip model filename.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Small MLPs run efficiently and reproducibly on CPU.
    env = Monitor(PeggleEnv(config))
    model = DQN(DQN_POLICY, env, seed=seed, device="cpu", verbose=verbose, **DQN_SETTINGS)
    callback = EpisodeLogger()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=100)
        model.save(model_path)
    finally:
        env.close()
    episodes = pd.DataFrame(callback.rows, columns=EpisodeLogger.columns)
    episodes.to_csv(results_dir / "training_episodes.csv", index=False)
    plt.close(plot_training(results_dir / "training_episodes.csv", results_dir / "training_rewards.png"))
    return model, episodes


def main():
    model, episodes = train_agent()
    print(f"Training shots: {model.num_timesteps}")
    print(f"Completed episodes: {len(episodes)}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
