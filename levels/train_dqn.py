import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

if __package__:
    from .config_rl import (CONFIG, DEFAULT_TIMESTEPS, MODEL_PATH, RESULTS_DIR, SEED,
                            experiment_id, experiment_spec)
    from .peggle_env import PeggleEnv
else:
    from config_rl import (CONFIG, DEFAULT_TIMESTEPS, MODEL_PATH, RESULTS_DIR, SEED,
                           experiment_id, experiment_spec)
    from peggle_env import PeggleEnv


class EpisodeLogger(BaseCallback):
    """One CSV row per completed training episode; never synthesize results."""
    fields = ["episode", "timesteps", "episode_reward", "shots_used", "orange_cleared",
              "success", "simulation_timeouts", "stuck_removals"]

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.count = 0

    def _on_training_start(self):
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.count += 1
                self.writer.writerow({"episode": self.count, "timesteps": self.num_timesteps,
                                      "episode_reward": info["episode"]["r"],
                                      **{key: info[key] for key in self.fields[3:]}})
                self.handle.flush()
        return True

    def _on_training_end(self):
        self.handle.close()


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
    """A single initial configuration, no hyperparameter search or action oracle."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive.")
    model_path, results_dir = Path(model_path), Path(results_dir)
    if model_path.suffix != ".zip":
        raise ValueError("Use a .zip model filename.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Small MLPs run efficiently and reproducibly on CPU.
    env = Monitor(PeggleEnv(config), filename=str(results_dir / "training.monitor.csv"),
                  info_keywords=("orange_cleared", "success", "shots_used"))
    params = dict(learning_rate=1e-3, buffer_size=20_000, learning_starts=200,
                  batch_size=64, gamma=0.99, train_freq=4, gradient_steps=1,
                  target_update_interval=500, exploration_fraction=0.6,
                  exploration_initial_eps=1.0, exploration_final_eps=0.05,
                  policy_kwargs=dict(net_arch=[64, 64]))
    model = DQN("MlpPolicy", env, seed=seed, device="cpu", verbose=verbose, **params)
    callback = EpisodeLogger(results_dir / "training_episodes.csv")
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=100)
        model.save(model_path)
    finally:
        if hasattr(callback, "handle") and not callback.handle.closed:
            callback.handle.close()
        env.close()
    metadata = {"algorithm": "Stable-Baselines3 DQN", "policy": "MlpPolicy", "seed": seed,
                "requested_timesteps": total_timesteps, "actual_timesteps": model.num_timesteps,
                "completed_episodes": callback.count, "elapsed_seconds": time.perf_counter() - started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(model_path.resolve()), "hyperparameters": params,
                "experiment_id": experiment_id(config), "experiment": experiment_spec(config),
                "python": platform.python_version(), "versions": {
                    name: importlib.metadata.version(name) for name in
                    ("numpy", "torch", "gymnasium", "stable-baselines3", "matplotlib", "pandas")}}
    # Sidecar ties evaluation conditions to this specific model configuration.
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (results_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plt.close(plot_training(results_dir / "training_episodes.csv", results_dir / "training_rewards.png"))
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    _, metadata = train_agent(args.timesteps, args.seed, args.model_path, args.results_dir)
    print(json.dumps({key: metadata[key] for key in ("actual_timesteps", "completed_episodes", "elapsed_seconds", "model_path")}, indent=2))


if __name__ == "__main__":
    main()
