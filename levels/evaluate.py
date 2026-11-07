"""Compare DQN with random shooting. Settings come from config_rl.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from levels.config_rl import (
    CONFIG, DEFAULT_EVAL_EPISODES, MODEL_PATH, N_ORANGE, RESULTS_DIR, SEED
)
from levels.peggle_env import PeggleEnv


def plot_comparison(summary, path=None):
    metrics = [("success_rate_pct", "Success rate (%)"),
               ("orange_cleared_pct", "Orange pegs cleared (%)"),
               ("avg_shots_success", "Mean shots in successful games")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title) in zip(axes, metrics):
        values = summary[column].to_numpy(dtype=float)
        bars = ax.bar(summary.agent, np.nan_to_num(values, nan=0.0), color=["#2786c4", "#ed8b22"])
        for bar, value in zip(bars, values):
            label = "N/A (no successes)" if np.isnan(value) else f"{value:.1f}"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 4), textcoords="offset points")
        ax.set(title=title, ylim=(0, 110 if "pct" in column else CONFIG.max_shots + 2))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Measured performance on the same fixed board and 10-shot budget")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=160)
    return fig


def evaluate_agents(model_path=MODEL_PATH, episodes=DEFAULT_EVAL_EPISODES,
                    seed=SEED + 10_000, results_dir=RESULTS_DIR, config=CONFIG,
                    model=None):
    """Evaluate an in-memory DQN or load its ZIP. No JSON sidecar is needed.

    Use the same config_rl.py board/settings for training and evaluation;
    retrain after changing the task. The notebook passes its freshly trained
    model directly, while the CLI loads the saved checkpoint.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    model_path, results_dir = Path(model_path), Path(results_dir)
    if model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Train the model first: {model_path}")
        model = DQN.load(model_path, device="cpu")
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    shots = []
    for agent in ("DQN", "Random"):
        env = PeggleEnv(config)
        try:
            for episode in range(episodes):
                episode_seed = seed + episode
                obs, _ = env.reset(seed=episode_seed)
                rng = np.random.default_rng(episode_seed)
                total_reward = 0.0
                done = False
                while not done:
                    if agent == "DQN":
                        predicted_action, _ = model.predict(obs, deterministic=True)
                        action = int(predicted_action)
                    else:
                        action = int(rng.integers(config.n_actions))
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    shots.append({"agent": agent, "episode": episode + 1, "seed": episode_seed,
                                  "shot": info["shots_used"], "action": action,
                                  "angle_degrees": info["angle_degrees"], "reward": reward,
                                  "new_orange_cleared": info["new_orange_cleared"],
                                  "orange_remaining": info["orange_remaining"],
                                  "simulation_frames": info["simulation_frames"],
                                  "shot_timed_out": info["shot_timed_out"]})
                    done = terminated or truncated
                rows.append({"agent": agent, "episode": episode + 1, "seed": episode_seed,
                             "episode_reward": total_reward, "success": info["success"],
                             "orange_cleared": info["orange_cleared"],
                             "orange_cleared_pct": 100 * info["orange_cleared"] / N_ORANGE,
                             "shots_used": info["shots_used"],
                             "simulation_timeouts": info["simulation_timeouts"],
                             "stuck_removals": info["stuck_removals"]})
        finally:
            env.close()
    episodes_df = pd.DataFrame(rows)
    summary_rows = []
    for agent, data in episodes_df.groupby("agent", sort=False):
        successes = data[data.success]
        if len(successes) > 0:
            average_successful_shots = successes.shots_used.mean()
        else:
            average_successful_shots = np.nan  # No successful games to average.
        summary_rows.append({"agent": agent, "episodes": len(data), "successes": len(successes),
                             "success_rate_pct": 100 * data.success.mean(),
                             "orange_cleared_pct": data.orange_cleared_pct.mean(),
                             "avg_shots_success": average_successful_shots,
                             "avg_shots_all": data.shots_used.mean(),
                             "mean_episode_reward": data.episode_reward.mean(),
                             "simulation_timeouts": int(data.simulation_timeouts.sum())})
    summary = pd.DataFrame(summary_rows)
    episodes_df.to_csv(results_dir / "evaluation_episodes.csv", index=False)
    pd.DataFrame(shots).to_csv(results_dir / "evaluation_shots.csv", index=False)
    summary.to_csv(results_dir / "evaluation_summary.csv", index=False)
    plt.close(plot_comparison(summary, results_dir / "dqn_vs_random.png"))
    return summary, episodes_df


def main():
    summary, _ = evaluate_agents()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
