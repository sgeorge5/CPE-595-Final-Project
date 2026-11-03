"""Measure deterministic DQN vs uniform random angles under identical conditions."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

if __package__:
    from .config_rl import (CONFIG, DEFAULT_EVAL_EPISODES, MODEL_PATH, N_ORANGE,
                            RESULTS_DIR, SEED, experiment_id, experiment_spec)
    from .peggle_env import PeggleEnv
else:
    from config_rl import (CONFIG, DEFAULT_EVAL_EPISODES, MODEL_PATH, N_ORANGE,
                           RESULTS_DIR, SEED, experiment_id, experiment_spec)
    from peggle_env import PeggleEnv


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
                    seed=SEED + 10_000, results_dir=RESULTS_DIR, config=CONFIG):
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    model_path, results_dir = Path(model_path), Path(results_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Train the model first: {model_path}")
    training = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if training["experiment_id"] != experiment_id(config):
        raise ValueError("The model's board/physics/reward configuration differs from evaluation.")
    results_dir.mkdir(parents=True, exist_ok=True)
    model = DQN.load(model_path, device="cpu")
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
                    action = int(model.predict(obs, deterministic=True)[0]) if agent == "DQN" else int(rng.integers(config.n_actions))
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
        summary_rows.append({"agent": agent, "episodes": len(data), "successes": len(successes),
                             "success_rate_pct": 100 * data.success.mean(),
                             "orange_cleared_pct": data.orange_cleared_pct.mean(),
                             "avg_shots_success": successes.shots_used.mean() if len(successes) else np.nan,
                             "avg_shots_all": data.shots_used.mean(),
                             "mean_episode_reward": data.episode_reward.mean(),
                             "simulation_timeouts": int(data.simulation_timeouts.sum())})
    summary = pd.DataFrame(summary_rows)
    episodes_df.to_csv(results_dir / "evaluation_episodes.csv", index=False)
    pd.DataFrame(shots).to_csv(results_dir / "evaluation_shots.csv", index=False)
    summary.to_csv(results_dir / "evaluation_summary.csv", index=False)
    metadata = {"episodes_per_agent": episodes, "first_episode_seed": seed,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "training_timesteps": training["actual_timesteps"],
                "training_seed": training["seed"], "experiment_id": experiment_id(config),
                "experiment": experiment_spec(config),
                "policy": "deterministic DQN; uniform random over the same 21 actions",
                "note": "Fixed deterministic board: repeated DQN episodes have identical trajectories."
                        " Random seeds vary random actions, not boards. No generalization claim."}
    (results_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plt.close(plot_comparison(summary, results_dir / "dqn_vs_random.png"))
    return summary, episodes_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED + 10_000)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    summary, _ = evaluate_agents(args.model_path, args.episodes, args.seed, args.results_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
