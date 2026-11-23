"""Train and evaluate the separate PPO all-peg agent."""
import shutil
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from levels.config_rl import PPO_MAX_SHOTS, PPO_MODEL_PATH, PPO_RESULTS_DIR, PPO_SETTINGS
from levels.config_rl import PPO_TIMESTEPS, PPO_VERSIONS_DIR, RLConfig, SEED
from levels.peggle_env import PeggleEnv
from levels.evaluate import summarize


# Set before training and evaluation. Start with three useful shots from the
# preliminary greedy probe, then sweep unused angles across the board.
FIXED_ANGLES = (0, -36, -30, -60, 60, -48, 48, -24, 24, -12, 12,
                -54, 54, -42, 42, -18, 18, -6, 6, 30, 36)


def record_hit_ids(shot, before_obs, after_obs):
    # Free-ball observations store the 77 hit flags after the 77 color flags.
    ids = np.flatnonzero(after_obs[77:154] > before_obs[77:154])
    assert len(ids) == shot["new_pegs_cleared"]
    shot.update(hit_ids=" ".join(map(str, ids)), peg32_newly_hit=32 in ids,
                pegs_before=shot["pegs_remaining"] + len(ids))


def record_peg32_episode(episode, shots):
    hit = next((shot for shot in shots if shot["peg32_newly_hit"]), None)
    episode["peg32_left"] = hit is None
    episode["peg32_shot"] = None if hit is None else hit["shot"]
    episode["peg32_angle"] = None if hit is None else hit["angle_degrees"]
    episode["pegs_before_32"] = None if hit is None else hit["pegs_before"]
    episode["other_pegs_with_32"] = None if hit is None else hit["new_pegs_cleared"] - 1
    episode["peg32_early"] = hit is not None and hit["pegs_before"] > 10
    episode["empty_shots"] = sum(shot["new_pegs_cleared"] == 0 for shot in shots)


def play_episode(config, strategy, model=None, seed=SEED, action_seed=None, track_pegs=False):
    """Measure one game under the all-peg rules."""
    env = PeggleEnv(config)
    if track_pegs and not config.free_balls:
        raise ValueError("Peg tracking needs the free-ball observation.")
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    if strategy == "PPO sampled":
        torch.manual_seed(seed if action_seed is None else action_seed)
    orange_ids = " ".join(str(i) for i, peg in enumerate(env.pegs) if peg.isOrange)
    shots = []
    done = False
    try:
        while not done:
            if strategy in ("PPO", "PPO sampled"):
                action, _ = model.predict(observation, deterministic=strategy == "PPO")
            elif strategy == "Random":
                action = rng.integers(config.n_actions)
            elif strategy == "Straight down":
                action = int(np.argmin(abs(config.angles_degrees)))
            elif strategy == "Fixed sequence":
                angle = FIXED_ANGLES[len(shots) % len(FIXED_ANGLES)]
                action = int(np.where(config.angles_degrees == angle)[0][0])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            before_obs = observation
            observation, reward, terminated, truncated, info = env.step(int(action))
            shots.append({"strategy": strategy, "seed": seed, "action": int(action),
                          "shot": info["shots_used"], "angle_degrees": info["angle_degrees"],
                          "new_pegs_cleared": info["new_pegs_cleared"],
                          "pegs_cleared": info["pegs_cleared"], "pegs_remaining": info["pegs_remaining"],
                          "shots_remaining": info["shots_remaining"], "reward": reward,
                          "success": info["success"], "shot_timed_out": info["shot_timed_out"]})
            if config.free_balls:
                for key in ("starting_balls", "balls_remaining", "free_balls_earned",
                            "shots_earning_free_balls", "shot_free_balls", "shot_score", "total_score"):
                    shots[-1][key] = info[key]
                shots[-1]["action_seed"] = seed if action_seed is None else action_seed
                shots[-1]["hit_colors"] = " ".join(env.shot_hit_colors)
                shots[-1]["orange_remaining"] = info["orange_remaining"]
            if track_pegs:
                record_hit_ids(shots[-1], before_obs, observation)
            done = terminated or truncated
    finally:
        env.close()
    episode = {"strategy": strategy, "seed": seed, "total_pegs": config.n_pegs,
               "shot_limit": config.max_shots, "pegs_cleared": info["pegs_cleared"],
               "clearance_pct": 100 * info["pegs_cleared"] / config.n_pegs,
               "pegs_remaining": info["pegs_remaining"], "success": info["success"],
               "shots_used": info["shots_used"], "reward": sum(shot["reward"] for shot in shots),
               "distinct_angles": len({shot["action"] for shot in shots}),
               "simulation_timeouts": info["simulation_timeouts"]}
    if config.free_balls:
        episode.update(orange_peg_ids=orange_ids, action_seed=seed if action_seed is None else action_seed,
                       fixed_shot_limit=False)
        for key in ("starting_balls", "balls_remaining", "free_balls_earned",
                    "shots_earning_free_balls", "total_score"):
            episode[key] = info[key]
    if track_pegs:
        record_peg32_episode(episode, shots)
    return episode, shots




def evaluate_ppo(model, config, results_dir=PPO_RESULTS_DIR, sampled_games=30, random_games=100):
    """Run each deterministic policy once on the single starting state."""
    # Report the deterministic episode first. Sampling changes actions, not layouts.
    trials = [("PPO", 1), ("Random", random_games), ("Straight down", 1), ("Fixed sequence", 1)]
    if sampled_games:
        trials.append(("PPO sampled", sampled_games))
    summaries, episodes, shots = [], [], []
    for strategy, games in trials:
        game_rows, shot_rows = [], []
        for game in range(games):
            episode, trace = play_episode(config, strategy, model, seed=50_042 + game)
            game_rows.append(episode)
            shot_rows.extend(trace)
        summaries.append(summarize(game_rows, shot_rows))
        episodes.extend(game_rows)
        shots.extend(shot_rows)
    folder = results_dir
    os.makedirs(folder, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(folder, "evaluation_summary.csv"), index=False, na_rep="N/A")
    pd.DataFrame(episodes).to_csv(os.path.join(folder, "evaluation_episodes.csv"), index=False)
    pd.DataFrame(shots).to_csv(os.path.join(folder, "evaluation_shots.csv"), index=False)
    sequence_length = config.max_shots
    if config.free_balls:
        sequence_length = max(shot["shot"] for shot in shots if shot["strategy"] == "Fixed sequence")
    used = [FIXED_ANGLES[i % len(FIXED_ANGLES)] for i in range(sequence_length)]
    pd.DataFrame({"shot": range(1, sequence_length + 1), "angle_degrees": used}).to_csv(
        os.path.join(folder, "fixed_sequence.csv"), index=False)
    plot_metrics(summary, "strategy", [("clearance_pct", "Pegs cleared (%)"),
                 ("completion_pct", "Completion (%)"), ("avg_shots_success", "Shots in wins")],
                 os.path.join(folder, "agent_comparison.png"), "All-peg strategies on the same layout", bars=True)
    return summary


def greedy_probe(config, results_dir):
    """Try every action on the current hit mask. This is a feasibility probe only."""
    env = PeggleEnv(config)
    env.reset(seed=SEED)
    rows = []
    for shot in range(1, config.max_shots + 1):
        hits = [peg.isHit for peg in env.pegs]
        best_gain = -1
        for action in range(config.n_actions):
            env.reset(seed=SEED)
            env.shots_remaining = config.max_shots - shot + 1
            for peg, hit in zip(env.pegs, hits):
                peg.isHit, peg.isVisible = hit, not hit
            _, _, _, _, info = env.step(action)
            if info["new_pegs_cleared"] > best_gain:
                best_gain = info["new_pegs_cleared"]
                best_hits = [peg.isHit for peg in env.pegs]
                best_info = info
        rows.append({"shot_limit": config.max_shots, "shot": shot,
                     "angle_degrees": best_info["angle_degrees"], "new_pegs_cleared": best_gain,
                     "pegs_cleared": best_info["pegs_cleared"], "pegs_remaining": best_info["pegs_remaining"],
                     "success": best_info["success"]})
        for peg, hit in zip(env.pegs, best_hits):
            peg.isHit, peg.isVisible = hit, not hit
        if best_gain == 0 or best_info["success"]:
            break
    env.close()
    folder = results_dir
    os.makedirs(folder, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(os.path.join(folder, "greedy_probe.csv"), index=False)
    return data


class PPOEpisodeLog(BaseCallback):
    def __init__(self, winning_games_dir=None, experiment="PPO", track_pegs=False):
        super().__init__()
        self.rows = []
        self.winning_games_dir = winning_games_dir
        self.actions = []
        self.shots = []
        self.experiment = experiment
        self.track_pegs = track_pegs

    def _on_step(self):
        for info in self.locals["infos"]:
            if self.winning_games_dir is not None:
                self.actions.append(int(self.locals["actions"][0]))
                fields = ("angle_degrees", "new_pegs_cleared", "pegs_cleared", "pegs_remaining",
                          "free_balls_earned", "shot_free_balls", "shot_score", "total_score",
                          "balls_remaining", "success")
                self.shots.append(dict({key: info[key] for key in fields}, action=self.actions[-1],
                                       shot=info["shots_used"], reward=float(self.locals["rewards"][0])))
                if self.track_pegs:
                    before = self.locals["obs_tensor"][0].cpu().numpy()
                    after = info.get("terminal_observation", self.locals["new_obs"][0])
                    record_hit_ids(self.shots[-1], before, after)
            if "episode" in info:
                self.rows.append({"episode": len(self.rows) + 1, "timesteps": self.num_timesteps,
                                  "reward": info["episode"]["r"], "pegs_cleared": info["pegs_cleared"],
                                  "clearance_pct": 100 * info["pegs_cleared"] /
                                  (info["pegs_cleared"] + info["pegs_remaining"]),
                                  "success": info["success"], "shots_used": info["shots_used"],
                                  "simulation_timeouts": info["simulation_timeouts"]})
                if "free_balls_earned" in info:
                    for key in ("free_balls_earned", "shots_earning_free_balls", "total_score"):
                        self.rows[-1][key] = info[key]
                if self.winning_games_dir is not None:
                    if info["success"]:
                        self.save_win(info)
                    self.actions = []
                    self.shots = []
        return True

    def save_win(self, info):
        # The pre-shot observation retains the episode's colors after auto-reset.
        orange_flags = self.locals["obs_tensor"][0, :77].cpu().numpy()
        orange_ids = " ".join(str(i) for i in np.flatnonzero(orange_flags))
        os.makedirs(self.winning_games_dir, exist_ok=True)
        stem = os.path.join(self.winning_games_dir, f"win_{self.num_timesteps}")
        episode = dict(self.rows[-1], orange_peg_ids=orange_ids, experiment=self.experiment,
                       phase="training", checkpoint=stem + ".zip", training_seed=self.model.seed,
                       actions=" ".join(str(action) for action in self.actions))
        if self.track_pegs:
            record_peg32_episode(episode, self.shots)
        assert len(self.actions) == info["shots_used"]
        self.model.save(stem + ".zip")
        pd.DataFrame([episode]).to_csv(stem + ".csv", index=False)
        pd.DataFrame(self.shots).to_csv(stem + "_shots.csv", index=False)
        print(f"Saved training win at timestep {self.num_timesteps:,}.", flush=True)


def plot_metrics(data, x, metrics, path, title, bars=False, rolling=False):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.65 * len(metrics)))
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, metrics):
        values = data[key]
        if bars:
            bars_drawn = ax.bar(data[x], values.fillna(0), color="#2786a8")
            labels = [f"{v:.1f}" if pd.notna(v) else "N/A" for v in values]
            ax.bar_label(bars_drawn, labels=labels, padding=3, fontsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.margins(y=0.2)
            if values.isna().all() or (values.fillna(0) == 0).all():
                ax.set_ylim(0, 1)
        elif rolling:
            ax.plot(data[x], values, color="#2786a8", alpha=0.15)
            ax.plot(data[x], values.rolling(50, min_periods=1).mean(), color="#164665",
                    label="50-episode rolling mean")
            ax.legend(fontsize=8)
        else:
            ax.plot(data[x], values, marker="o", color="#2786a8")
            if values.isna().all():
                ax.text(0.5, 0.5, "N/A — no wins", ha="center", transform=ax.transAxes)
        if key in ("clearance_pct", "completion_pct", "completion_pct_validation", "sampled_wins"):
            ax.set_ylim(-2, 110 if bars else 102)
        elif key in ("pegs_cleared", "deterministic_pegs_cleared",
                     "avg_pegs_cleared_validation", "deterministic_pegs", "sampled_mean"):
            ax.set_ylim(0, 85 if bars else 80)
        elif key == "reward":
            ax.set_ylim(0, max(485, values.max() + 10))
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_title(title)
    if not bars:
        axes[-1].set_xlabel("Training timesteps (shots)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def checkpoint_score(row):
    if row["completion_pct"] > 0:
        return row["completion_pct"], -row["avg_shots_success"]
    return 0, row["clearance_pct"]


def validation_score(row):
    average_shots = row["avg_shots_success"] if row["wins"] else 0
    return row["completion_pct"], -average_shots, row["avg_pegs_cleared"]


def train_ppo(total_timesteps=PPO_TIMESTEPS, config=None, model_path=PPO_MODEL_PATH,
              results_dir=PPO_RESULTS_DIR, settings=None, resume_from=None, experiment="PPO",
              winning_games_dir=None, validate_checkpoint=None, checkpoint_interval=5000,
              track_pegs=False, validation_key=validation_score):
    """Train to a cumulative target, saving each checkpoint after a PPO update."""
    config = RLConfig(max_shots=PPO_MAX_SHOTS, goal="all") if config is None else config
    if config.goal != "all" or total_timesteps < 1:
        raise ValueError("PPO needs an all-peg goal and a positive training budget.")
    if winning_games_dir is not None and not config.free_balls:
        raise ValueError("Training replay logging needs the color flags in the free-ball observation.")
    model_path, folder = str(model_path), results_dir
    checkpoint_dir = os.path.splitext(model_path)[0] + "_checkpoints"
    checkpoint_log = os.path.join(folder, "checkpoint_evaluation.csv")
    if (os.path.exists(model_path) or os.path.exists(checkpoint_log)
            or os.path.exists(checkpoint_dir)):
        raise FileExistsError("Choose new PPO model and results paths for a fresh run.")
    settings = dict(PPO_SETTINGS if settings is None else settings)
    torch.set_num_threads(1)
    env = Monitor(PeggleEnv(config))
    if resume_from is None:
        model = PPO("MlpPolicy", env, seed=SEED, device="cpu", **settings)
    else:
        model = PPO.load(resume_from, env=env, device="cpu", **settings)
    if model.ent_coef != settings["ent_coef"]:
        env.close()
        raise ValueError("The requested PPO entropy coefficient was not applied.")
    start = model.num_timesteps
    if start >= total_timesteps:
        env.close()
        raise ValueError("The target must exceed the starting checkpoint's timesteps.")
    os.makedirs(checkpoint_dir)
    os.makedirs(folder, exist_ok=True)
    run_settings = {
        "experiment": experiment, "training_mode": "fresh" if resume_from is None else "continued",
        "starting_checkpoint": resume_from, "starting_timesteps": start,
        "requested_timesteps": total_timesteps, "requested_additional_timesteps": total_timesteps - start,
        "shot_limit": config.max_shots, "goal": config.goal, "input_size": config.n_pegs + 1,
        "seed": SEED, "device": "cpu", "policy": "MlpPolicy", "peg_reward": config.peg_reward,
        "shot_penalty": config.shot_penalty, "completion_bonus": config.completion_bonus,
        "empty_shot_penalty": config.empty_shot_penalty, **settings,
    }
    if config.free_balls:
        run_settings.update(free_balls=True, input_size=155, starting_balls=config.max_shots,
                            fixed_shot_limit=False,
                            ball_observation_scale=config.ball_observation_scale,
                            scoring="sum(base * post-hit Fever multiplier) * newly hit count",
                            deterministic_evaluation_seed=50_042)
    if config.n_actions != 21:
        run_settings.update(n_actions=config.n_actions, min_angle=config.min_angle,
                            max_angle=config.max_angle)
    if validate_checkpoint is not None:
        run_settings.update(endgame_bonus=config.endgame_bonus)
    if track_pegs:
        run_settings.update(early_peg_bonus=config.early_peg_bonus)
    # Save rollout-aligned snapshots near the requested interval.
    interval = max(settings["n_steps"], round(checkpoint_interval / settings["n_steps"]) * settings["n_steps"])
    log, scores = PPOEpisodeLog(winning_games_dir, experiment, track_pegs), []
    validation_rows = []
    try:
        while model.num_timesteps < total_timesteps:
            model.learn(min(interval, total_timesteps - model.num_timesteps),
                        reset_num_timesteps=False, callback=log)
            step = model.num_timesteps
            filename = f"checkpoint_{step}.zip"
            model.save(os.path.join(checkpoint_dir, filename))
            episode, shots = play_episode(config, "PPO", model, seed=50_042 if config.free_balls else SEED,
                                          track_pegs=track_pegs)
            pd.DataFrame(shots).to_csv(os.path.join(folder, f"shots_{step}.csv"), index=False)
            row = summarize([episode], shots)
            row.update(run_settings, timesteps=step, additional_timesteps=step-start,
                       deterministic_pegs_cleared=episode["pegs_cleared"],
                       deterministic_pegs_remaining=episode["pegs_remaining"],
                       deterministic_shots=episode["shots_used"], checkpoint=filename)
            scores.append(row)
            if validate_checkpoint is not None:
                checkpoint = os.path.join(checkpoint_dir, filename)
                # Validation sampling must not consume training's random numbers.
                with torch.random.fork_rng(devices=[]):
                    validation = validate_checkpoint(model, config, checkpoint, episode, shots)
                validation_rows.append(validation)
                pd.DataFrame(validation_rows).to_csv(
                    os.path.join(folder, "validation_checkpoints.csv"), index=False, na_rep="N/A")
            pd.DataFrame(scores).to_csv(checkpoint_log, index=False, na_rep="N/A")
            print(f"{experiment} at {step:,} timesteps: cleared {episode['pegs_cleared']}/{config.n_pegs} "
                  f"in {episode['shots_used']} shots, {row['distinct_angles']} angles.", flush=True)
    finally:
        env.close()

    if validation_rows:
        selected = max(validation_rows, key=validation_key)
        best = next(row for row in scores if row["timesteps"] == selected["timesteps"])
        for row in validation_rows:
            row["is_best"] = row is selected
        pd.DataFrame(validation_rows).to_csv(
            os.path.join(folder, "validation_checkpoints.csv"), index=False, na_rep="N/A")
    else:
        best = max(scores, key=checkpoint_score)
    shutil.copyfile(os.path.join(checkpoint_dir, best["checkpoint"]), model_path)
    for row in scores:
        row["is_best"] = row is best
    checkpoints = pd.DataFrame(scores)
    checkpoints.to_csv(checkpoint_log, index=False, na_rep="N/A")
    history = pd.DataFrame(log.rows, columns=["episode", "timesteps", "reward", "pegs_cleared",
                           "clearance_pct", "success", "shots_used", "simulation_timeouts"] +
                          (["free_balls_earned", "shots_earning_free_balls", "total_score"]
                           if config.free_balls else []))
    history.to_csv(os.path.join(folder, "training_episodes.csv"), index=False)
    run_settings.update(actual_timesteps=model.num_timesteps,
                        actual_additional_timesteps=model.num_timesteps-start,
                        best_timestep=best["timesteps"],
                        selected_checkpoint=os.path.join(checkpoint_dir, best["checkpoint"]))
    pd.DataFrame([run_settings]).to_csv(
        os.path.join(folder, "training_settings.csv"), index=False)
    history["completion_pct"] = 100 * history.success.astype(float)
    plot_metrics(history, "timesteps", [("reward", "Episode reward"), ("pegs_cleared", "Pegs cleared"),
                 ("completion_pct", "Completion (%)")], os.path.join(folder, "training_progress.png"),
                 "PPO training with sampled actions", rolling=True)
    plot_metrics(checkpoints, "timesteps", [("deterministic_pegs_cleared", "Pegs cleared / 77"),
                 ("completion_pct", "Completion (%)"), ("avg_shots_success", "Shots in wins")],
                 os.path.join(folder, "checkpoint_progress.png"), "Deterministic checkpoint performance")
    return PPO.load(model_path, device="cpu"), history


def sample_checkpoints(folder, games=100, first_seed=50_042):
    """Measure sampled consistency for each saved policy on the same fixed board."""
    settings = pd.read_csv(os.path.join(folder, "training_settings.csv")).iloc[0]
    config = RLConfig(goal="all", max_shots=int(settings.shot_limit),
                      free_balls=bool(settings.get("free_balls", False)),
                      n_actions=int(settings.get("n_actions", 21)))
    checkpoints = pd.read_csv(os.path.join(folder, "checkpoint_evaluation.csv"))
    model_file = str(settings.selected_checkpoint).replace("\\", "/")
    checkpoint_dir = os.path.dirname(model_file)
    summaries, episodes = [], []
    for row in checkpoints.itertuples():
        model = PPO.load(os.path.join(checkpoint_dir, row.checkpoint), device="cpu")
        game_rows, shots = [], []
        for game in range(games):
            episode, trace = play_episode(config, "PPO sampled", model, seed=first_seed + game)
            game_rows.append(episode)
            shots.extend(trace)
        summary = summarize(game_rows, shots)
        summary.update(timesteps=row.timesteps, best_pegs=max(r["pegs_cleared"] for r in game_rows))
        summaries.append(summary)
        for episode in game_rows:
            episode["timesteps"] = row.timesteps
        episodes.extend(game_rows)
        pd.DataFrame(summaries).to_csv(os.path.join(folder, "sampled_checkpoints.csv"), index=False, na_rep="N/A")
        print(f"Sampled checkpoint {row.timesteps}: mean {summary['avg_pegs_cleared']:.2f}, "
              f"wins {summary['wins']}/{games}", flush=True)
    pd.DataFrame(episodes).to_csv(os.path.join(folder, "sampled_checkpoint_episodes.csv"), index=False)
    return pd.DataFrame(summaries)


def compare_ppo_versions():
    """Compare saved versions using one deterministic and 100 sampled games each."""
    rows = []
    folders = [("Version 1", "models/ppo_version1.zip", "version1", 25088),
               ("Version 2", "models/ppo_version2.zip", "version2", 50176),
               ("Version 3", "models/ppo_version3_best.zip", "ppo_version3", 0)]
    for version, path, name, total in folders:
        folder = os.path.join(PPO_VERSIONS_DIR, name)
        model = PPO.load(path, device="cpu")
        if model.observation_space.shape != (78,) or model.action_space.n != 21:
            raise ValueError("The saved PPO model must use 78 inputs and 21 actions.")
        if version == "Version 3":
            settings = pd.read_csv(os.path.join(folder, "training_settings.csv")).iloc[0]
            total = int(settings.actual_timesteps)
        summary = pd.read_csv(os.path.join(folder, "evaluation_summary.csv")).set_index("strategy")
        deterministic, sampled = summary.loc["PPO"], summary.loc["PPO sampled"]
        episodes = pd.read_csv(os.path.join(folder, "evaluation_episodes.csv"))
        samples = episodes[episodes.strategy == "PPO sampled"]
        if len(samples) != 100 or not samples.shot_limit.eq(10).all():
            raise ValueError("Each version needs 100 sampled games with a ten-shot limit.")
        if samples.seed.tolist() != list(range(50_042, 50_142)):
            raise ValueError("PPO versions must use the same evaluation seeds.")
        rows.append({"version": version, "model_path": path, "results_dir": folder,
                     "main_change": {"Version 1": "Initial PPO", "Version 2": "Entropy 0.02",
                                     "Version 3": "Longer training, entropy 0.02"}[version],
                     "total_training_timesteps": total, "selected_timestep": model.num_timesteps,
                     "shot_limit": 10, "deterministic_pegs": int(deterministic.avg_pegs_cleared),
                     "deterministic_shots": int(deterministic.avg_shots_used),
                     "completion_pct": deterministic.completion_pct,
                     "avg_shots_success": deterministic.avg_shots_success,
                     "sampled_games": int(sampled.games), "sampled_mean": sampled.avg_pegs_cleared,
                     "sampled_clearance_pct": sampled.clearance_pct,
                     "sampled_remaining": sampled.avg_pegs_remaining,
                     "sampled_best": int(samples.pegs_cleared.max()), "sampled_wins": int(sampled.wins),
                     "sampled_avg_shots_success": sampled.avg_shots_success,
                     "sampled_min_shots_success": sampled.min_shots_success})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(PPO_VERSIONS_DIR, "version_comparison.csv"), index=False, na_rep="N/A")
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    for ax, column, label in zip(axes,
            ("deterministic_pegs", "sampled_mean", "sampled_wins"),
            ("Deterministic pegs", "Sampled mean pegs", "Wins / 100 sampled games")):
        bars = ax.bar(table.version, table[column], color="#2786a8")
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(ylabel=label, ylim=(0, 100 if column == "sampled_wins" else 82))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_title("PPO versions on fixed Level 1, ten-shot limit")
    fig.tight_layout()
    fig.savefig(os.path.join(PPO_VERSIONS_DIR, "version_comparison.png"), dpi=200)
    plt.close(fig)
    old, new = rows[1], rows[2]
    chosen = new if checkpoint_score(dict(new, clearance_pct=new["deterministic_pegs"])) > (
        checkpoint_score(dict(old, clearance_pct=old["deterministic_pegs"]))) else old
    pd.DataFrame([dict(chosen, menu_model_path=chosen["model_path"])]).to_csv(
        os.path.join(PPO_VERSIONS_DIR, "selected_model.csv"), index=False)
    larger = os.path.join(PPO_VERSIONS_DIR, "ppo_version3_15shots", "evaluation_summary.csv")
    if os.path.isfile(larger):
        pd.read_csv(larger).to_csv(os.path.join(PPO_VERSIONS_DIR,
                                  "larger_shot_limit_comparison.csv"), index=False)
    return table


def main():
    # Edit these values before starting another continuation experiment.
    run_name = "ppo_version3_new_run"
    total_timesteps = 150_000
    resume_from = "models/ppo_version2.zip"
    folder = os.path.join(PPO_VERSIONS_DIR, run_name)
    model_path = os.path.join("models", run_name + ".zip")
    config = RLConfig(goal="all", max_shots=10)
    model, _ = train_ppo(total_timesteps, config, model_path, folder,
        dict(PPO_SETTINGS, ent_coef=0.02), resume_from=resume_from, experiment=run_name)
    print(evaluate_ppo(model, config, folder, sampled_games=100).to_string(index=False))


if __name__ == "__main__":
    main()
