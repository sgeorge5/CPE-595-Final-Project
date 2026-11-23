"""DQN Model 2 on the original ten-shot, all-peg task."""

import argparse
from contextlib import contextmanager
import glob
import os
import shutil
import tempfile
import time
from urllib.request import pathname2url
import webbrowser

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule

from levels.config_rl import RLConfig, SEED
from levels.peggle_env import PeggleEnv, random_peg_colors
from levels.evaluate import summarize
from levels.visualize_data import plot_gameplay, record_gameplay
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


RESULTS_DIR = "results/dqn_all_pegs"
MODEL_DIR = "models/dqn_all_pegs"
BEST_PATH = "models/dqn_all_pegs_best.zip"
PPO_PATH = "models/ppo_version3_best.zip"
STAGES = (25_000, 50_000, 100_000, 150_000, 200_000)
INTERVAL = 10_000
DECAY_STEPS = 60_000
VALIDATION_SEED = 45_042
VALIDATION_GAMES = 50
TEST_SEED = 91_042
TEST_GAMES = 100
DEMO_SEED = 50_042
EPSILON = 0.05
SETTINGS = {
    "learning_rate": 0.0003, "gamma": 0.99, "buffer_size": 20_000,
    "batch_size": 64, "learning_starts": 1000, "policy_kwargs": {"net_arch": [64, 64]},
    "train_freq": 4, "gradient_steps": 1, "target_update_interval": 500,
    "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.6,
}


def all_pegs_config():
    return RLConfig(goal="all", max_shots=10, free_balls=False, n_actions=21)


def write_csv(data, path):
    temporary = path + ".tmp"
    data.to_csv(temporary, index=False, na_rep="N/A")
    os.replace(temporary, path)


def append_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, mode="a", header=not os.path.isfile(path))


@contextmanager
def run_lock():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, ".run.lock"), "a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("A DQN Model 2 train or report command is already running.") from error
        yield


def play_game(model, method, seed):
    """Run one greedy, fixed-epsilon, or sampled-policy game on the shared board."""
    env = PeggleEnv(all_pegs_config())
    rng = np.random.default_rng(seed)
    if method == "PPO sampled":
        torch.manual_seed(seed)
    elif method not in ("Greedy", "Epsilon-greedy"):
        raise ValueError(f"Unknown evaluation method: {method}")
    observation, info = env.reset(seed=seed)
    colors = " ".join(str(i) for i, peg in enumerate(env.pegs) if peg.isOrange)
    trace = []
    try:
        done = False
        while not done:
            exploratory = method == "Epsilon-greedy" and rng.random() < EPSILON
            if exploratory:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(model.predict(observation, deterministic=method != "PPO sampled")[0])
            observation, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            trace.append(dict(strategy=method, seed=seed, action_seed=seed, action=action,
                              exploratory=exploratory, shot=info["shots_used"], reward=reward,
                              **{key: info[key] for key in ("angle_degrees", "new_pegs_cleared", "pegs_cleared",
                                 "pegs_remaining", "shots_remaining", "success", "shot_timed_out")}))
    finally:
        env.close()
    game = dict(strategy=method, seed=seed, action_seed=seed, orange_peg_ids=colors,
                total_pegs=77, shot_limit=10, epsilon=EPSILON if method == "Epsilon-greedy" else None,
                clearance_pct=100 * info["pegs_cleared"] / 77, reward=sum(s["reward"] for s in trace),
                actions=" ".join(str(s["action"]) for s in trace),
                angle_sequence=" ".join(str(int(s["angle_degrees"])) for s in trace),
                **{key: info[key] for key in ("pegs_cleared", "pegs_remaining", "shots_used", "success",
                                            "simulation_timeouts")})
    return game, trace


class TrainingMonitor(Monitor):
    """Keep the random board seed with its terminal transition for exact replay."""

    def __init__(self, env, seed):
        super().__init__(env)
        self.boards = np.random.default_rng(seed)

    def reset(self, **kwargs):
        if kwargs.get("seed") is None:
            kwargs["seed"] = int(self.boards.integers(1_000_000, 2**31))
        self.board_seed = kwargs["seed"]
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        if info["success"]:
            info["board_seed"] = self.board_seed
            info["orange_peg_ids"] = " ".join(str(i) for i, peg in enumerate(self.env.pegs) if peg.isOrange)
        return observation, reward, done, truncated, info


class TrainingLog(BaseCallback):
    def __init__(self, folder=None):
        super().__init__()
        self.rows, self.shots = [], []
        self.folder = folder
        self.run = time.time_ns()

    def _on_step(self):
        info = self.locals["infos"][0]
        self.shots.append(dict(action=int(self.locals["actions"][0]),
                               shot=info["shots_used"], reward=float(self.locals["rewards"][0]),
                               **{key: info[key] for key in ("angle_degrees", "new_pegs_cleared", "pegs_cleared",
                                  "pegs_remaining", "shots_remaining", "success")}))
        if "episode" in info:
            row = dict(timesteps=self.num_timesteps, reward=info["episode"]["r"],
                       **{key: info[key] for key in ("pegs_cleared", "shots_used", "success",
                                                   "simulation_timeouts")})
            self.rows.append(row)
            if self.folder and info["success"]:
                path = os.path.join(self.folder, f"run_{self.run}")
                game = dict(row, model="DQN Model 2", phase="training", strategy="Training exploration",
                            seed=info["board_seed"], orange_peg_ids=info["orange_peg_ids"],
                            actions=" ".join(str(shot["action"]) for shot in self.shots),
                            angle_sequence=" ".join(str(int(shot["angle_degrees"])) for shot in self.shots))
                assert len(self.shots) == row["shots_used"]
                append_csv([dict(shot, timesteps=self.num_timesteps) for shot in self.shots], path + "_shots.csv")
                append_csv([game], path + "_games.csv")
            self.shots = []
        return True


def checkpoint_step(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def checkpoint_paths():
    paths = [p for p in glob.glob(os.path.join(MODEL_DIR, "checkpoint_*.zip")) if ".pending." not in p]
    return sorted(paths, key=lambda p: (checkpoint_step(p), os.path.getmtime(p)))


def latest_model(env=None):
    paths = checkpoint_paths()
    for path in reversed(paths):
        try:
            model = DQN.load(path, env=env, device="cpu")
            if (model.num_timesteps != checkpoint_step(path) or model.observation_space.shape != (78,)
                    or model.action_space.n != 21 or model.seed != SEED):
                raise ValueError("Checkpoint does not match DQN Model 2.")
            if env is not None:
                model.load_replay_buffer(os.path.splitext(path)[0] + ".pkl")
            return model, path
        except Exception as error:
            print(f"Skipping invalid checkpoint {path}: {error}", flush=True)
    if paths:
        raise RuntimeError("No usable checkpoint found. Existing files have been preserved.")
    return None, None


def save_checkpoint(model, rows):
    stem = os.path.join(MODEL_DIR, f"checkpoint_{model.num_timesteps}")
    saved = stem
    retry = 1
    while os.path.exists(saved + ".zip"):
        saved = stem + f"_retry{retry}"
        retry += 1
    name = os.path.basename(saved)
    history = os.path.join(RESULTS_DIR, "history", name + ".csv")
    write_csv(pd.DataFrame(rows, columns=["timesteps", "reward", "pegs_cleared", "shots_used", "success",
                                          "simulation_timeouts"]), history)
    model.save_replay_buffer(saved + ".pending.pkl")
    model.save(saved + ".pending.zip")
    os.replace(saved + ".pending.pkl", saved + ".pkl")
    os.replace(saved + ".pending.zip", saved + ".zip")
    print(f"Saved {saved}.zip at {model.num_timesteps:,} timesteps.", flush=True)
    return saved + ".zip"


def evaluate(model, checkpoint, label, phase, first_seed, games, method):
    labels = dict(model=label, checkpoint=checkpoint, timesteps=model.num_timesteps, phase=phase)
    demo, demo_shots = play_game(model, "Greedy", DEMO_SEED)
    episodes, shots = [dict(demo, **labels)], [dict(shot, **labels) for shot in demo_shots]
    for seed in range(first_seed, first_seed + games):
        game, trace = play_game(model, method, seed)
        episodes.append(dict(game, **labels))
        shots.extend(dict(shot, **labels) for shot in trace)
    stats = summarize(episodes[1:], shots[len(demo_shots):])
    stats.update(labels, epsilon=EPSILON if method == "Epsilon-greedy" else None,
                 greedy_pegs=demo["pegs_cleared"], greedy_pct=demo["clearance_pct"],
                 greedy_shots=demo["shots_used"], greedy_win=demo["success"],
                 greedy_angles=demo["angle_sequence"])
    return stats, pd.DataFrame(episodes), pd.DataFrame(shots)


def validate(model, checkpoint):
    name = os.path.splitext(os.path.basename(checkpoint))[0]
    folder = os.path.join(RESULTS_DIR, "validation")
    score = os.path.join(folder, name + ".csv")
    if not os.path.isfile(score):
        stats, games, shots = evaluate(model, checkpoint, "DQN Model 2", "validation",
                                       VALIDATION_SEED, VALIDATION_GAMES, "Epsilon-greedy")
        write_csv(shots, os.path.join(folder, name + "_shots.csv"))
        write_csv(games, os.path.join(folder, name + "_games.csv"))
        write_csv(pd.DataFrame([stats]), score)
        print(f"Validation {model.num_timesteps:,}: greedy {stats['greedy_pegs']}/77, "
              f"epsilon mean {stats['avg_pegs_cleared']:.2f}/77, {stats['wins']}/{VALIDATION_GAMES} wins.",
              flush=True)


def collect_progress(checkpoint, target):
    step = checkpoint_step(checkpoint)
    write_csv(pd.DataFrame([dict(actual_timesteps=step, checkpoint=checkpoint, target_timesteps=target)]),
              os.path.join(RESULTS_DIR, "latest_checkpoint.csv"))
    paths = {checkpoint_step(path): path for path in checkpoint_paths() if checkpoint_step(path) <= step}
    for part, output in (("history", "training_episodes.csv"), ("validation", "checkpoint_evaluation.csv")):
        names = [os.path.splitext(os.path.basename(path))[0] + ".csv" for path in paths.values()]
        files = [os.path.join(RESULTS_DIR, part, name) for name in names]
        tables = [pd.read_csv(file) for file in files if os.path.isfile(file)]
        if tables:
            data = pd.concat(tables, ignore_index=True)
            if part == "history":
                data.insert(0, "episode", np.arange(1, len(data) + 1))
            write_csv(data, os.path.join(RESULTS_DIR, output))


def train(target):
    if not 4 <= target <= STAGES[-1] or target % SETTINGS["train_freq"]:
        raise ValueError("Choose a cumulative budget from 4 to 200,000, divisible by four.")
    if os.path.isfile(os.path.join(RESULTS_DIR, "selection.csv")):
        raise ValueError("The checkpoint choice is frozen for this finished experiment.")
    for folder in ("history", "validation", "wins_training"):
        os.makedirs(os.path.join(RESULTS_DIR, folder), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.set_num_threads(1)
    env = TrainingMonitor(PeggleEnv(all_pegs_config()), SEED)
    try:
        model, checkpoint = latest_model(env)
        if model is None:
            model = DQN("MlpPolicy", env, seed=SEED, device="cpu", **SETTINGS)
            print("Training a new DQN from scratch.", flush=True)
        else:
            print(f"Resuming {checkpoint} at {model.num_timesteps:,} timesteps.", flush=True)
            env.boards = np.random.default_rng(1_000_000 + SEED + model.num_timesteps)
            model.get_env().seed(1_000_000 + SEED + model.num_timesteps)
        plan = dict(seed=SEED, policy="MlpPolicy", inputs=78, actions=21, shot_limit=10, free_balls=False,
                    peg_reward=5, shot_penalty=2, completion_bonus=100, target_timesteps=max(target, model.num_timesteps),
                    exploration_decay_steps=DECAY_STEPS, checkpoint_interval=INTERVAL,
                    validation_seed=VALIDATION_SEED, validation_games=VALIDATION_GAMES,
                    test_seed=TEST_SEED, test_games=TEST_GAMES, evaluation_epsilon=EPSILON, **SETTINGS)
        write_csv(pd.DataFrame([plan]), os.path.join(RESULTS_DIR, "training_settings.csv"))
        if checkpoint is not None:
            collect_progress(checkpoint, plan["target_timesteps"])
            validate(model, checkpoint)
            collect_progress(checkpoint, plan["target_timesteps"])
        log = TrainingLog(os.path.join(RESULTS_DIR, "wins_training"))
        for end in sorted({target, *range(INTERVAL, target, INTERVAL), *(s for s in STAGES if s < target)}):
            if end <= model.num_timesteps:
                continue
            start = len(log.rows)
            # SB3 measures progress against this call's end. Keep the same absolute decay horizon.
            model.exploration_schedule = LinearSchedule(SETTINGS["exploration_initial_eps"],
                                                        SETTINGS["exploration_final_eps"], DECAY_STEPS / end)
            model.learn(end - model.num_timesteps, reset_num_timesteps=False, callback=log, log_interval=None)
            checkpoint = save_checkpoint(model, log.rows[start:])
            collect_progress(checkpoint, target)
            validate(model, checkpoint)
            collect_progress(checkpoint, target)
            if end in STAGES and end != target:
                print(f"Finished the {end:,}-step stage. Continuing to the next stage.", flush=True)
        return model.num_timesteps
    finally:
        env.close()


def smoke():
    torch.set_num_threads(1)
    env = TrainingMonitor(PeggleEnv(all_pegs_config()), SEED)
    try:
        model = DQN("MlpPolicy", env, seed=SEED, device="cpu", **dict(SETTINGS, learning_starts=16))
        first = [p.clone() for p in model.q_net.parameters()]
        model.learn(128, callback=TrainingLog())
        assert any(not torch.equal(before, after) for before, after in zip(first, model.q_net.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "smoke.zip")
            model.save(path)
            restored = DQN.load(path, device="cpu")
            assert restored.num_timesteps == 128 and restored.observation_space.shape == (78,)
            assert restored.action_space.n == 21
            game, _ = play_game(restored, "Greedy", DEMO_SEED)
            print(f"Smoke test passed: 128 training steps, changed weights, working checkpoint, "
                  f"greedy gameplay finished in {game['shots_used']} shots.", flush=True)
    finally:
        env.close()


def validation_rank(row):
    winning_shots = float(row["avg_shots_success"]) if row["wins"] else 0
    return (row["greedy_win"], row["wins"], row["greedy_pegs"], row["avg_pegs_cleared"],
            -row["greedy_shots"] if row["greedy_win"] else 0, -winning_shots)


def select_checkpoint():
    path = os.path.join(RESULTS_DIR, "selection.csv")
    if os.path.isfile(path):
        return pd.read_csv(path).iloc[0]
    model, latest = latest_model()
    plan = pd.read_csv(os.path.join(RESULTS_DIR, "training_settings.csv")).iloc[0]
    if model is None or model.num_timesteps < plan.target_timesteps:
        raise RuntimeError("Finish the planned training stage before freezing evaluation.")
    validate(model, latest)
    collect_progress(latest, plan.target_timesteps)
    scores_path = os.path.join(RESULTS_DIR, "checkpoint_evaluation.csv")
    scores = pd.read_csv(scores_path)
    best = max(scores.to_dict("records"), key=validation_rank)
    assert DQN.load(best["checkpoint"], device="cpu").num_timesteps == best["timesteps"]
    if os.path.isfile(BEST_PATH):
        with open(BEST_PATH, "rb") as saved, open(best["checkpoint"], "rb") as source:
            if saved.read() != source.read():
                raise FileExistsError(f"A different model already exists at {BEST_PATH}.")
    else:
        temporary = BEST_PATH + ".tmp"
        shutil.copyfile(best["checkpoint"], temporary)
        os.replace(temporary, BEST_PATH)
    scores["is_best"] = scores.checkpoint.eq(best["checkpoint"])
    write_csv(scores, scores_path)
    best.update(selected_model=BEST_PATH, actual_timesteps=model.num_timesteps,
                selection_rule="greedy full clear, epsilon-greedy wins, greedy clearance, mean clearance, shots in wins")
    write_csv(pd.DataFrame([best]), path)
    print(f"Selection frozen before the final test: {best['checkpoint']}.", flush=True)
    return pd.Series(best)


def read_or_evaluate(label, key, model, checkpoint, method):
    stem = os.path.join(RESULTS_DIR, key)
    summary_path = stem + "_summary.csv"
    if not os.path.isfile(summary_path):
        stats, games, shots = evaluate(model, checkpoint, label, "final", TEST_SEED, TEST_GAMES, method)
        write_csv(shots, stem + "_shots.csv")
        write_csv(games, stem + "_games.csv")
        write_csv(pd.DataFrame([stats]), summary_path)
    stats = pd.read_csv(summary_path).iloc[0]
    games = pd.read_csv(stem + "_games.csv")
    shots = pd.read_csv(stem + "_shots.csv")
    assert stats.checkpoint == checkpoint
    assert games[games.strategy == method].seed.tolist() == list(range(TEST_SEED, TEST_SEED + TEST_GAMES))
    return stats, games, shots


def replay_data(game, trace, verified=None):
    actions = [int(action) for action in str(game["actions"]).split()]
    key = tuple(actions)
    frames = None
    if verified is not None and key in verified:
        # With scoring disabled, different colors cannot affect the same action sequence.
        result = verified[key]
    else:
        frames, result = record_gameplay(None, int(game["seed"]), all_pegs_config(), actions=actions)
        if verified is not None:
            verified[key] = result
    colors = random_peg_colors(77, 25, np.random.default_rng(int(game["seed"])))
    assert " ".join(str(i) for i, color in enumerate(colors) if color == "orange") == game["orange_peg_ids"]
    assert len(result) == game["shots_used"] == len(trace)
    for field in ("shot", "angle_degrees", "new_pegs_cleared", "pegs_cleared", "pegs_remaining",
                  "shots_remaining", "reward", "success"):
        np.testing.assert_array_equal(result[field], trace[field])
    return frames


def export_replay(game, trace, folder, animate=True):
    frames = replay_data(game, trace)
    os.makedirs(folder, exist_ok=True)
    write_csv(pd.DataFrame([game]), os.path.join(folder, "game.csv"))
    write_csv(trace, os.path.join(folder, "shots.csv"))
    config = all_pegs_config()
    fig, ax = plt.subplots(figsize=(4.4, 4.3))

    def draw(shot_number, frame_index):
        shot = frames[shot_number]
        plot_gameplay(shot["board"], shot["trajectory"], frame_index,
                      shot["info"]["angle_degrees"], shot_number + 1, config, ax=ax)
        ax.set_title(ax.get_title().replace("PPO angle:", game["model"] + " angle:")
                     + "\n" + game["strategy"], fontsize=9)

    draw(0, len(frames[0]["trajectory"]) // 3)
    fig.savefig(os.path.join(folder, "beginning.png"), dpi=300)
    draw(len(frames) - 1, len(frames[-1]["trajectory"]) - 1)
    fig.savefig(os.path.join(folder, "final.png"), dpi=300)
    path = os.path.join(folder, "replay.gif")
    if animate:
        sequence = [(i, int(j)) for i, shot in enumerate(frames)
                    for j in np.unique(np.linspace(0, len(shot["trajectory"]) - 1, 10, dtype=int))]
        sequence.extend([sequence[-1]] * 9)
        animation = FuncAnimation(fig, lambda frame: draw(*frame), frames=sequence, interval=110)
        animation.save(path, writer=PillowWriter(fps=9), dpi=90)
    plt.close(fig)
    return path


def save_winning_replays():
    records, verified = [], {}
    folder = os.path.join(RESULTS_DIR, "wins_training")
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        if path.endswith("_shots.csv"):
            continue
        stem = path[:-10] if path.endswith("_games.csv") else path[:-4]
        shot_path = stem + "_shots.csv"
        episodes = pd.read_csv(path)
        shots = pd.read_csv(shot_path)
        for row in episodes.to_dict("records"):
            trace = shots[shots.timesteps == row["timesteps"]] if "timesteps" in shots else shots
            replay_data(row, trace, verified)
            records.append(dict(row, game_log=path, shot_log=shot_path))
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "validation", "*_games.csv")))
    paths.extend(os.path.join(RESULTS_DIR, name + "_games.csv") for name in ("dqn_final", "ppo_v3_matched"))
    for path in paths:
        shot_path = path.replace("_games.csv", "_shots.csv")
        episodes = pd.read_csv(path)
        shots = pd.read_csv(shot_path)
        for row in episodes[episodes.success].to_dict("records"):
            trace = shots[(shots.seed == row["seed"]) & (shots.strategy == row["strategy"])]
            replay_data(row, trace, verified)
            records.append(dict(row, game_log=path, shot_log=shot_path))
    columns = ["model", "phase", "timesteps", "strategy", "seed", "shots_used", "pegs_cleared", "game_log", "shot_log"]
    catalog = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    write_csv(catalog, os.path.join(RESULTS_DIR, "verified_wins.csv"))
    if records:
        dqn = catalog[catalog.model == "DQN Model 2"]
        for phase in ("training", "validation"):
            wins = dqn[dqn.phase == phase].sort_values(["shots_used", "timesteps", "seed"])
            if not wins.empty:
                best = wins.iloc[0]
                trace = pd.read_csv(best.shot_log)
                if phase == "training" and "timesteps" in trace:
                    trace = trace[trace.timesteps == best.timesteps]
                if phase == "validation":
                    trace = trace[(trace.seed == best.seed) & (trace.strategy == best.strategy)]
                export_replay(best, trace, os.path.join(RESULTS_DIR, "gameplay", phase + "_win"))


def save_figures(final):
    history = pd.read_csv(os.path.join(RESULTS_DIR, "training_episodes.csv"))
    checks = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint_evaluation.csv"))
    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True, layout="constrained")
        for ax, field, label in zip(axes, ("reward", "pegs_cleared"), ("Mean episode reward", "Mean pegs cleared")):
            ax.plot(history.timesteps / 1000, history[field].rolling(100, min_periods=1).mean(), color="#24557a")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[0].set_title("DQN training · 100-episode mean")
        axes[-1].set_xlabel("Training shots (thousands)")
        fig.savefig(os.path.join(RESULTS_DIR, "training_progress.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True, layout="constrained")
        steps = checks.timesteps / 1000
        axes[0].plot(steps, checks.greedy_pegs, ".-", label="Greedy")
        axes[0].plot(steps, checks.avg_pegs_cleared, ".-", label="Epsilon-greedy mean")
        axes[0].scatter(steps[checks.is_best], checks.greedy_pegs[checks.is_best], c="#a64b31", zorder=3)
        axes[0].set_ylabel("Pegs cleared / 77")
        axes[0].set_title("Checkpoint validation")
        axes[0].legend(fontsize=7)
        axes[1].plot(steps, checks.completion_pct, ".-", color="#24557a")
        axes[1].set_ylabel("Epsilon-greedy wins (%)")
        axes[1].set_ylim(-1, max(5, checks.completion_pct.max() + 2))
        axes[1].set_xlabel("Training shots (thousands)")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.savefig(os.path.join(RESULTS_DIR, "checkpoint_progress.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.65), layout="constrained")
        colors = ["#24557a", "#b45d3b"]
        for ax, field, title, limit in zip(axes.flat, ("greedy_pegs", "avg_pegs_cleared", "wins", "avg_shots_used"),
                                           ("Greedy pegs / 77", "Stochastic mean pegs", "Wins out of 100", "Mean shots"),
                                           (87, 87, max(2, final.wins.max() * 1.25 + 1), 12)):
            bars = ax.bar(["DQN", "PPO V3"], final[field], color=colors, width=0.65)
            ax.bar_label(bars, fmt="%g" if field in ("greedy_pegs", "wins") else "%.2f", fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.set_ylim(0, limit)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("DQN ε=0.05 and PPO sampled actions", fontsize=9)
        fig.savefig(os.path.join(RESULTS_DIR, "dqn_vs_ppo.png"), dpi=300)
        plt.close(fig)


def report():
    torch.set_num_threads(1)
    choice = select_checkpoint()
    model = DQN.load(choice.selected_model, device="cpu")
    ppo = PPO.load(PPO_PATH, device="cpu")
    assert ppo.num_timesteps == 105_472 and ppo.observation_space.shape == (78,) and ppo.action_space.n == 21
    dqn_stats, dqn_games, dqn_shots = read_or_evaluate("DQN Model 2", "dqn_final", model,
                                                      choice.selected_model, "Epsilon-greedy")
    ppo_stats, ppo_games, _ = read_or_evaluate("PPO V3", "ppo_v3_matched", ppo, PPO_PATH, "PPO sampled")
    keys = ["seed", "action_seed", "orange_peg_ids"]
    pd.testing.assert_frame_equal(dqn_games[keys], ppo_games[keys])
    assert ppo_stats.greedy_pegs == 70
    final = pd.DataFrame([dqn_stats, ppo_stats])
    write_csv(final, os.path.join(RESULTS_DIR, "matched_comparison.csv"))

    prior = pd.read_csv("results/ppo_121_angles/final_comparison.csv")
    best_ppo = prior[prior.experiment.eq("D")].iloc[0]
    overview = [dict(model=row.model, algorithm="DQN" if row.model == "DQN Model 2" else "PPO",
                     objective="All 77 pegs", scoring_and_free_balls=False, stochastic_method=row.strategy,
                     training_checkpoint=row.timesteps, greedy_pegs=row.greedy_pegs,
                     mean_pegs=row.avg_pegs_cleared, wins=row.wins, games=row.games,
                     evaluation="Matched seeds 91042–91141") for row in final.itertuples()]
    overview.append(dict(model="PPO Improved D", algorithm="PPO", objective="All 77 pegs", scoring_and_free_balls=True,
                         stochastic_method="PPO sampled", training_checkpoint=best_ppo.timesteps,
                         greedy_pegs=best_ppo.deterministic_pegs, mean_pegs=best_ppo.avg_pegs_cleared,
                         wins=best_ppo.wins, games=best_ppo.games, evaluation="Published seeds 90042–90141"))
    write_csv(pd.DataFrame(overview), os.path.join(RESULTS_DIR, "report_model_comparison.csv"))
    save_figures(final)

    for method, name in (("Greedy", "greedy"), ("Epsilon-greedy", "best_epsilon")):
        game = dqn_games[dqn_games.strategy == method].sort_values(
            ["pegs_cleared", "shots_used", "seed"], ascending=[False, True, True]).iloc[0]
        trace = dqn_shots[(dqn_shots.strategy == method) & (dqn_shots.seed == game.seed)]
        export_replay(game, trace, os.path.join(RESULTS_DIR, "gameplay", name))
    save_winning_replays()
    for row in final.itertuples():
        print(f"{row.model}: greedy {row.greedy_pegs}/77; {row.strategy} mean {row.avg_pegs_cleared:.2f}/77, "
              f"best {row.best_pegs_cleared}/77, {row.wins}/{row.games} wins.", flush=True)


def playback(show=True):
    model = DQN.load(BEST_PATH, device="cpu")
    game, trace = play_game(model, "Greedy", DEMO_SEED)
    game["model"] = "DQN Model 2"
    folder = os.path.join("results", "final_demonstrations", "dqn")
    path = export_replay(game, pd.DataFrame(trace), folder)
    print(f"Cleared {game['pegs_cleared']}/77 pegs in {game['shots_used']} shots. Replay: {path}")
    if show:
        webbrowser.open("file:" + pathname2url(os.path.abspath(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "report", "play"))
    parser.add_argument("--steps", type=int, default=100_000, help="cumulative training budget")
    parser.add_argument("--no-browser", action="store_true", help="save playback without opening a window")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "train":
        with run_lock():
            train(args.steps)
    elif args.command == "report":
        with run_lock():
            report()
    else:
        playback(show=not args.no_browser)


if __name__ == "__main__":
    main()
