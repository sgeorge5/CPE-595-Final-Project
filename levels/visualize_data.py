
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from levels.config_rl import (
    CONFIG, PEGGLE_ROOT, IMPROVED_RESULTS_DIR, SEED, load_level_coordinates
)
from levels.peggle_env import PeggleEnv
from levels.evaluate import model_observation

RENDER_EVERY_N_EPISODES = 10  # Show episode 1, then 10, 20, 30, ...
PROGRESS_EVERY_N_SHOTS = 1000
ANIMATION_FRAME_STRIDE = 8
ANIMATION_DELAY = 0.02  # Display speed only; never changes the physics timestep.


class LiveGame:
    """Draw recorded frames from the real environment, without stepping it."""

    def __init__(self):
        try:
            if str(matplotlib.get_backend()).lower() not in (
                    "tkagg", "qtagg", "qt5agg", "wxagg", "macosx"):
                plt.switch_backend("TkAgg")
            self.fig, (self.board_ax, self.info_ax) = plt.subplots(
                1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.4, 1]})
        except Exception as error:
            raise RuntimeError("Live graphics are unavailable. Choose headless execution. "
                               f"Matplotlib reported: {error}") from error
        self.active, self.paused = True, False
        self.stride = ANIMATION_FRAME_STRIDE
        self.fig.canvas.mpl_connect("close_event", self._closed)
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.19, wspace=0.24)
        plt.show(block=False)

    def _closed(self, event):
        if self.active:
            print("Visualization closed; execution continues headlessly.", flush=True)
        self.active, self.paused = False, False

    def _key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ("+", "="):
            self.stride = min(64, self.stride * 2)
        elif event.key == "-":
            self.stride = max(1, self.stride // 2)

    def panel(self, stats):
        if not self.active:
            return
        self.info_ax.clear()
        self.info_ax.axis("off")
        lines = [f"{key}: {value}" for key, value in stats.items()]
        self.info_ax.text(0, 1, "\n".join(lines), va="top", fontsize=11, linespacing=1.6,
                          transform=self.info_ax.transAxes, wrap=True)
        self.fig.suptitle("Peggle reinforcement learning", fontsize=16)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def stage(self, name, detail="No gameplay during this stage"):
        if self.active:
            self.board_ax.clear()
            self.board_ax.axis("off")
            self.board_ax.text(0.5, 0.5, detail, ha="center", va="center", wrap=True,
                               transform=self.board_ax.transAxes, fontsize=13)
            self.panel({"Stage": name, "Status": detail})

    def animate(self, shot, config, stats):
        if not self.active:
            return
        trajectory, board = shot["trajectory"], shot["board"]
        index = 0
        while self.active:
            while self.paused and self.active:
                plt.pause(0.05)
            if not self.active:
                break
            plot_gameplay(board, trajectory, index, shot["info"]["angle_degrees"],
                          shot["info"]["shots_used"], config, ax=self.board_ax, layout=False)
            frame_stats = dict(stats)
            frame = trajectory[index]
            cleared = sum(hit for peg, hit in zip(board, frame["is_hit"])
                          if config.goal == "all" or peg["color"] == "orange")
            target = config.n_pegs if config.goal == "all" else config.n_orange
            frame_stats["Pegs cleared"] = f"{cleared}/{target}"
            if index < len(trajectory) - 1:
                frame_stats["Episode reward"] = "N/A (shot in progress)"
                frame_stats["Status"] = "Shot in progress"
            self.panel(frame_stats)
            plt.pause(ANIMATION_DELAY)
            if index == len(trajectory) - 1:
                break
            index = min(index + self.stride, len(trajectory) - 1)

    def finish(self):
        if self.active:
            print("Close the visualization window to finish playback.", flush=True)
            plt.show(block=True)

    def close(self):
        self.active = False
        plt.close(self.fig)


class ObservedTrainingEnv(gym.Wrapper):
    """Save one real training shot before SB3 automatically resets the board."""

    def __init__(self, env, display):
        super().__init__(env)
        self.display = display
        self.episode = display.episodes
        self.last_shot = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode += 1
        self.unwrapped.record_trajectory = bool(self.display.viewer.active and
            (self.episode == 1 or self.episode % self.display.render_every == 0))
        self.observation = observation.copy()
        return observation, info

    def step(self, action):
        recording = self.unwrapped.record_trajectory and self.display.viewer.active
        self.unwrapped.record_trajectory = recording
        board = self.unwrapped.board_records() if recording else None
        before = self.observation
        observation, reward, done, truncated, info = self.env.step(action)
        self.last_shot = None
        if recording:
            self.last_shot = dict(board=board, trajectory=self.unwrapped.trajectory,
                                  info=dict(info), action=int(action), episode=self.episode,
                                  observation=before, next_observation=observation.copy())
        self.observation = observation.copy()
        return observation, reward, done, truncated, info


class TrainingDisplay(BaseCallback):
    """Read SB3's real actions/rewards; never select actions or change training."""

    def __init__(self, label, total_steps, viewer=None, render_every=None):
        super().__init__()
        self.label, self.total_steps, self.viewer = label, total_steps, viewer
        self.render_every = RENDER_EVERY_N_EPISODES if render_every is None else render_every
        if self.render_every < 1:
            raise ValueError("The rendering interval must be positive.")
        self.stage_name = "Standard DQN Training"
        self.completed_steps = self.episodes = self.successes = 0
        self.episode_reward = 0.0
        self.recent_clearance = []
        self.observed_env = None
        self.config = None

    def wrap(self, env):
        self.config = env.unwrapped.config
        self.episode_reward = 0.0
        if self.viewer is None:
            return env
        self.observed_env = ObservedTrainingEnv(env, self)
        return self.observed_env

    def stage(self, name, detail="No gameplay during this stage"):
        self.stage_name = name
        print(name, flush=True)
        if self.viewer is not None:
            self.viewer.stage(name, detail)
            self.viewer.panel({"Model": self.label, "Mode": "Visual training",
                               "Stage": name, "Status": detail})

    def _on_step(self):
        info = self.locals["infos"][0]
        action = int(self.locals["actions"][0])
        self.completed_steps += 1
        self.episode_reward += float(self.locals["rewards"][0])
        target = self.config.n_pegs if self.config.goal == "all" else self.config.n_orange
        cleared = info["pegs_cleared"] if self.config.goal == "all" else info["orange_cleared"]
        episode = self.episodes + 1
        reward = self.episode_reward
        if "episode" in info:
            self.episodes += 1
            self.successes += int(info["success"])
            self.recent_clearance = (self.recent_clearance + [cleared])[-50:]
            self.episode_reward = 0.0
        recent = (f"{np.mean(self.recent_clearance):.2f}/{target}"
                  if self.recent_clearance else "N/A")
        stats = {"Model": self.label, "Mode": "Visual training" if self.viewer else "Headless training",
                 "Stage": self.stage_name, "Episode": episode,
                 "Timesteps": f"{self.num_timesteps:,} (current DQN)",
                 "Training progress": f"{self.completed_steps:,} / {self.total_steps:,} shots",
                 "Current shot / angle": f"{info['shots_used']} / {info['angle_degrees']:+.0f} degrees",
                 "Pegs cleared": f"{cleared}/{target}", "Shots remaining": info["shots_remaining"],
                 "Episode reward": f"{reward:.2f}",
                 "Exploration epsilon": f"{self.model.exploration_rate:.3f}",
                 "Successful episodes": self.successes, "Recent average clearance": recent,
                 "Status": ("Success" if info["success"] else "Shot budget exhausted")
                     if self.locals["dones"][0] else "Training"}
        if self.num_timesteps <= self.model.learning_starts:
            stats["Exploration epsilon"] += " (random warm-up)"
        if self.viewer is not None and self.viewer.active:
            shot = self.observed_env.last_shot
            if shot is not None:
                assert shot["action"] == action
                self.viewer.animate(shot, self.config, stats)
            elif self.completed_steps % 50 == 0 or "episode" in info:
                self.viewer.stage(self.stage_name, "Training continues without animation")
                self.viewer.panel(stats)
        if self.completed_steps == 1 or self.completed_steps % PROGRESS_EVERY_N_SHOTS == 0:
            print(" | ".join(f"{key}: {value}" for key, value in stats.items()
                            if key not in ("Mode", "Status")), flush=True)
        return True


def play_policy(model, config, label, seed=50042, viewer=None):
    """Live or headless deterministic inference; no files and no training."""
    env = PeggleEnv(config)
    observation, _ = env.reset(seed=seed)
    done, total_reward = False, 0.0
    rows = []
    print(label)
    try:
        while not done:
            env.record_trajectory = viewer is not None and viewer.active
            board = env.board_records() if env.record_trajectory else None
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(int(action))
            done = done or truncated
            total_reward += reward
            cleared = info["pegs_cleared"] if config.goal == "all" else info["orange_cleared"]
            target = config.n_pegs if config.goal == "all" else config.n_orange
            rows.append(dict(action=int(action), reward=reward, **info))
            print(f"Shot {info['shots_used']}: Angle {info['angle_degrees']:+.0f} degrees | "
                  f"Pegs cleared {cleared}/{target} | Shots remaining {info['shots_remaining']}", flush=True)
            if viewer is not None and viewer.active:
                stats = {"Model": label, "Mode": "Visual pretrained / selected-policy playback",
                         "Current shot / angle": f"{info['shots_used']} / {info['angle_degrees']:+.0f} degrees",
                         "Pegs cleared": f"{cleared}/{target}", "Shots remaining": info["shots_remaining"],
                         "Episode reward": f"{total_reward:.2f}",
                         "Status": ("Success" if info["success"] else "Shot budget exhausted") if done else "Playing"}
                viewer.animate(dict(board=board, trajectory=env.trajectory, info=info), config, stats)
        print(f"GAME COMPLETE | Pegs cleared: {cleared}/{target} | "
              f"Shots used: {info['shots_used']} | Success: {info['success']}", flush=True)
        if viewer is not None:
            viewer.finish()
        return pd.DataFrame(rows)
    finally:
        env.close()


def original_level_data(level_name="Level 1.lvl"):

    positions = load_level_coordinates(level_name, center=False)
    centered = load_level_coordinates(level_name, width=CONFIG.width, height=CONFIG.height)
    return pd.DataFrame({"peg_id": np.arange(len(positions)),
                         "archive_x": positions[:, 0], "archive_y": positions[:, 1],
                         "x": centered[:, 0], "y": centered[:, 1],
                         "color_before_assignment": "blue", "initial_is_hit": False,
                         "color_stored_in_file": False})


def level_inventory():
    level_dir = os.path.join(PEGGLE_ROOT, "levels")
    filenames = [name for name in os.listdir(level_dir) if name.endswith(".lvl")]
    filenames.sort(key=lambda name: int(os.path.splitext(name)[0].split()[-1]))
    return pd.DataFrame([{"level": name, "peg_count": len(original_level_data(name))}
                         for name in filenames])


def plot_board(records=None, path=None, config=CONFIG, ax=None):
    if records is None:
        env = PeggleEnv(config)
        env.reset(seed=SEED)
        records = env.board_records()
        env.close()
    board = pd.DataFrame(records)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    for color, display_color in (("orange", "#ed8b22"), ("blue", "#2786c4")):
        subset = board[(board.color == color) & board.is_visible]
        ax.scatter(subset.x, subset.y, s=90,
                   c=display_color, edgecolors="black", linewidths=0.5,
                   label=f"{color.title()} ({len(subset)})")
        hit = subset[subset.is_hit]
        ax.scatter(hit.x, hit.y, marker="x", s=30, c="black")
    ax.scatter([config.width / 2], [config.height / 25], marker="v", c="black", s=70, label="Launcher")
    ax.set(xlim=(0, config.width), ylim=(config.height, 0), xlabel="x (pixels)", ylabel="y (pixels)",
           title=f"{len(board)}-peg board: {(board.color == 'orange').sum()} orange targets")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=160)
    return fig


def plot_arrangements(first, second, path=None, config=CONFIG):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for number, (board, ax) in enumerate(zip((first, second), axes), start=1):
        plot_board(board, config=config, ax=ax)
        ax.set_title(f"Arrangement {number}: {config.n_orange} orange / {config.n_pegs} pegs")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200)
    return fig


def plot_original_board(data=None, path=None):
    data = original_level_data() if data is None else data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data.x, data.y, c="#2786c4", s=45, label="Pegs before runtime color assignment")
    ax.set(xlim=(0, CONFIG.width), ylim=(CONFIG.height, 0), xlabel="Centered x (pixels)",
           ylabel="Centered y (pixels)", title=f"Original PegglePy Level 1: {len(data)} coordinate records")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=160)
    return fig


def plot_gameplay(board, trajectory, frame_index, angle, shot_number,
                  config=CONFIG, ax=None, layout=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure
        ax.clear()
    frame = trajectory[frame_index]
    path = trajectory[:frame_index + 1]
    ax.plot([point["x"] for point in path], [point["y"] for point in path],
            color="#555555", linewidth=1, zorder=1)

    orange_remaining = 0
    new_orange_hits = 0
    for i, peg in enumerate(board):
        hit = frame["is_hit"][i]
        if peg["color"] == "orange" and not hit:
            orange_remaining += 1
        if peg["color"] == "orange" and hit and not peg["is_hit"]:
            new_orange_hits += 1
        if not frame["is_visible"][i]:
            continue
        color = "#ed8b22" if peg["color"] == "orange" else "#2786c4"
        # Hit pegs stay solid until removed: keep their color and mark an X.
        ax.add_patch(plt.Circle((peg["x"], peg["y"]), peg["radius"],
                               facecolor=color, edgecolor="black", linewidth=0.7,
                               zorder=3))
        if hit:
            ax.plot(peg["x"], peg["y"], marker="x", color="black",
                    markersize=5, markeredgewidth=1, zorder=4)
    if frame["ball_alive"]:
        ax.add_patch(plt.Circle((frame["x"], frame["y"]), config.ball_radius,
                               facecolor="black", zorder=5))
    ax.plot(config.width / 2, config.height / 25, marker="v", color="black",
            markersize=5)
    for label, color, marker in (("Orange", "#ed8b22", "o"),
                                 ("Blue", "#2786c4", "o"),
                                 ("Ball", "black", "o"),
                                 ("Hit peg", "black", "x")):
        ax.plot([], [], marker=marker, color=color, linestyle="none", label=label)
    if config.goal == "all":
        cleared = sum(frame["is_hit"])
        new_hits = sum(hit and not peg["is_hit"]
                       for hit, peg in zip(frame["is_hit"], board))
        status = " | Complete" if cleared == len(board) else ""
        title = (f"DQN angle: {angle:+.0f}° | Shot {shot_number}\n"
                 f"New hits: {new_hits} | Total cleared: {cleared}\n"
                 f"Pegs left: {len(board) - cleared} | Shots left: "
                 f"{config.max_shots - shot_number}{status}")
        if config.free_balls:
            version = 5 if config.n_actions == 41 else 4
            agent = "PPO (121 angles)" if config.n_actions == 121 else f"PPO Version {version}"
            title = (f"{agent} | Shot {frame['shots_used']} | Angle {angle:+.0f}°\n"
                     f"New hits: {new_hits} | Cleared: {cleared}/{len(board)}\n"
                     f"Shot score: {frame['shot_score']:,} | Free balls: {frame['shot_free_balls']}\n"
                     f"Balls left: {frame['balls_remaining']} | Total free balls: {frame['free_balls_earned']}\n"
                     f"Total shots fired: {frame['shots_used']}{status}")
    else:
        title = (f"DQN angle: {angle:+.0f}° from downward vertical\n"
                 f"Shot {shot_number} | New orange hits: {new_orange_hits}\n"
                 f"Orange left: {orange_remaining} | Shots left: {config.max_shots - shot_number}")
    ax.set(xlim=(0, config.width), ylim=(config.height, 0),
           xlabel="x (pixels)", ylabel="y (pixels)",
           title=title)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.title.set_fontsize(10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=4, fontsize=8, frameon=False, handletextpad=0.3,
              columnspacing=1)
    if layout:
        fig.tight_layout()
    return fig


def record_gameplay(model, seed, config=CONFIG, actions=None, orange_peg_ids=None):
    env = PeggleEnv(config, record_trajectory=True)
    options = None if orange_peg_ids is None else {"orange_peg_ids": orange_peg_ids}
    obs, _ = env.reset(seed=seed, options=options)
    shots, rows = [], []
    done = False
    while not done:
        board = env.board_records()
        if actions is not None:
            action = actions[len(shots)]
        elif config.goal == "all":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action, _ = model.predict(model_observation(env, model), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        shots.append({"board": board, "trajectory": env.trajectory, "info": info})
        rows.append({"shot": info["shots_used"], "angle_degrees": info["angle_degrees"],
                     "new_orange_cleared": info["new_orange_cleared"],
                     "orange_remaining": info["orange_remaining"],
                     "shots_remaining": info["shots_remaining"], "reward": reward})
        if config.goal == "all":
            for key in ("new_pegs_cleared", "pegs_cleared", "pegs_remaining", "success"):
                rows[-1][key] = info[key]
        if config.free_balls:
            for key in ("starting_balls", "balls_remaining", "free_balls_earned",
                        "shots_earning_free_balls", "shot_free_balls", "shot_score", "total_score"):
                rows[-1][key] = info[key]
        done = terminated or truncated
    env.close()
    return shots, pd.DataFrame(rows)


def demonstrate(model, config, label, seed=50042, show=True, output_dir=None):
    import tempfile
    import webbrowser
    from urllib.request import pathname2url
    from matplotlib.animation import FuncAnimation, PillowWriter

    replay, table = record_gameplay(model, seed, config)
    final = replay[-1]["info"]
    cleared = final["pegs_cleared"] if config.goal == "all" else final["orange_cleared"]
    target = 77 if config.goal == "all" else 25
    print(table.to_string(index=False))
    print(f"{label}: {cleared}/{target} pegs, {final['shots_used']} shots, success {final['success']}.")
    if output_dir is None:
        os.makedirs("results/source_playback", exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix="game_", dir="results/source_playback")
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, "shots.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    frames = [(i, int(j)) for i, shot in enumerate(replay)
              for j in np.unique(np.linspace(0, len(shot["trajectory"]) - 1, 10, dtype=int))]

    def draw(frame):
        i, j = frame
        shot = replay[i]
        plot_gameplay(shot["board"], shot["trajectory"], j, shot["info"]["angle_degrees"], i + 1, config, ax=ax)
        ax.set_title(label + "\n" + ax.get_title(), fontsize=9)

    animation = FuncAnimation(fig, draw, frames=frames, interval=120)
    path = os.path.join(output_dir, "replay.gif")
    animation.save(path, writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    print(path)
    if show:
        webbrowser.open("file:" + pathname2url(os.path.abspath(path)))
    return replay, table


def export_data(output_dir=IMPROVED_RESULTS_DIR, seed=SEED, config=CONFIG):
    os.makedirs(output_dir, exist_ok=True)
    env = PeggleEnv(config)
    env.reset(seed=seed)
    board = pd.DataFrame(env.board_records())
    env.reset()  
    second = pd.DataFrame(env.board_records())
    board.to_csv(os.path.join(output_dir, "board_data.csv"), index=False)
    second.to_csv(os.path.join(output_dir, "board_arrangement_2.csv"), index=False)
    original = original_level_data()
    original.to_csv(os.path.join(output_dir, "original_level_data.csv"), index=False)
    level_inventory().to_csv(os.path.join(output_dir, "level_inventory.csv"), index=False)
    plt.close(plot_board(board, os.path.join(output_dir, "board_visualization.png"), config=config))
    plt.close(plot_arrangements(board, second, os.path.join(output_dir, "random_arrangements.png"), config))
    plt.close(plot_original_board(original, os.path.join(output_dir, "original_board_visualization.png")))
    env.close()
    return board, original


if __name__ == "__main__":
    board, original = export_data()
    print(f"Extracted {len(original)} original coordinates; exported the {len(board)}-peg RL board to {IMPROVED_RESULTS_DIR}")
