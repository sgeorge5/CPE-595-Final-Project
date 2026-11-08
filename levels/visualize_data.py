"""Read PegglePy's level coordinates and plot the original and training boards."""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from levels.config_rl import CONFIG, PEGGLE_ROOT, RESULTS_DIR, ensure_directories
from levels.peggle_env import PeggleEnv


class _CoordinateUnpickler(pickle.Unpickler):
    """Read coordinate values while rejecting executable objects in level files."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Level data must contain primitive coordinates only.")


def original_level_data(level_name="Level 1.lvl"):
    """Read a bundled .lvl: it stores coordinates, NOT colors or hit labels.

    PegglePy's loadData centers the coordinate bounding box on the screen. We
    reproduce that translation and show both archived and centered values. Its
    normal loader creates blue, unhit Pegs before random color assignment.
    """
    level_dir = (PEGGLE_ROOT / "levels").resolve()
    path = (level_dir / level_name).resolve()
    if path.parent != level_dir or path.suffix != ".lvl":
        raise ValueError("Choose a bundled .lvl filename.")
    with path.open("rb") as level_file:
        raw_positions = _CoordinateUnpickler(level_file).load()
    positions = np.asarray(raw_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError("Expected a nonempty list of finite (x, y) coordinates.")
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    centered = positions + np.array([CONFIG.width / 2, CONFIG.height / 2]) - center
    return pd.DataFrame({"peg_id": np.arange(len(positions)),
                         "archive_x": positions[:, 0], "archive_y": positions[:, 1],
                         "x": centered[:, 0], "y": centered[:, 1],
                         "color_before_assignment": "blue", "initial_is_hit": False,
                         "color_stored_in_file": False})


def level_inventory():
    return pd.DataFrame([{"level": path.name, "peg_count": len(original_level_data(path.name))}
                         for path in sorted((PEGGLE_ROOT / "levels").glob("*.lvl"),
                                            key=lambda p: int(p.stem.split()[-1]))])


def plot_board(records=None, path=None):
    """Scatter plot with orange/blue pegs and game-style downward-positive y."""
    if records is None:
        env = PeggleEnv()
        env.reset()
        records = env.board_records()
        env.close()
    board = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 6))
    for color, display_color in (("orange", "#ed8b22"), ("blue", "#2786c4")):
        subset = board[board.color == color]
        ax.scatter(subset.x, subset.y, s=220, c=display_color, edgecolors="black", label=f"{color.title()} ({len(subset)})")
        for peg in subset.itertuples():
            ax.annotate(str(peg.peg_id), (peg.x, peg.y), ha="center", va="center", color="white", fontsize=9)
    ax.scatter([CONFIG.width / 2], [CONFIG.height / 25], marker="v", c="black", s=100, label="Launcher")
    ax.set(xlim=(0, CONFIG.width), ylim=(CONFIG.height, 0), xlabel="x (pixels)", ylabel="y (pixels)",
           title="Mid-stage fixed board: 3 orange targets and 7 blue obstacles")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=160)
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
                  config=CONFIG, ax=None):
    """Draw one recorded moment, using that moment's hit and visibility states."""
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
    for i, peg in enumerate(board):
        hit = frame["is_hit"][i]
        if peg["color"] == "orange" and not hit:
            orange_remaining += 1
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
    # Empty markers make a small legend without adding anything to the board.
    for label, color, marker in (("Orange", "#ed8b22", "o"),
                                 ("Blue", "#2786c4", "o"),
                                 ("Ball", "black", "o"),
                                 ("Hit peg", "black", "x")):
        ax.plot([], [], marker=marker, color=color, linestyle="none", label=label)
    ax.set(xlim=(0, config.width), ylim=(config.height, 0),
           xlabel="x (pixels)", ylabel="y (pixels)",
           title=f"DQN angle: {angle:+.0f}\N{DEGREE SIGN} from downward vertical\n"
                 f"Shots fired: {shot_number} | Orange remaining: {orange_remaining}")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.title.set_fontsize(10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=4, fontsize=8, frameon=False, handletextpad=0.3,
              columnspacing=1)
    fig.tight_layout()
    return fig


def export_data(output_dir=RESULTS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PeggleEnv()
    env.reset()
    board = pd.DataFrame(env.board_records())
    board.to_csv(output_dir / "board_data.csv", index=False)
    original = original_level_data()
    original.to_csv(output_dir / "original_level_data.csv", index=False)
    level_inventory().to_csv(output_dir / "level_inventory.csv", index=False)
    plt.close(plot_board(board, output_dir / "board_visualization.png"))
    plt.close(plot_original_board(original, output_dir / "original_board_visualization.png"))
    env.close()
    return board, original


if __name__ == "__main__":
    ensure_directories()
    board, original = export_data()
    print(f"Extracted {len(original)} original coordinates; exported the {len(board)}-peg RL board to {RESULTS_DIR}")
