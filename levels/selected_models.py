"""Demonstrate the three saved all-peg policies used in the final report."""
import argparse
import os
from urllib.request import pathname2url
import webbrowser

import torch
import numpy as np
from stable_baselines3 import DQN, PPO

from levels.config_rl import RLConfig
from levels.dqn_all_pegs import export_replay
from levels.peggle_env import PeggleEnv


def load_model(choice):
    if choice == "ppo-v3":
        model = PPO.load("models/ppo_version3_best.zip", device="cpu")
        config = RLConfig(goal="all", max_shots=10, free_balls=False)
        label = "PPO V3"
    elif choice == "ppo-d":
        model = PPO.load("models/ppo_v4_improved_D_best.zip", device="cpu")
        config = RLConfig(goal="all", max_shots=10, free_balls=True, endgame_bonus=True)
        label = "PPO Improved D"
    elif choice == "dqn":
        model = DQN.load("models/dqn_all_pegs_best.zip", device="cpu")
        config = RLConfig(goal="all", max_shots=10, free_balls=False)
        label = "DQN Model 2"
    else:
        raise ValueError("Choose ppo-v3, ppo-d, or dqn.")
    assert model.observation_space.shape == ((155,) if config.free_balls else (78,))
    assert model.action_space.n == config.n_actions == 21
    return model, config, label


def play_game(choice):
    model, config, label = load_model(choice)
    env = PeggleEnv(config)
    obs, _ = env.reset(seed=50_042)
    orange_ids = " ".join(str(i) for i, peg in enumerate(env.pegs) if peg.isOrange)
    trace = []
    try:
        done = False
        while not done:
            action = int(model.predict(obs, deterministic=True)[0])
            obs, reward, terminal, truncated, info = env.step(action)
            trace.append(dict(action=action, shot=info["shots_used"], reward=reward,
                              **{k: info[k] for k in ("angle_degrees", "new_pegs_cleared", "pegs_cleared",
                                                     "pegs_remaining", "shots_remaining", "success")}))
            done = terminal or truncated
    finally:
        env.close()
    game = dict(model=label, strategy="Greedy", seed=50_042, orange_peg_ids=orange_ids,
                actions=" ".join(str(s["action"]) for s in trace),
                pegs_cleared=info["pegs_cleared"], shots_used=info["shots_used"], success=info["success"])
    if config.free_balls:
        game["free_balls_earned"] = info["free_balls_earned"]
    return game, trace, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["ppo-v3", "ppo-d", "dqn"])
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="print one greedy game without saving files")
    args = parser.parse_args()
    torch.set_num_threads(1)
    game, trace, config = play_game(args.model)
    print(f"{game['model']}: {game['pegs_cleared']}/77 pegs in {game['shots_used']} shots.")
    if config.free_balls:
        print(f"Free balls earned: {game['free_balls_earned']}.")
    if args.check_only:
        return
    # New demonstrations write only to their own directory.
    import pandas as pd
    from levels.visualize_data import record_gameplay, plot_gameplay
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt

    folder = os.path.join("results", "final_demonstrations", args.model)
    if not config.free_balls:
        path = export_replay(game, pd.DataFrame(trace), folder)
    else:
        replay, table = record_gameplay(None, game["seed"], config, actions=[s["action"] for s in trace])
        os.makedirs(folder, exist_ok=True)
        table.to_csv(os.path.join(folder, "shots.csv"), index=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        frames = [(i, int(j)) for i, shot in enumerate(replay)
                  for j in np.unique(np.linspace(0, len(shot['trajectory']) - 1, 10, dtype=int))]

        def draw(frame):
            i, j = frame
            shot = replay[i]
            plot_gameplay(shot["board"], shot["trajectory"], j, shot["info"]["angle_degrees"], i + 1, config, ax=ax)
            ax.set_title(game["model"] + "\n" + ax.get_title(), fontsize=9)

        animation = FuncAnimation(fig, draw, frames=frames, interval=120)
        path = os.path.join(folder, "replay.gif")
        animation.save(path, writer=PillowWriter(fps=8), dpi=90)
        plt.close(fig)
    print(path)
    if not args.no_browser:
        webbrowser.open("file:" + pathname2url(os.path.abspath(path)))


if __name__ == "__main__":
    main()
