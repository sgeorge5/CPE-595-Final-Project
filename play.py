"""Choose a saved Peggle agent and watch its shot-by-shot results in the terminal."""
import os

import pandas as pd
from stable_baselines3 import DQN, PPO

from levels.config_rl import (BEST_MODEL_PATH, PPO_MODEL_PATH, PPO_RESULTS_DIR,
                             PPO_VERSIONS_DIR, PPO_V4_RESULTS_DIR, PPO_V5_RESULTS_DIR,
                             PPO_V4_FURTHER_DIR, PPO_V4_IMPROVED_DIR, PPO_V4_SHOT_ORDER_DIR, RLConfig)
from levels.visualize_data import record_gameplay


def saved_ppo_files():
    for folder in (PPO_V4_SHOT_ORDER_DIR, PPO_V4_FURTHER_DIR, PPO_V4_IMPROVED_DIR,
                   PPO_V5_RESULTS_DIR, PPO_V4_RESULTS_DIR):
        selection_path = os.path.join(folder, "selected_model.csv")
        if os.path.isfile(selection_path):
            selected = pd.read_csv(selection_path).iloc[0]
            if (os.path.isfile(selected.menu_model_path)
                    and os.path.isfile(os.path.join(selected.results_dir, "training_settings.csv"))):
                return selected.menu_model_path, selected.results_dir
    selection_path = os.path.join(PPO_VERSIONS_DIR, "selected_model.csv")
    if os.path.isfile(selection_path):
        selected = pd.read_csv(selection_path).iloc[0]
        return selected.menu_model_path, selected.results_dir
    return "models/ppo_experiments/PPO-2.zip", "results/ppo_all_pegs/experiments/PPO-2"


def load_game_mode(choice, dqn_path=BEST_MODEL_PATH, ppo_path=None, ppo_results=None):
    if choice == "1":
        return DQN.load(dqn_path, device="cpu"), RLConfig(observation="compact")
    elif choice == "2":
        if ppo_path is None and ppo_results is None:
            ppo_path, ppo_results = saved_ppo_files()
        ppo_path = PPO_MODEL_PATH if ppo_path is None else ppo_path
        ppo_results = PPO_RESULTS_DIR if ppo_results is None else ppo_results
        settings = pd.read_csv(os.path.join(ppo_results, "training_settings.csv")).iloc[0]
        bonus = float(settings.get("completion_bonus", 100))
        config = RLConfig(max_shots=int(settings.shot_limit), goal="all",
                          empty_shot_penalty=float(settings.get("empty_shot_penalty", 0)),
                          free_balls=bool(settings.get("free_balls", False)),
                          n_actions=int(settings.get("n_actions", 21)),
                          endgame_bonus=bool(settings.get("endgame_bonus", False)),
                          ppo_completion_bonus=bonus,
                          early_peg_bonus=bool(settings.get("early_peg_bonus", False)))
        return PPO.load(ppo_path, device="cpu"), config
    raise ValueError("Please enter 1 or 2.")


def main():
    print("PEGGLE REINFORCEMENT LEARNING")
    print("1. DQN - Clear all orange pegs")
    print("2. PPO - Clear all pegs")
    choice = input("Enter your choice: ").strip()
    while choice not in ("1", "2"):
        choice = input("Please enter 1 or 2: ").strip()

    try:
        model, config = load_game_mode(choice)
    except FileNotFoundError as error:
        print(f"Saved model or settings are missing: {error.filename or error}")
        return
    # A new game draws orange colors using a fresh seed.
    shots, table = record_gameplay(model, seed=None, config=config)
    final = shots[-1]["info"]
    if choice == "1":
        print(table.to_string(index=False))
        print(f"DQN cleared {final['orange_cleared']}/{config.n_orange} orange pegs.")
    else:
        print(table.drop(columns=["new_orange_cleared", "orange_remaining"]).to_string(index=False))
        print(f"PPO cleared {final['pegs_cleared']}/{config.n_pegs} pegs.")
    result = "Win" if final["success"] else "Loss"
    if config.free_balls:
        print(f"Starting balls: {final['starting_balls']}; free balls earned: {final['free_balls_earned']}; "
              f"balls remaining: {final['balls_remaining']}; score: {final['total_score']:,}.")
        print(f"{result} in {final['shots_used']} total shots fired.")
    else:
        print(f"{result} in {final['shots_used']}/{config.max_shots} shots.")


if __name__ == "__main__":
    main()
