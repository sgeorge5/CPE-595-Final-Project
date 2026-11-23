"""Choose a DQN objective, then pretrained playback or a fresh training run."""
import os
import tempfile
import pandas as pd
from stable_baselines3 import DQN
from levels.config_rl import BEST_MODEL_PATH, FINAL_DQN_PATH, RLConfig
from levels.visualize_data import demonstrate


def load_game_mode(choice):
    if choice == "1":
        path = BEST_MODEL_PATH
        config = RLConfig(observation="compact")
    elif choice == "2":
        from levels.dqn_eight_shot_experiment import all_pegs_config
        path = FINAL_DQN_PATH
        config = all_pegs_config()
    else:
        raise ValueError("Please enter 1 or 2.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained checkpoint missing: {path}. Select fresh training to create a new model.")
    model = DQN.load(path, device="cpu")
    expected_inputs = 155 if choice == "1" else 78
    if model.observation_space.shape != (expected_inputs,) or model.action_space.n != 21:
        raise ValueError(f"Incompatible checkpoint {path}; expected {expected_inputs} inputs and 21 actions.")
    return model, config


def train_fresh(choice, output_root="models/retrained"):
    """An empty, uniquely named run prevents reuse of any previous trained weights."""
    from levels.train_dqn import train_model_one, train_model_two
    if choice not in ("1", "2"):
        raise ValueError("Please enter 1 or 2.")
    parent = os.path.join(output_root, "model" + choice)
    os.makedirs(parent, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="run_", dir=parent)
    print(f"Fresh training output: {run_dir}")
    if choice == "1":
        print("Training a new orange-only DQN from random network weights.")
        path = train_model_one(run_dir)
        config = RLConfig(observation="compact")
    else:
        from levels.dqn_eight_shot_experiment import all_pegs_config
        print("Training a new initial all-peg DQN, then demonstration-assisted DQN learning.")
        path = train_model_two(run_dir)
        config = all_pegs_config()
    print(f"New selected checkpoint: {path}")
    return DQN.load(path, device="cpu"), config, run_dir


def evaluate_new_model(model, config, run_dir):
    from levels.evaluate import run_policy, evaluate_model_two
    if config.goal == "orange":
        summary, games, shots, arrangements = run_policy(model, config=config)
        arrangements.to_csv(os.path.join(run_dir, "evaluation_arrangements.csv"), index=False)
    else:
        summary, games, shots = evaluate_model_two(model)
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "evaluation_summary.csv"), index=False)
    games.to_csv(os.path.join(run_dir, "evaluation_games.csv"), index=False)
    shots.to_csv(os.path.join(run_dir, "evaluation_shots.csv"), index=False)
    print("Actual results for this newly trained model")
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary


def main():
    while True:
        print("PEGGLE REINFORCEMENT LEARNING\n")
        print("1. DQN Model 1 - Clear all orange pegs")
        print("2. DQN Model 2 - Clear all 77 pegs")
        choice = input("Enter your choice: ").strip()
        if choice not in ("1", "2"):
            print("Please enter 1 or 2.")
            continue
        while True:
            print("1. Use existing pretrained model")
            print("2. Train a new model from scratch")
            print("3. Back to model selection")
            action = input("Enter your choice: ").strip()
            if action == "3":
                break
            if action not in ("1", "2"):
                print("Please enter 1, 2, or 3.")
                continue
            try:
                if action == "1":
                    model, config = load_game_mode(choice)
                else:
                    model, config, run_dir = train_fresh(choice)
                    evaluate_new_model(model, config, run_dir)
            except Exception as error:
                print(f"Unable to prepare DQN Model {choice}: {error}")
                continue
            demonstrate(model, config, "DQN Model " + choice)
            return


if __name__ == "__main__":
    main()
