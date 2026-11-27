import os
import argparse
import tempfile
import pandas as pd
from stable_baselines3 import DQN
from levels.config_rl import BEST_MODEL_PATH, FINAL_DQN_PATH, RLConfig
from levels.visualize_data import LiveGame, TrainingDisplay, play_policy

STANDARD_ALL_PATH = "models/dqn_all_pegs_best.zip"
# DQN evaluation episode 78: 25/25 in seven shots (evaluation_episodes.csv).
ORANGE_PRESENTATION_SEED = 50119
MODEL_LABELS = {"1": "Standard DQN - Clear orange pegs",
                "2": "Standard DQN - Clear all 77 pegs",
                "3": "Demonstration-assisted DQN - Clear all 77 pegs"}


def load_game_mode(choice):
    if choice == "1":
        path = BEST_MODEL_PATH
        config = RLConfig(observation="compact")
    elif choice == "2":
        path = STANDARD_ALL_PATH
        config = RLConfig(goal="all", max_shots=10, free_balls=False, n_actions=21)
    elif choice == "3":
        from levels.demonstation_assisted_dqn import all_pegs_config
        path = FINAL_DQN_PATH
        config = all_pegs_config()
    else:
        raise ValueError("Please enter 1, 2, or 3.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained checkpoint missing: {path}. Select fresh training to create a new model.")
    model = DQN.load(path, device="cpu")
    expected_inputs = 155 if choice == "1" else 78
    if model.observation_space.shape != (expected_inputs,) or model.action_space.n != 21:
        raise ValueError(f"Incompatible checkpoint {path}; expected {expected_inputs} inputs and 21 actions.")
    return model, config


def train_fresh(choice, output_root="models/retrained", viewer=None):
    from levels.train_dqn import train_model_one, train_standard_all_pegs, train_model_two
    if choice not in ("1", "2", "3"):
        raise ValueError("Please enter 1, 2, or 3.")
    parent = os.path.join(output_root, "model" + choice)
    os.makedirs(parent, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="run_", dir=parent)
    print(f"Fresh training output: {run_dir}")
    budget = {"1": 50_000, "2": 100_000, "3": 115_000}[choice]
    display = TrainingDisplay(MODEL_LABELS[choice], budget, viewer=viewer)
    if choice == "1":
        print("Training a new orange-only DQN from random network weights.")
        path = train_model_one(run_dir, display=display)
        config = RLConfig(observation="compact")
    elif choice == "2":
        print("Training a new standard all-peg DQN from random weights, without demonstrations.")
        path = train_standard_all_pegs(run_dir, display=display)
        config = RLConfig(goal="all", max_shots=10, free_balls=False, n_actions=21)
    else:
        from levels.demonstation_assisted_dqn import all_pegs_config
        print("Training a new initial all-peg DQN, then demonstration-assisted DQN learning.")
        path = train_model_two(run_dir, display=display)
        config = all_pegs_config()
    print(f"New selected checkpoint: {path}")
    return DQN.load(path, device="cpu"), config, run_dir


def evaluate_new_model(model, config, run_dir, assisted=False):
    from levels.evaluate import run_policy, evaluate_model_two, evaluate_standard_all_pegs
    if config.goal == "orange":
        summary, games, shots, arrangements = run_policy(model, config=config)
        arrangements.to_csv(os.path.join(run_dir, "evaluation_arrangements.csv"), index=False)
    elif assisted:
        summary, games, shots = evaluate_model_two(model)
    else:
        summary, games, shots = evaluate_standard_all_pegs(model)
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "evaluation_summary.csv"), index=False)
    games.to_csv(os.path.join(run_dir, "evaluation_games.csv"), index=False)
    shots.to_csv(os.path.join(run_dir, "evaluation_shots.csv"), index=False)
    print("Actual results for this newly trained model")
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary


def select_orange_arrangement():
    while True:
        print("\nSELECT MODEL 1 PLAYBACK ARRANGEMENT\n")
        print("1. Random orange peg arrangement")
        print("2. Previously verified winning arrangement (seed 50119)")
        print("3. Enter a custom seed")
        print("4. Back")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            return choice, None
        if choice == "2":
            return choice, ORANGE_PRESENTATION_SEED
        if choice == "3":
            try:
                seed = int(input("Enter a nonnegative integer seed: ").strip())
            except ValueError:
                print("Please enter a nonnegative integer seed.")
                continue
            if seed < 0:
                print("Please enter a nonnegative integer seed.")
                continue
            return choice, seed
        if choice == "4":
            return choice, None
        print("Please enter 1, 2, 3, or 4.")


def main(seed=None, winning_arrangement=False):
    while True:
        print("PEGGLE REINFORCEMENT LEARNING\n")
        print("1. Standard DQN - Clear orange pegs")
        print("2. Standard DQN - Clear all 77 pegs")
        print("3. Demonstration-assisted DQN - Clear all 77 pegs")
        choice = input("Enter your choice: ").strip()
        if choice not in ("1", "2", "3"):
            print("Please enter 1, 2, or 3.")
            continue
        while True:
            print("\nSELECT MODEL SOURCE\n")
            print("1. Use existing pretrained model")
            print("2. Train a new model from scratch")
            print("3. Back to model selection")
            action = input("Enter your choice: ").strip()
            if action == "3":
                break
            if action not in ("1", "2"):
                print("Please enter 1, 2, or 3.")
                continue
            while True:
                if action == "1":
                    print("\nSELECT PLAYBACK MODE\n")
                    print("1. Visual playback - Watch the agent play live")
                    print("2. Headless playback - Run without graphics")
                else:
                    print("\nSELECT TRAINING MODE\n")
                    print("1. Visual training - Watch the agent learn live")
                    print("2. Headless training - Train without graphics (fastest)")
                print("3. Back")
                mode = input("Enter your choice: ").strip()
                if mode == "3":
                    break
                if mode not in ("1", "2"):
                    print("Please enter 1, 2, or 3.")
                    continue
                playback_seed, arrangement_label = None, None
                if choice == "1":
                    if winning_arrangement:
                        arrangement, playback_seed = "2", ORANGE_PRESENTATION_SEED
                    elif seed is not None:
                        arrangement, playback_seed = "3", seed
                    else:
                        arrangement, playback_seed = select_orange_arrangement()
                        if arrangement == "4":
                            continue  # Back to playback/training mode, before starting anything.
                    if arrangement == "1":
                        arrangement_label = "Random orange peg arrangement"
                    elif arrangement == "2":
                        arrangement_label = "Selected demonstration case (previously successful)"
                        print("Previously verified winning arrangement: the original pretrained Model 1 "
                              "cleared 25/25 orange pegs in seven shots with seed 50119. "
                              "This is a selected demonstration case, not a randomly generated test game "
                              "or an unbiased evaluation; a newly trained model may perform differently.")
                    else:
                        arrangement_label = "Custom seed arrangement"
                elif seed is not None or winning_arrangement:
                    print("Seed options apply only to Model 1; keeping this all-peg model's original playback.")
                viewer = None
                try:
                    if mode == "1":
                        viewer = LiveGame()
                    if action == "1":
                        model, config = load_game_mode(choice)
                    else:
                        model, config, run_dir = train_fresh(choice, viewer=viewer)
                        if viewer is not None:
                            viewer.stage("Final checkpoint evaluation", "Measuring the selected new policy in headless games.")
                        evaluate_new_model(model, config, run_dir, assisted=choice == "3")
                    play_policy(model, config, MODEL_LABELS[choice], seed=playback_seed,
                                viewer=viewer, arrangement_label=arrangement_label)
                    return
                except Exception as error:
                    print(f"Unable to run DQN Model {choice}: {error}")
                finally:
                    if viewer is not None:
                        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a pretrained DQN or train a new policy.")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, help="Reproduce a Model 1 orange arrangement using this seed.")
    seed_options.add_argument("--winning-arrangement", action="store_true",
                              help="Use Model 1's selected demonstration arrangement (seed 50119).")
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        parser.error("The seed must be nonnegative.")
    main(seed=args.seed, winning_arrangement=args.winning_arrangement)
