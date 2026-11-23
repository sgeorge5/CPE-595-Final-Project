"""Play either final DQN objective, training from source if its model is absent."""
import os
from stable_baselines3 import DQN
from levels.config_rl import BEST_MODEL_PATH, RLConfig
from levels.visualize_data import demonstrate


def load_game_mode(choice):
    if choice == "1":
        path = BEST_MODEL_PATH if os.path.isfile(BEST_MODEL_PATH) else "models/source_only/model1/best_model.zip"
        if not os.path.isfile(path):
            from levels.train_dqn import train_model_one
            print("No Model 1 checkpoint found. Starting fresh orange-DQN training.")
            path = train_model_one()
        model = DQN.load(path, device="cpu")
        config = RLConfig(observation="compact")
        assert model.observation_space.shape == (155,) and model.action_space.n == 21
        return model, config
    elif choice == "2":
        from levels.dqn_eight_shot_experiment import load_policy, all_pegs_config
        return load_policy(), all_pegs_config()
    raise ValueError("Please enter 1 or 2.")


def main():
    print("PEGGLE REINFORCEMENT LEARNING\n")
    print("1. DQN Model 1 - Clear all orange pegs")
    print("2. DQN Model 2 - Clear all 77 pegs")
    choice = input("Enter your choice: ").strip()
    while choice not in ("1", "2"):
        choice = input("Please enter 1 or 2: ").strip()
    model, config = load_game_mode(choice)
    demonstrate(model, config, "DQN Model " + choice)


if __name__ == "__main__":
    main()
