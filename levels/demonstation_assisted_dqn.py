import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from levels.config_rl import RLConfig, FINAL_DQN_PATH, DEMONSTRATION_ANGLES
from levels.peggle_env import PeggleEnv

BONUS = 20
DEMO_SEED = 50042


def all_pegs_config():
    return RLConfig(goal="all", max_shots=10, free_balls=False, n_actions=21)



class CompletionReward(gym.Wrapper):
    def __init__(self, bonus=BONUS):
        super().__init__(PeggleEnv(all_pegs_config()))
        self.bonus = bonus

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if info["success"]:
            reward += self.bonus * info["shots_remaining"]
        return obs, reward, done, truncated, info


def collect_demonstration(viewer=None):
    env = CompletionReward()
    obs, _ = env.reset(seed=DEMO_SEED)
    transitions = []
    try:
        for angle in DEMONSTRATION_ANGLES:
            env.unwrapped.record_trajectory = viewer is not None and viewer.active
            board = env.unwrapped.board_records() if env.unwrapped.record_trajectory else None
            action = int((angle + 60) // 6)
            assert env.unwrapped.config.angles_degrees[action] == angle
            before = obs.copy()
            obs, reward, done, truncated, info = env.step(action)
            assert before.shape == obs.shape == (78,) and before.dtype == np.float32
            transitions.append((before, action, reward, obs.copy(), done))
            if viewer is not None and viewer.active:
                viewer.animate(dict(board=board, trajectory=env.unwrapped.trajectory, info=info),
                    env.unwrapped.config, {"Model": "Demonstration-assisted DQN",
                    "Mode": "Demonstration data (not independent RL)",
                    "Stage": "Stage 2: Collecting Demonstration Transitions",
                    "Current shot / angle": f"{info['shots_used']} / {angle:+.0f} degrees",
                    "Pegs cleared": f"{info['pegs_cleared']}/77",
                    "Shots remaining": info["shots_remaining"],
                    "Episode reward": f"{sum(t[2] for t in transitions):.2f}",
                    "Status": "Success" if done else "Collecting demonstration data"})
        assert info["success"] and info["shots_used"] == 8 and info["shots_remaining"] == 2
        assert len(transitions) == 8 and transitions[-1][2] == 148
        assert not truncated
    finally:
        env.close()
    return transitions


def play(model, seed=DEMO_SEED, sampled=False, bonus=BONUS):
    env = CompletionReward(bonus)
    obs, _ = env.reset(seed=seed)
    colors = " ".join(str(i) for i, peg in enumerate(env.unwrapped.pegs) if peg.isOrange)
    rng = np.random.default_rng(seed)
    trace = []
    done = False
    try:
        while not done:
            exploratory = sampled and rng.random() < 0.05
            if exploratory:
                action = int(rng.integers(21))
            else:
                action = int(model.predict(obs, deterministic=True)[0])
            obs, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            trace.append(dict(seed=seed, strategy="Epsilon-greedy" if sampled else "Greedy", action=action,
                              exploratory=exploratory, shot=info["shots_used"], reward=reward,
                              **{k: info[k] for k in ("angle_degrees", "new_pegs_cleared", "pegs_cleared",
                                                     "pegs_remaining", "shots_remaining", "success")}))
    finally:
        env.close()
    game = dict(seed=seed, strategy="Epsilon-greedy" if sampled else "Greedy", orange_peg_ids=colors,
                shots_used=info["shots_used"], pegs_cleared=info["pegs_cleared"], success=info["success"],
                balls_remaining=info["shots_remaining"], reward=sum(s["reward"] for s in trace),
                actions=" ".join(str(s["action"]) for s in trace),
                angle_sequence=" ".join(str(int(s["angle_degrees"])) for s in trace))
    return game, trace


def rank(row):
    winning_shots = row["average_winning_shots"] if row["wins"] else 10
    return row["wins"], -winning_shots, row["greedy_win"], -row["greedy_shots"], row["mean_pegs"]


def load_policy(path=None, train_if_missing=False):
    """Use the historical final policy locally, or build a new source-only policy."""
    chosen = path or FINAL_DQN_PATH
    if not os.path.isfile(chosen):
        if path or not train_if_missing:
            raise FileNotFoundError(chosen)
        from levels.train_dqn import train_model_two
        print("No Model 2 checkpoint found. Starting the initial and demonstration-assisted training stages.")
        chosen = train_model_two()
    model = DQN.load(chosen, device="cpu")
    assert model.observation_space.shape == (78,) and model.action_space.n == 21
    return model


def playback(show=True, checkpoint=None):
    from levels.visualize_data import demonstrate
    model = load_policy(checkpoint)
    game, _ = play(model)
    print(f"DQN Model 2 cleared {game['pegs_cleared']}/77 in {game['shots_used']} greedy shots.")
    print("This policy was trained using simulator demonstrations and DQN replay learning.")
    return demonstrate(model, all_pegs_config(), "DQN Model 2", show=show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["play", "train"])
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--checkpoint", help="optional explicit checkpoint for playback")
    args = parser.parse_args()
    if args.command == "train":
        from levels.train_dqn import train_model_two
        train_model_two()
    else:
        playback(show=not args.no_browser, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
