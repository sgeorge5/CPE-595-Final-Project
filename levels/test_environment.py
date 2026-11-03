"""Focused behavior/integration checks. Run: python -m pytest levels/test_environment.py -q"""

from dataclasses import replace
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env as sb3_check_env

if __package__:
    from .config_rl import CONFIG, FIXED_BOARD, N_ORANGE, ROOT
    from .peggle_env import PeggleEnv
    from .visualize_data import original_level_data, level_inventory
    from .train_dqn import train_agent
    from .evaluate import evaluate_agents
else:
    from config_rl import CONFIG, FIXED_BOARD, N_ORANGE, ROOT
    from peggle_env import PeggleEnv
    from visualize_data import original_level_data, level_inventory
    from train_dqn import train_agent
    from evaluate import evaluate_agents


def test_reset_and_preprocessing():
    env = PeggleEnv()
    initial, info = env.reset(seed=42)
    assert initial.shape == (41,) and initial.dtype == np.float32
    assert env.observation_space.contains(initial)
    assert info["orange_remaining"] == 3 and info["shots_remaining"] == 10
    features = initial[:-1].reshape(10, 4)
    np.testing.assert_allclose(features[:, :2], np.array([p[:2] for p in FIXED_BOARD]) / [1200, 900])
    assert features[:, 2].sum() == 3 and features[:, 3].sum() == 0
    assert initial[-1] == 1.0
    env.step(10)
    reset, reset_info = env.reset(seed=42)
    np.testing.assert_array_equal(initial, reset)
    assert reset_info == info


@pytest.mark.parametrize("action", range(21))
def test_full_shot_and_reward_accounting(action):
    env = PeggleEnv()
    initial, _ = env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert obs[-1] == np.float32(0.9)
    assert info["shots_used"] == 1 and info["simulation_frames"] > 1
    assert not truncated
    assert not info["shot_timed_out"]
    expected = 10 * info["new_orange_cleared"] - 1 + (25 if info["success"] else 0)
    assert reward == expected
    assert terminated == info["success"]
    # Coordinates/colors keep the same slots even when pegs have been removed.
    np.testing.assert_array_equal(obs[:-1].reshape(10, 4)[:, :3], initial[:-1].reshape(10, 4)[:, :3])


def test_physical_hit_not_rewarded_twice():
    env = PeggleEnv()
    env.reset(seed=42)
    _, first_reward, _, _, first = env.step(10)
    assert first["new_orange_cleared"] >= 1 and first_reward >= 9
    _, second_reward, _, _, second = env.step(10)
    assert second["new_orange_cleared"] == 0 and second_reward == -1
    assert second["orange_cleared"] == first["orange_cleared"]


def test_success_bonus_and_terminal_guard():
    env = PeggleEnv()
    env.reset(seed=42)
    # Construct a reachable final-orange state, then physically hit that orange.
    env.pegs[0].isHit = env.pegs[2].isHit = True
    _, reward, terminated, truncated, info = env.step(10)
    assert terminated and not truncated and info["success"]
    assert info["new_orange_cleared"] == 1 and reward == 34
    assert any(not p.isHit and p.color == "blue" for p in env.pegs)
    with pytest.raises(RuntimeError):
        env.step(10)


def test_budget_exhaustion():
    env = PeggleEnv(replace(CONFIG, max_shots=1))
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and not info["success"]
    assert obs[-1] == 0 and info["terminal_reason"] == "shot_budget"
    assert reward == -1


def test_invalid_actions_and_step_before_reset():
    env = PeggleEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    for action in (-1, 21, 0.5, [10]):
        with pytest.raises(ValueError):
            env.step(action)


def test_deterministic_sequences_and_isolated_instances():
    left, right = PeggleEnv(), PeggleEnv()
    left.reset(seed=1)
    right.reset(seed=999)
    for action in (3, 10, 17, 0, 7):
        a, b = left.step(action), right.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        if a[2]:
            break
    before = right._observation().copy()
    left.reset()
    np.testing.assert_array_equal(before, right._observation())


def test_gymnasium_and_sb3_contract():
    gym_check_env(PeggleEnv(), skip_render_check=True)
    sb3_check_env(PeggleEnv(), warn=True)


def test_upstream_data_and_no_graphical_side_effects():
    data = original_level_data()
    assert len(data) == 77 and len(level_inventory()) == 10
    assert not data.color_stored_in_file.any()
    assert not data.initial_is_hit.any()
    assert "local.config" not in sys.modules
    assert "pygame" not in sys.modules
    assert not (ROOT / "user.cfg").exists()


def test_dqn_updates_weights_and_save_load(tmp_path):
    torch.set_num_threads(1)
    env = PeggleEnv()
    model = DQN("MlpPolicy", env, seed=42, device="cpu", learning_starts=16,
                buffer_size=128, batch_size=16, train_freq=4,
                policy_kwargs={"net_arch": [32, 32]})
    before = {k: v.clone() for k, v in model.q_net.state_dict().items()}
    model.learn(total_timesteps=64)
    assert model._n_updates > 0
    assert any(not torch.equal(before[k], v) for k, v in model.q_net.state_dict().items())
    model.save(tmp_path / "compatibility.zip")
    restored = DQN.load(tmp_path / "compatibility.zip", device="cpu")
    obs, _ = env.reset(seed=42)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
    env.close()


def test_training_and_evaluation_artifacts(tmp_path):
    path = tmp_path / "dqn_orange.zip"
    _, metadata = train_agent(32, model_path=path, results_dir=tmp_path, verbose=0)
    assert path.exists() and metadata["completed_episodes"] > 0
    # 32 steps deliberately checks the pipeline, not a claim of useful learning.
    summary, episodes = evaluate_agents(path, episodes=3, results_dir=tmp_path)
    assert len(episodes) == 6 and list(summary.agent) == ["DQN", "Random"]
    saved = pd.read_csv(tmp_path / "evaluation_episodes.csv")
    assert saved.orange_cleared.between(0, N_ORANGE).all()
    assert saved.shots_used.between(1, 10).all()
    for row in summary.itertuples():
        actual = saved[saved.agent == row.agent]
        assert row.success_rate_pct == pytest.approx(actual.success.mean() * 100)
        assert row.orange_cleared_pct == pytest.approx(actual.orange_cleared.mean() / N_ORANGE * 100)
        successful = actual[actual.success]
        assert (np.isnan(row.avg_shots_success) if successful.empty else
                row.avg_shots_success == pytest.approx(successful.shots_used.mean()))
    assert json.loads((tmp_path / "evaluation_metadata.json").read_text())["training_timesteps"] == 32
    for name in ("training_rewards.png", "dqn_vs_random.png"):
        assert (tmp_path / name).stat().st_size > 1000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
