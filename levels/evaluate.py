"""Measured policy comparisons and separate environment diagnostics."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from levels.config_rl import (CONFIG, RLConfig, DEFAULT_EVAL_EPISODES, BEST_MODEL_PATH,
    IMPROVED_RESULTS_DIR, ORIGINAL_LEVEL1_MODEL_PATH, TEST_SEED)
from levels.peggle_env import PeggleEnv


def model_observation(env, model):
    """Use the same numerical representation this model was trained with."""
    size = model.observation_space.shape[0]
    if size == env.config.n_pegs * 4 + 1:
        return env._observation('full')
    if size == env.config.n_pegs * 2 + 1:
        return env._observation('compact')
    raise ValueError('Model input size does not match this board.')


def run_policy(model=None, policy='dqn', episodes=100, seed=TEST_SEED,
               config=CONFIG, name='DQN'):
    """Play without saving files: used for selection and separate final tests."""
    if episodes < 1 or policy not in ('dqn', 'random', 'fixed'):
        raise ValueError('Choose a supported policy and a positive episode count.')
    rows, shots, arrangements = [], [], []
    env = PeggleEnv(config)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            env.reset(seed=episode_seed)
            if policy == 'dqn':
                model_observation(env, model)
            orange_ids = [i for i, peg in enumerate(env.pegs) if peg.isOrange]
            arrangements.append({'agent': name, 'episode': episode + 1, 'seed': episode_seed,
                                 'orange_peg_ids': ' '.join(str(i) for i in orange_ids)})
            rng = np.random.default_rng(episode_seed + 1_000_000)
            total_reward, done = 0.0, False
            while not done:
                if policy == 'dqn':
                    action, _ = model.predict(model_observation(env, model), deterministic=True)
                    action = int(action)
                elif policy == 'random':
                    action = int(rng.integers(config.n_actions))
                else:
                    action = int(np.argmin(abs(config.angles_degrees)))
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                shots.append({'agent': name, 'episode': episode + 1, 'seed': episode_seed,
                    'shot': info['shots_used'], 'action': action,
                    'angle_degrees': info['angle_degrees'], 'reward': reward,
                    'new_orange_cleared': info['new_orange_cleared'],
                    'orange_remaining': info['orange_remaining'],
                    'shots_remaining': info['shots_remaining'],
                    'simulation_frames': info['simulation_frames'],
                    'shot_timed_out': info['shot_timed_out']})
                done = terminated or truncated
            rows.append({'agent': name, 'episode': episode + 1, 'seed': episode_seed,
                'episode_reward': total_reward, 'success': info['success'],
                'orange_initial': info['orange_initial'], 'orange_cleared': info['orange_cleared'],
                'orange_remaining': info['orange_remaining'],
                'orange_cleared_pct': 100 * info['orange_cleared'] / info['orange_initial'],
                'shots_used': info['shots_used'], 'simulation_timeouts': info['simulation_timeouts'],
                'stuck_removals': info['stuck_removals']})
    finally:
        env.close()
    data, shot_data = pd.DataFrame(rows), pd.DataFrame(shots)
    successful = data[data.success]
    angles = shot_data.angle_degrees.value_counts()
    summary = {'agent': name, 'episodes': len(data), 'successes': len(successful),
        'success_rate_pct': 100 * data.success.mean(), 'avg_orange_cleared': data.orange_cleared.mean(),
        'orange_cleared_pct': data.orange_cleared_pct.mean(),
        'avg_orange_remaining': data.orange_remaining.mean(),
        'avg_shots_success': successful.shots_used.mean() if len(successful) else np.nan,
        'avg_shots_all': data.shots_used.mean(), 'mean_episode_reward': data.episode_reward.mean(),
        'distinct_angles': len(angles), 'most_common_angle': float(angles.index[0]),
        'most_common_angle_pct': 100 * angles.iloc[0] / len(shot_data),
        'simulation_timeouts': int(data.simulation_timeouts.sum())}
    return summary, data, shot_data, pd.DataFrame(arrangements)


def plot_comparison(summary, path=None):
    metrics = [('success_rate_pct', 'Win rate (%)'), ('orange_cleared_pct', 'Orange cleared (%)'),
        ('avg_orange_remaining', 'Mean oranges remaining'), ('avg_shots_success', 'Shots in wins'),
        ('mean_episode_reward', 'Mean episode reward'), ('distinct_angles', 'Distinct angles used')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (column, title) in zip(axes.flat, metrics):
        values = summary[column].to_numpy(dtype=float)
        bars = ax.bar(summary.agent, np.nan_to_num(values, nan=0),
                      color=['#2786c4', '#ed8b22', '#888888', '#308b55'][:len(summary)])
        for bar, value in zip(bars, values):
            label = 'N/A' if np.isnan(value) else f'{value:.1f}'
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8, xytext=(0,3), textcoords='offset points')
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelrotation=20, labelsize=8)
        ax.margins(y=0.25)
        if 'pct' in column:
            ax.set_ylim(0, 110)
        if np.isnan(values).all():
            ax.set_ylim(0, 2)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Matched starting arrangements, physics, rewards and ten-shot budget')
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180)
    return fig


def evaluate_agents(model_path=None, episodes=DEFAULT_EVAL_EPISODES, seed=TEST_SEED,
                    results_dir=None, config=CONFIG, model=None, include_baselines=False,
                    original_model_path=ORIGINAL_LEVEL1_MODEL_PATH):
    """Rerun all policies under current rules, including the same empty-shot penalty."""
    model_path = Path(BEST_MODEL_PATH if model_path is None else model_path)
    folder = Path(IMPROVED_RESULTS_DIR if results_dir is None else results_dir)
    if model is None:
        model = DQN.load(model_path, device='cpu')
    agents = [('DQN', 'dqn', model), ('Random', 'random', None)]
    if include_baselines:
        agents.append(('Straight down', 'fixed', None))
        if original_model_path is not None and Path(original_model_path).exists():
            agents.append(('Original DQN', 'dqn', DQN.load(original_model_path, device='cpu')))
    summaries, episode_tables, shot_tables, board_tables = [], [], [], []
    for name, policy, agent in agents:
        summary, data, shots, boards = run_policy(agent, policy, episodes, seed, config, name)
        summaries.append(summary)
        episode_tables.append(data)
        shot_tables.append(shots)
        board_tables.append(boards)
    summary = pd.DataFrame(summaries)
    data = pd.concat(episode_tables, ignore_index=True)
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder/'evaluation_episodes.csv', index=False)
    pd.concat(shot_tables, ignore_index=True).to_csv(folder/'evaluation_shots.csv', index=False)
    pd.concat(board_tables, ignore_index=True).to_csv(folder/'evaluation_arrangements.csv', index=False)
    summary.to_csv(folder/'evaluation_summary.csv', index=False)
    plt.close(plot_comparison(summary, folder/'dqn_vs_random.png'))
    return summary, data


def angle_diagnostic(seeds=(42,43,44), config=CONFIG, results_dir=IMPROVED_RESULTS_DIR):
    """One fresh copy of the same seeded arrangement per angle; never trains DQN."""
    rows = []
    env = PeggleEnv(config, record_trajectory=True)
    for seed in seeds:
        for action in range(config.n_actions):
            env.reset(seed=seed)
            _, reward, _, _, info = env.step(action)
            point = env.trajectory[min(10, len(env.trajectory)-1)]
            rows.append({'seed': seed, 'action': action, 'angle_degrees': info['angle_degrees'],
                'orange_cleared': info['new_orange_cleared'], 'reward': reward,
                'simulation_frames': info['simulation_frames'],
                'x_at_frame_10': point['x'], 'y_at_frame_10': point['y']})
    env.close()
    data = pd.DataFrame(rows)
    folder = Path(results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder/'angle_diagnostic.csv', index=False)
    fig, ax = plt.subplots(figsize=(9,4))
    for seed, group in data.groupby('seed'):
        ax.plot(group.angle_degrees, group.orange_cleared, marker='o', label=f'Seed {seed}')
    ax.set(xlabel='Launch angle (degrees)', ylabel='New orange pegs cleared',
           title='Fresh starting board for each angle; separate from learning')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder/'angle_diagnostic.png', dpi=180)
    plt.close(fig)
    return data


def greedy_diagnostic(seeds=(30042,30043,30044), config=CONFIG, results_dir=IMPROVED_RESULTS_DIR):
    """Small simulator-lookahead feasibility probe. NEVER supplies DQN actions."""
    episodes, shots = [], []
    for seed in seeds:
        env = PeggleEnv(config)
        env.reset(seed=seed)
        done = False
        while not done:
            best_action, best_clearance = 0, -1
            for action in range(config.n_actions):
                trial = PeggleEnv(config)
                trial.reset(seed=seed)
                trial.shots_remaining = env.shots_remaining
                for peg, source in zip(trial.pegs, env.pegs):
                    peg.isHit, peg.isVisible = source.isHit, source.isVisible
                _, _, _, _, info = trial.step(action)
                if info['new_orange_cleared'] > best_clearance:
                    best_action, best_clearance = action, info['new_orange_cleared']
                trial.close()
            _, reward, terminated, truncated, info = env.step(best_action)
            shots.append({'seed': seed, 'shot': info['shots_used'], 'action': best_action,
                'angle_degrees': info['angle_degrees'], 'reward': reward,
                'new_orange_cleared': info['new_orange_cleared'], 'orange_remaining': info['orange_remaining']})
            done = terminated or truncated
        episodes.append({'seed': seed, 'success': info['success'], 'shots_used': info['shots_used'],
            'orange_cleared': info['orange_cleared'], 'orange_remaining': info['orange_remaining'],
            'orange_cleared_pct': 100 * info['orange_cleared'] / info['orange_initial']})
        env.close()
    folder = Path(results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episodes).to_csv(folder/'greedy_diagnostic_episodes.csv', index=False)
    pd.DataFrame(shots).to_csv(folder/'greedy_diagnostic_shots.csv', index=False)
    return pd.DataFrame(episodes)


def main():
    selected = pd.read_csv(IMPROVED_RESULTS_DIR/'selected_model.csv').iloc[0]
    config = RLConfig(observation=selected.observation,
                      empty_shot_penalty=float(selected.empty_shot_penalty))
    summary, _ = evaluate_agents(config=config,include_baselines=True)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
