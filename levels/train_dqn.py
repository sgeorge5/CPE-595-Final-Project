"""Standard DQN training, measured checkpoints, and resumable experiment history."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from levels.config_rl import (CONFIG, DEFAULT_TIMESTEPS, DQN_POLICY, DQN_SETTINGS, SEED,
    IMPROVED_MODELS_DIR, IMPROVED_RESULTS_DIR, CHECKPOINT_INTERVAL,
    VALIDATION_EPISODES, VALIDATION_SEED, ORIGINAL_LEVEL1_MODEL_PATH, SMALL_MODEL_PATH,
    RESULTS_DIR, SMALL_RESULTS_DIR, BEST_MODEL_PATH)
from levels.peggle_env import PeggleEnv
from levels.evaluate import run_policy


def checkpoint_key(row):
    # All checkpoints see the same validation arrangements. Final test games are separate.
    return (float(row['orange_cleared_pct']), float(row['success_rate_pct']))


def save_checkpoint(model, model_path, results_dir, config, rows, eval_episodes, eval_seed):
    step = model.num_timesteps
    if any(int(row['timesteps']) == step for row in rows):
        return
    checkpoint = model_path.parent / f'checkpoint_{step}.zip'
    model.save(checkpoint)
    score, _, _, _ = run_policy(model, episodes=eval_episodes, seed=eval_seed, config=config)
    score.update(timesteps=step, checkpoint=str(checkpoint), validation_seed=eval_seed)
    previous_best = max((checkpoint_key(row) for row in rows), default=(-1,-1))
    if checkpoint_key(score) > previous_best:
        shutil.copyfile(checkpoint, model_path.parent / 'best_model.zip')
        model.save_replay_buffer(model_path.parent / 'best_model.pkl')
    rows.append(score)
    best_index = max(range(len(rows)), key=lambda i: checkpoint_key(rows[i]))
    for i, row in enumerate(rows):
        row['is_best'] = i == best_index
    pd.DataFrame(rows).to_csv(results_dir / 'checkpoint_evaluation.csv', index=False)
    print(f"Checkpoint {step}: clearance {score['orange_cleared_pct']:.1f}%, "
          f"wins {score['success_rate_pct']:.1f}%, angles {score['distinct_angles']}", flush=True)


class EpisodeLogger(BaseCallback):
    """The SB3 callback collects completed episodes and periodically checks the policy."""
    columns = ['episode','timesteps','episode_reward','shots_used','orange_initial',
        'orange_cleared','orange_cleared_pct','success','simulation_timeouts','stuck_removals']

    def __init__(self, previous_rows=None, checkpoint_rows=None, model_path=None,
                 results_dir=None, config=CONFIG, interval=CHECKPOINT_INTERVAL,
                 eval_episodes=VALIDATION_EPISODES, eval_seed=VALIDATION_SEED):
        super().__init__()
        self.rows = [] if previous_rows is None else previous_rows
        self.checkpoints = [] if checkpoint_rows is None else checkpoint_rows
        self.model_path, self.results_dir, self.config = model_path, results_dir, config
        self.interval, self.eval_episodes, self.eval_seed = interval, eval_episodes, eval_seed

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.rows.append({'episode':len(self.rows)+1, 'timesteps':self.num_timesteps,
                    'episode_reward':info['episode']['r'], 'shots_used':info['shots_used'],
                    'orange_initial':info['orange_initial'], 'orange_cleared':info['orange_cleared'],
                    'orange_cleared_pct':100*info['orange_cleared']/info['orange_initial'],
                    'success':info['success'], 'simulation_timeouts':info['simulation_timeouts'],
                    'stuck_removals':info['stuck_removals']})
        if self.model_path is not None and self.num_timesteps % self.interval == 0:
            save_checkpoint(self.model, self.model_path, self.results_dir, self.config,
                            self.checkpoints, self.eval_episodes, self.eval_seed)
        return True


def plot_training(log_path=None, output_path=None):
    log_path = IMPROVED_RESULTS_DIR/'training_episodes.csv' if log_path is None else log_path
    data = pd.read_csv(log_path)
    fig, axes = plt.subplots(3,1,figsize=(9,8),sharex=True)
    for ax, column, label in zip(axes, ['episode_reward','success','orange_cleared_pct'],
                                 ['Episode return','Win rate (%)','Orange cleared (%)']):
        if len(data):
            values = data[column].astype(float) * (100 if column == 'success' else 1)
            ax.plot(data.timesteps, values, alpha=0.2, color='#2786c4')
            ax.plot(data.timesteps, values.rolling(min(50,len(data)),min_periods=1).mean(),
                    color='#124c79', label='Rolling mean (up to 50 episodes)')
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        if column != 'episode_reward':
            ax.set_ylim(-2,102)
    axes[0].set_title('Training episodes, including exploratory actions')
    axes[-1].set_xlabel('Cumulative environment timesteps (complete shots)')
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path,dpi=180)
    return fig


def plot_checkpoints(csv_path, output_path=None):
    data = pd.read_csv(csv_path)
    fig, axes = plt.subplots(3,1,figsize=(8,7),sharex=True)
    for ax, column, label in zip(axes,['orange_cleared_pct','success_rate_pct','distinct_angles'],
                                 ['Orange cleared (%)','Win rate (%)','Distinct angles']):
        ax.plot(data.timesteps,data[column],marker='o')
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].set_title('Deterministic checkpoints on fixed validation arrangements')
    axes[-1].set_xlabel('Cumulative training shots')
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path,dpi=180)
    return fig


def load_training_model(path, env, warmup=200):
    """SB3 restores weights and optimizer; replay is a separate supported save file."""
    path = Path(path)
    model = DQN.load(path, env=env, device='cpu')
    replay = path.with_suffix('.pkl')
    if replay.exists():
        model.load_replay_buffer(replay)
    else:
        print('No saved replay buffer: retaining weights/optimizer and collecting fresh experience.')
        model.learning_starts = model.num_timesteps + warmup
    return model


def record_experiment(name, config, model, episodes):
    """One measured summary per experiment; detailed checkpoints retain its history."""
    scores = pd.read_csv(IMPROVED_RESULTS_DIR/name/'checkpoint_evaluation.csv')
    best = scores[scores.is_best].iloc[0]
    row = {'experiment':name,'observation':config.observation,
        'input_size':model.observation_space.shape[0],'learning_rate':model.learning_rate,
        'empty_shot_penalty':config.empty_shot_penalty,'total_training_shots':model.num_timesteps,
        'completed_episodes':len(episodes),'best_step':int(best.timesteps),
        'validation_clearance_pct':best.orange_cleared_pct,
        'validation_win_rate_pct':best.success_rate_pct,
        'validation_distinct_angles':int(best.distinct_angles)}
    path = IMPROVED_RESULTS_DIR/'experiments.csv'
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(previous):
        previous = previous[previous.experiment != name]
    data = pd.concat([previous,pd.DataFrame([row])],ignore_index=True)
    data.to_csv(path,index=False)
    return data


def select_best_experiment():
    """Publish the validation winner; the separate final evaluation is never consulted."""
    experiments = pd.read_csv(IMPROVED_RESULTS_DIR/'experiments.csv')
    selected = experiments.iloc[0]
    for _, row in experiments.iterrows():
        if (row.validation_clearance_pct,row.validation_win_rate_pct) > (
                selected.validation_clearance_pct,selected.validation_win_rate_pct):
            selected = row
    source = IMPROVED_MODELS_DIR/selected.experiment/'best_model.zip'
    shutil.copyfile(source,BEST_MODEL_PATH)
    shutil.copyfile(source.with_suffix('.pkl'),BEST_MODEL_PATH.with_suffix('.pkl'))
    selection = selected.to_dict()
    selection['source_model'] = str(source)
    selection['published_model'] = str(BEST_MODEL_PATH)
    pd.DataFrame([selection]).to_csv(IMPROVED_RESULTS_DIR/'selected_model.csv',index=False)
    return selection


def train_agent(total_timesteps=DEFAULT_TIMESTEPS, seed=SEED, model_path=None,
                results_dir=None, config=CONFIG, verbose=1, dqn_settings=None,
                resume_from=None, checkpoint_interval=CHECKPOINT_INTERVAL,
                validation_episodes=VALIDATION_EPISODES, validation_seed=VALIDATION_SEED):
    """Train this many ADDITIONAL shots. Resume never resets the learned weights.

    Use a new output folder for a new experiment. Resume last_model.zip to retain
    replay, optimizer and cumulative steps. SB3 extends its exploration schedule
    to the new total budget; environment resets are not bit-exact interrupted play.
    """
    if total_timesteps < 1 or checkpoint_interval < 1 or validation_episodes < 1:
        raise ValueError('Training and evaluation budgets must be positive.')
    model_path = Path(IMPROVED_MODELS_DIR/'last_model.zip' if model_path is None else model_path)
    folder = Path(IMPROVED_RESULTS_DIR if results_dir is None else results_dir)
    if model_path.suffix != '.zip':
        raise ValueError('Use a .zip model filename.')
    if model_path.resolve() in (ORIGINAL_LEVEL1_MODEL_PATH.resolve(), SMALL_MODEL_PATH.resolve()):
        raise ValueError('Preserve the original model: choose an improved experiment path.')
    if folder.resolve() in (RESULTS_DIR.resolve(), SMALL_RESULTS_DIR.resolve()):
        raise ValueError('Preserve the original results: choose an improved experiment folder.')
    if resume_from is None and (model_path.exists() or (folder/'training_episodes.csv').exists()):
        raise FileExistsError('Existing experiment: resume it or choose new model/results paths.')
    model_path.parent.mkdir(parents=True,exist_ok=True)
    folder.mkdir(parents=True,exist_ok=True)
    settings = dict(DQN_SETTINGS if dqn_settings is None else dqn_settings)
    torch.set_num_threads(1)
    env = Monitor(PeggleEnv(config))
    if resume_from is None:
        model = DQN(DQN_POLICY,env,seed=seed,device='cpu',verbose=verbose,**settings)
    else:
        model = load_training_model(resume_from,env,settings['learning_starts'])
        model.verbose = verbose
    start_step = model.num_timesteps
    if resume_from is not None:
        # A resumed segment starts a fresh, reproducible sequence of games.
        model.get_env().seed(seed + 1_000_000 + start_step)
    previous_rows, checkpoint_rows, settings_rows = [], [], []
    if resume_from is not None:
        if (folder/'training_episodes.csv').exists():
            previous_rows = pd.read_csv(folder/'training_episodes.csv').to_dict('records')
        if (folder/'checkpoint_evaluation.csv').exists():
            checkpoint_rows = pd.read_csv(folder/'checkpoint_evaluation.csv').to_dict('records')
        if any(row['timesteps'] > start_step for row in previous_rows + checkpoint_rows):
            env.close()
            raise ValueError('History extends beyond this checkpoint. Use a new output folder to branch.')
        if (folder/'training_settings.csv').exists():
            settings_rows = pd.read_csv(folder/'training_settings.csv').to_dict('records')
            last = settings_rows[-1]
            if (last['observation'] != config.observation or
                    float(last['empty_shot_penalty']) != config.empty_shot_penalty):
                env.close()
                raise ValueError('Use a new experiment folder when changing observations or rewards.')
    settings_rows.append({'starting_step':start_step,'additional_shots':total_timesteps,
        'seed':seed,'observation':config.observation,'input_size':env.observation_space.shape[0],
        'orange_targets':config.n_orange,'empty_shot_penalty':config.empty_shot_penalty,
        'learning_rate':model.learning_rate,'gamma':model.gamma,'buffer_size':model.buffer_size,
        'batch_size':model.batch_size,'learning_starts':model.learning_starts,
        'exploration_fraction':model.exploration_fraction,'final_epsilon':model.exploration_final_eps,
        'network':'64 64','validation_seed':validation_seed,'validation_episodes':validation_episodes})
    pd.DataFrame(settings_rows).to_csv(folder/'training_settings.csv',index=False)
    callback = EpisodeLogger(previous_rows,checkpoint_rows,model_path,folder,config,
                             checkpoint_interval,validation_episodes,validation_seed)
    try:
        model.learn(total_timesteps=total_timesteps,callback=callback,
                    reset_num_timesteps=resume_from is None,log_interval=100)
        model.save(model_path)
        model.save_replay_buffer(model_path.with_suffix('.pkl'))
        save_checkpoint(model,model_path,folder,config,callback.checkpoints,
                        validation_episodes,validation_seed)
    finally:
        env.close()
    data = pd.DataFrame(callback.rows,columns=EpisodeLogger.columns)
    data.to_csv(folder/'training_episodes.csv',index=False)
    plt.close(plot_training(folder/'training_episodes.csv',folder/'training_rewards.png'))
    plt.close(plot_checkpoints(folder/'checkpoint_evaluation.csv',folder/'checkpoint_evaluation.png'))
    return model,data


def main():
    # Edit these simple settings for a new run. Existing results are protected.
    from levels.config_rl import RLConfig
    run_name = 'new_run'
    resume = False
    config = RLConfig(observation='compact')
    settings = dict(DQN_SETTINGS)
    settings['learning_rate'] = 0.0003
    path = IMPROVED_MODELS_DIR/run_name/'last_model.zip'
    model, episodes = train_agent(config=config,dqn_settings=settings,model_path=path,
        results_dir=IMPROVED_RESULTS_DIR/run_name,resume_from=path if resume else None)
    record_experiment(run_name,config,model,episodes)
    selection = select_best_experiment()
    print('Total training shots:',model.num_timesteps,'Completed episodes:',len(episodes))
    print('Best validation model:',selection['source_model'])

if __name__ == '__main__':
    main()
