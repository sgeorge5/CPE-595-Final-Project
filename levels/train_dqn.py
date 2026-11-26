"""Standard DQN training, measured checkpoints, and resumable experiment history."""
import os
import shutil
import argparse
import glob
import numpy as np
from stable_baselines3.common.utils import LinearSchedule
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
    VALIDATION_EPISODES, VALIDATION_SEED, ORIGINAL_LEVEL1_MODEL_PATH, RESULTS_DIR, BEST_MODEL_PATH)
from levels.peggle_env import PeggleEnv
from levels.evaluate import run_policy


def checkpoint_key(row):
    # All checkpoints see the same validation arrangements. Final test games are separate.
    return (float(row['orange_cleared_pct']), float(row['success_rate_pct']))


def save_checkpoint(model, model_path, results_dir, config, rows, eval_episodes, eval_seed):
    step = model.num_timesteps
    if any(int(row['timesteps']) == step for row in rows):
        return
    model_folder = os.path.dirname(model_path)
    checkpoint = os.path.join(model_folder, f'checkpoint_{step}.zip')
    model.save(checkpoint)
    score, _, _, _ = run_policy(model, episodes=eval_episodes, seed=eval_seed, config=config)
    score.update(timesteps=step, checkpoint=str(checkpoint), validation_seed=eval_seed)
    previous_best = max((checkpoint_key(row) for row in rows), default=(-1,-1))
    if checkpoint_key(score) > previous_best:
        shutil.copyfile(checkpoint, os.path.join(model_folder, 'best_model.zip'))
        model.save_replay_buffer(os.path.join(model_folder, 'best_model.pkl'))
    rows.append(score)
    best_index = max(range(len(rows)), key=lambda i: checkpoint_key(rows[i]))
    for i, row in enumerate(rows):
        row['is_best'] = i == best_index
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'checkpoint_evaluation.csv'), index=False)
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
    log_path = os.path.join(IMPROVED_RESULTS_DIR, 'training_episodes.csv') if log_path is None else log_path
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
    model = DQN.load(path, env=env, device='cpu')
    replay = os.path.splitext(path)[0] + '.pkl'
    if os.path.exists(replay):
        model.load_replay_buffer(replay)
    else:
        print('No saved replay buffer: retaining weights/optimizer and collecting fresh experience.')
        model.learning_starts = model.num_timesteps + warmup
    return model


def record_experiment(name, config, model, episodes):
    """One measured summary per experiment; detailed checkpoints retain its history."""
    scores = pd.read_csv(os.path.join(IMPROVED_RESULTS_DIR, name, 'checkpoint_evaluation.csv'))
    best = scores[scores.is_best].iloc[0]
    row = {'experiment':name,'observation':config.observation,
        'input_size':model.observation_space.shape[0],'learning_rate':model.learning_rate,
        'empty_shot_penalty':config.empty_shot_penalty,'total_training_shots':model.num_timesteps,
        'completed_episodes':len(episodes),'best_step':int(best.timesteps),
        'validation_clearance_pct':best.orange_cleared_pct,
        'validation_win_rate_pct':best.success_rate_pct,
        'validation_distinct_angles':int(best.distinct_angles)}
    path = os.path.join(IMPROVED_RESULTS_DIR, 'experiments.csv')
    previous = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(previous):
        previous = previous[previous.experiment != name]
    data = pd.concat([previous,pd.DataFrame([row])],ignore_index=True)
    data.to_csv(path,index=False)
    return data


def select_best_experiment():
    """Publish the validation winner; the separate final evaluation is never consulted."""
    experiments = pd.read_csv(os.path.join(IMPROVED_RESULTS_DIR, 'experiments.csv'))
    selected = experiments.iloc[0]
    for _, row in experiments.iterrows():
        if (row.validation_clearance_pct,row.validation_win_rate_pct) > (
                selected.validation_clearance_pct,selected.validation_win_rate_pct):
            selected = row
    source = os.path.join(IMPROVED_MODELS_DIR, selected.experiment, 'best_model.zip')
    shutil.copyfile(source,BEST_MODEL_PATH)
    shutil.copyfile(os.path.splitext(source)[0] + '.pkl', os.path.splitext(BEST_MODEL_PATH)[0] + '.pkl')
    selection = selected.to_dict()
    selection['source_model'] = str(source)
    selection['published_model'] = str(BEST_MODEL_PATH)
    pd.DataFrame([selection]).to_csv(os.path.join(IMPROVED_RESULTS_DIR, 'selected_model.csv'),index=False)
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
    model_path = os.path.join(IMPROVED_MODELS_DIR, 'last_model.zip') if model_path is None else str(model_path)
    folder = IMPROVED_RESULTS_DIR if results_dir is None else str(results_dir)
    if os.path.splitext(model_path)[1] != '.zip':
        raise ValueError('Use a .zip model filename.')
    if os.path.normcase(os.path.realpath(model_path)) == os.path.normcase(os.path.realpath(ORIGINAL_LEVEL1_MODEL_PATH)):
        raise ValueError('Preserve the original model: choose an improved experiment path.')
    if os.path.normcase(os.path.realpath(folder)) in (os.path.normcase(os.path.realpath(RESULTS_DIR)),
                                                         os.path.normcase(os.path.realpath('results'))):
        raise ValueError('Preserve the original results: choose an improved experiment folder.')
    if resume_from is None and (os.path.exists(model_path) or os.path.exists(os.path.join(folder, 'training_episodes.csv'))):
        raise FileExistsError('Existing experiment: resume it or choose new model/results paths.')
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs(folder, exist_ok=True)
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
        if os.path.exists(os.path.join(folder, 'training_episodes.csv')):
            previous_rows = pd.read_csv(os.path.join(folder, 'training_episodes.csv')).to_dict('records')
        if os.path.exists(os.path.join(folder, 'checkpoint_evaluation.csv')):
            checkpoint_rows = pd.read_csv(os.path.join(folder, 'checkpoint_evaluation.csv')).to_dict('records')
        if any(row['timesteps'] > start_step for row in previous_rows + checkpoint_rows):
            env.close()
            raise ValueError('History extends beyond this checkpoint. Use a new output folder to branch.')
        if os.path.exists(os.path.join(folder, 'training_settings.csv')):
            settings_rows = pd.read_csv(os.path.join(folder, 'training_settings.csv')).to_dict('records')
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
    pd.DataFrame(settings_rows).to_csv(os.path.join(folder, 'training_settings.csv'),index=False)
    callback = EpisodeLogger(previous_rows,checkpoint_rows,model_path,folder,config,
                             checkpoint_interval,validation_episodes,validation_seed)
    try:
        model.learn(total_timesteps=total_timesteps,callback=callback,
                    reset_num_timesteps=resume_from is None,log_interval=100)
        model.save(model_path)
        model.save_replay_buffer(os.path.splitext(model_path)[0] + '.pkl')
        save_checkpoint(model,model_path,folder,config,callback.checkpoints,
                        validation_episodes,validation_seed)
    finally:
        env.close()
    data = pd.DataFrame(callback.rows,columns=EpisodeLogger.columns)
    data.to_csv(os.path.join(folder, 'training_episodes.csv'),index=False)
    plt.close(plot_training(os.path.join(folder, 'training_episodes.csv'),os.path.join(folder, 'training_rewards.png')))
    plt.close(plot_checkpoints(os.path.join(folder, 'checkpoint_evaluation.csv'),os.path.join(folder, 'checkpoint_evaluation.png')))
    return model,data



ALL_PEG_SETTINGS = {'learning_rate': 0.0003, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 64, 'learning_starts': 1000, 'policy_kwargs': {'net_arch': [64, 64]}, 'train_freq': 4, 'gradient_steps': 1, 'target_update_interval': 500, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'exploration_fraction': 0.6}

class TrainingMonitor(Monitor):
    """Keep the random board seed with its terminal transition for exact replay."""

    def __init__(self, env, seed):
        super().__init__(env)
        self.boards = np.random.default_rng(seed)

    def reset(self, **kwargs):
        if kwargs.get("seed") is None:
            kwargs["seed"] = int(self.boards.integers(1_000_000, 2**31))
        self.board_seed = kwargs["seed"]
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        if info["success"]:
            info["board_seed"] = self.board_seed
            info["orange_peg_ids"] = " ".join(str(i) for i, peg in enumerate(self.env.pegs) if peg.isOrange)
        return observation, reward, done, truncated, info


class TrainingLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows, self.shots = [], []

    def _on_step(self):
        info = self.locals["infos"][0]
        self.shots.append(dict(action=int(self.locals["actions"][0]),
                               shot=info["shots_used"], reward=float(self.locals["rewards"][0]),
                               **{key: info[key] for key in ("angle_degrees", "new_pegs_cleared", "pegs_cleared",
                                  "pegs_remaining", "shots_remaining", "success")}))
        if "episode" in info:
            row = dict(timesteps=self.num_timesteps, reward=info["episode"]["r"],
                       **{key: info[key] for key in ("pegs_cleared", "shots_used", "success",
                                                   "simulation_timeouts")})
            self.rows.append(row)
            self.shots = []
        return True


def write_csv(data, path):
    temporary = path + ".tmp"
    data.to_csv(temporary, index=False, na_rep="N/A")
    os.replace(temporary, path)


def checkpoint_step(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def insert_demonstration(model, transitions, copies=1):
    for _ in range(copies):
        for obs, action, reward, after, done in transitions:
            model.replay_buffer.add(obs[None], after[None], np.array([action]),
                                    np.array([reward]), np.array([done]), [{}])


def clone_actions(model, transitions, updates=1000):
    """Train Q-network action preferences using cross-entropy on eight observations."""
    observations = torch.as_tensor(np.stack([t[0] for t in transitions]), device=model.device)
    actions = torch.as_tensor([t[1] for t in transitions], device=model.device)
    optimizer = torch.optim.Adam(model.q_net.parameters(), lr=0.001)
    log = []
    for update in range(updates):
        logits = model.q_net(observations)
        loss = torch.nn.functional.cross_entropy(logits, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 100 == 0 or update == updates - 1:
            log.append(dict(update=update + 1, loss=float(loss.detach()),
                            correct=int((logits.argmax(dim=1) == actions).sum())))
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    # Normal SB3 DQN learning continues with a fresh optimizer after supervised pretraining.
    model.policy.optimizer = torch.optim.Adam(model.q_net.parameters(), lr=ALL_PEG_SETTINGS["learning_rate"])
    return pd.DataFrame(log)


class EpisodeLog(TrainingLog):
    def __init__(self, transitions=None):
        super().__init__()
        self.transitions = transitions

    def _on_step(self):
        actions = [shot["action"] for shot in self.shots] + [int(self.locals["actions"][0])]
        result = super()._on_step()
        if "episode" in self.locals["infos"][0]:
            self.rows[-1]["actions"] = " ".join(map(str, actions))
        if self.transitions and self.num_timesteps % 32 == 0:
            # Uniform replay sampling mixes these eight transitions with actual gameplay.
            insert_demonstration(self.model, self.transitions)
        return result


def train_model_one(output_dir="models/source_only/model1", steps=50_000):
    """Fresh orange-DQN training with the selected historical compact settings."""
    from levels.config_rl import RLConfig
    os.makedirs(output_dir, exist_ok=True)
    settings = dict(DQN_SETTINGS, learning_rate=0.0003)
    train_agent(steps, model_path=os.path.join(output_dir, "last_model.zip"),
                results_dir=output_dir, config=RLConfig(observation="compact"),
                dqn_settings=settings, verbose=0)
    return os.path.join(output_dir, "best_model.zip")


def train_initial_all_pegs(output_dir, steps=90_000, on_checkpoint=None):
    """Standard DQN only; the optional hook evaluates saved checkpoints.

    The assisted trainer keeps its original 90k prerequisite and no hook.
    An empty output directory starts random weights and an empty replay buffer.
    """
    from levels.config_rl import RLConfig
    if steps < 4 or steps % 4:
        raise ValueError("Initial training steps must be positive and divisible by four.")
    torch.set_num_threads(1)
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(output_dir, "checkpoint_*.zip")), key=checkpoint_step)
    env = TrainingMonitor(PeggleEnv(RLConfig(goal="all", max_shots=10,
                                          free_balls=False, n_actions=21)), SEED)
    try:
        if paths:
            model = DQN.load(paths[-1], env=env, device="cpu")
            model.load_replay_buffer(os.path.splitext(paths[-1])[0] + ".pkl")
            env.boards = np.random.default_rng(1_000_000 + SEED + model.num_timesteps)
            model.get_env().seed(1_000_000 + SEED + model.num_timesteps)
        else:
            model = DQN("MlpPolicy", env, seed=SEED, device="cpu", **ALL_PEG_SETTINGS)
        boundaries = sorted({steps, *range(10_000, steps, 10_000),
                             *(s for s in (25_000, 50_000) if s < steps)})
        log = TrainingLog()
        for target in boundaries:
            if target <= model.num_timesteps:
                continue
            model.exploration_schedule = LinearSchedule(1.0, 0.05, 60_000 / target)
            begin = len(log.rows)
            model.learn(target - model.num_timesteps, reset_num_timesteps=False,
                        callback=log, log_interval=None)
            path = os.path.join(output_dir, f"checkpoint_{model.num_timesteps}")
            if os.path.exists(path + ".zip"):
                raise FileExistsError(path)
            model.save_replay_buffer(path + ".pkl")
            model.save(path + ".zip")
            write_csv(pd.DataFrame(log.rows[begin:]), path + "_episodes.csv")
            print(f"Initial all-peg stage saved at {model.num_timesteps:,} shots.", flush=True)
            if on_checkpoint is not None:
                on_checkpoint(model, path + ".zip")
        return os.path.join(output_dir, f"checkpoint_{model.num_timesteps}.zip")
    finally:
        env.close()


def standard_validation_key(row):
    """Original a1511cc all-peg selection rule, using the current metric names."""
    winning_shots = row["average_winning_shots"] if row["wins"] else 0
    return (row["greedy_win"], row["wins"], row["greedy_pegs"], row["mean_pegs"],
            -row["greedy_shots"] if row["greedy_win"] else 0, -winning_shots)


def train_standard_all_pegs(output_dir="models/source_only/standard_all", steps=100_000,
                            validation_games=50):
    """Train an independent all-peg DQN; never load historical weights or demos."""
    from levels.evaluate import evaluate_standard_all_pegs
    if steps < 4 or steps % 4 or validation_games < 1:
        raise ValueError("Use positive steps divisible by four and positive validation games.")
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError("Fresh standard training needs a new empty output folder.")
    os.makedirs(output_dir, exist_ok=True)
    settings = dict(seed=SEED, policy="MlpPolicy", inputs=78, actions=21, shot_limit=10,
                    free_balls=False, peg_reward=5, shot_penalty=2, completion_bonus=100,
                    target_timesteps=steps, exploration_decay_steps=60_000,
                    validation_seed=45_042, validation_games=validation_games,
                    training_method="Standard DQN; random weights; no demonstrations",
                    **ALL_PEG_SETTINGS)
    write_csv(pd.DataFrame([settings]), os.path.join(output_dir, "training_settings.csv"))
    scores = []

    def validate(model, checkpoint):
        # This evaluator uses local RNGs and greedy network predictions, so it
        # does not change training exploration or import demonstration helpers.
        summary, games, shots = evaluate_standard_all_pegs(
            model, episodes=validation_games, seed=45_042)
        summary.update(checkpoint=checkpoint, timesteps=model.num_timesteps)
        scores.append(summary)
        stem = os.path.splitext(checkpoint)[0] + "_validation"
        write_csv(games, stem + "_games.csv")
        write_csv(shots, stem + "_shots.csv")
        write_csv(pd.DataFrame(scores), os.path.join(output_dir, "checkpoint_evaluation.csv"))
        print(f"Standard validation {model.num_timesteps:,}: "
              f"{summary['greedy_pegs']}/77 greedy, {summary['wins']}/{validation_games} wins.", flush=True)

    train_initial_all_pegs(output_dir, steps, on_checkpoint=validate)
    chosen = max(scores, key=standard_validation_key)
    best_path = os.path.join(output_dir, "best_model.zip")
    shutil.copyfile(chosen["checkpoint"], best_path)
    shutil.copyfile(os.path.splitext(chosen["checkpoint"])[0] + ".pkl",
                    os.path.join(output_dir, "best_model.pkl"))
    write_csv(pd.DataFrame([chosen]), os.path.join(output_dir, "selected_model.csv"))
    print(f"Selected independent standard DQN: {chosen['checkpoint']}. "
          "Fresh training may not reproduce the historical ten-shot result.", flush=True)
    return best_path


def train_model_two(output_dir="models/source_only/model2", base_steps=90_000,
                    additional_steps=25_000, cloning_updates=1000, validation_games=50,
                    interval=5000):
    """Source-only initial stage, supervised pretraining, then standard DQN replay learning."""
    from levels.demonstation_assisted_dqn import CompletionReward, collect_demonstration, rank
    from levels.evaluate import evaluate_model_two
    if additional_steps < 4 or additional_steps % 4 or interval < 4 or interval % 4:
        raise ValueError("Training interval and additional steps must be multiples of four.")
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "best_model.zip")
    # Existing completed source-only runs are reused without touching historical models.
    if os.path.isfile(best_path):
        return best_path
    settings = dict(seed=SEED, base_steps=base_steps, additional_steps=additional_steps,
                    cloning_updates=cloning_updates, validation_games=validation_games,
                    validation_seed=47_042, checkpoint_interval=interval,
                    replay_demo_copies=100, replay_demo_interval=32,
                    terminal_bonus_per_unused_shot=20, **ALL_PEG_SETTINGS)
    write_csv(pd.DataFrame([settings]), os.path.join(output_dir, "settings.csv"))
    base_path = train_initial_all_pegs(os.path.join(output_dir, "initial_stage"), base_steps)
    folder = os.path.join(output_dir, "demonstration_stage")
    os.makedirs(folder, exist_ok=True)
    transitions = collect_demonstration()
    env = Monitor(CompletionReward())
    torch.set_num_threads(1)
    paths = sorted(glob.glob(os.path.join(folder, "checkpoint_*.zip")), key=checkpoint_step)
    rows = []
    scores_path = os.path.join(output_dir, "validation.csv")
    if os.path.isfile(scores_path):
        rows = pd.read_csv(scores_path).to_dict("records")
    try:
        if paths:
            model = DQN.load(paths[-1], env=env, device="cpu")
            model.load_replay_buffer(os.path.splitext(paths[-1])[0] + ".pkl")
        else:
            model = DQN.load(base_path, env=env, device="cpu")
            model.replay_buffer.reset()
            model.learning_starts = model.num_timesteps + 1000
            cloning = clone_actions(model, transitions, updates=cloning_updates)
            write_csv(cloning, os.path.join(output_dir, "supervised_updates.csv"))
            insert_demonstration(model, transitions, copies=100)
        model.get_env().seed(SEED + model.num_timesteps)
        log = EpisodeLog(transitions)
        targets = sorted({base_steps + additional_steps,
                          *range(base_steps + interval, base_steps + additional_steps, interval)})
        for target in targets:
            path = os.path.join(folder, f"checkpoint_{target}.zip")
            if target > model.num_timesteps:
                begin = len(log.rows)
                model.exploration_schedule = LinearSchedule(0.05, 0.05, 0.5)
                model.learn(target - model.num_timesteps, callback=log,
                            reset_num_timesteps=False, log_interval=None)
                if os.path.isfile(path):
                    raise FileExistsError(path)
                model.save_replay_buffer(os.path.splitext(path)[0] + ".pkl")
                model.save(path)
                write_csv(pd.DataFrame(log.rows[begin:]), os.path.splitext(path)[0] + "_episodes.csv")
            if not any(row["checkpoint"] == path for row in rows):
                candidate = DQN.load(path, device="cpu")
                score, _, _ = evaluate_model_two(candidate, episodes=validation_games, seed=47_042)
                score.update(checkpoint=path, timesteps=target)
                rows.append(score)
                write_csv(pd.DataFrame(rows), scores_path)
        chosen = max(rows, key=rank)
        shutil.copyfile(chosen["checkpoint"], best_path)
        write_csv(pd.DataFrame([chosen]), os.path.join(output_dir, "selected_model.csv"))
        print(f"Selected {chosen['checkpoint']} using validation games.", flush=True)
        return best_path
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Train orange, standard all-peg, or assisted all-peg DQN.")
    parser.add_argument("model", choices=["orange", "standard", "all"],
                        help="standard has no demonstrations; all retains the assisted training command")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, help="Orange/standard training steps, or assisted additional steps")
    parser.add_argument("--base-steps", type=int, default=90_000)
    args = parser.parse_args()
    if args.model == "orange":
        train_model_one(args.output or "models/source_only/model1", args.steps or 50_000)
    elif args.model == "standard":
        train_standard_all_pegs(args.output or "models/source_only/standard_all", args.steps or 100_000)
    else:
        train_model_two(args.output or "models/source_only/model2", args.base_steps, args.steps or 25_000)


if __name__ == "__main__":
    main()
