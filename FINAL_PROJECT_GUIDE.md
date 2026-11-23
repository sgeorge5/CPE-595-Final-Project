# Final project code guide

Run commands from the original `Final Project` directory with the project Python environment. `main.ipynb` is the report entry point. Its earlier cells and outputs are preserved. The appended final comparison reads saved results.

## Demonstrations

| Model | Command | Saved checkpoint | Greedy game |
|---|---|---|---|
| PPO without free balls | `python -m levels.selected_models ppo-v3` | `models/ppo_version3_best.zip` | 70/77, ten shots |
| PPO with free balls | `python -m levels.selected_models ppo-d` | `models/ppo_v4_improved_D_best.zip` | 76/77, eleven shots, one earned ball |
| DQN Model 2 | `python -m levels.dqn_all_pegs play` | `models/dqn_all_pegs_best.zip` | 77/77, ten shots |
| Simulator Search — Eight-Shot Solution | `python -m levels.dqn_shot_efficiency play` | Saved candidate 12466 and its shot log | 77/77, eight shots |

The three all-peg policies are also available through `python -m levels.selected_models dqn`, `ppo-v3`, or `ppo-d`. The command creates an animated recording through the shared visualizer and opens it. Add `--no-browser` to save it without opening a browser. Add `--check-only` to run one greedy game without writing files.

`play.py` retains its original two options and checkpoint selection. The orange-only DQN remains part of the mid-stage work. DQN actions in policy demonstrations come from `model.predict`. The eight-shot demonstration replays the saved simulator actions and prints its search label.

## Code organization and audit

`file_audit.csv` lists every current project Python file, its purpose, keep/remove decision, and concrete notebook or Python callers. The bundled PegglePy files were also inspected and remain together in their original directory. The virtual environment contains installed dependencies.

The main retained implementation files are `config_rl.py`, `peggle_env.py`, `evaluate.py`, `visualize_data.py`, `train_dqn.py`, `train_ppo.py`, `dqn_all_pegs.py`, and `play.py`. Their roles cover shared settings and physics, observations and rewards, evaluation, rendering, the original orange-DQN training, PPO training, and all-peg DQN training/playback. The generic DQN and PPO learning implementations remain available for explaining the completed work. Normal demonstration commands only load checkpoints.

`selected_models.py` provides the three report demonstrations. `dqn_shot_efficiency.py` now contains only sequence replay helpers and the saved eight-shot demonstration.

The following small modules retain the exact names imported by earlier notebook cells and behavioral tests.

| Module | Retained functionality |
|---|---|
| `ppo_version4.py` | Read-only recorded-game replay for the existing notebook |
| `ppo_version5.py` | 41-angle configuration and historical selection ranking |
| `ppo_v4_improved.py` | A/B configurations, model paths, original checkpoint verification |
| `ppo_v4_further.py` | A–D configurations, hyperparameters and checkpoint verification |
| `ppo_v4_shot_order.py` | F configuration, historical ranking and checkpoint verification |

Completed experimental training, optimization, and report-writing entry points were removed from these modules. Their removal also prevents obsolete commands from overwriting historical reports. The earlier cleanup removed `ppo_121_angles.py`, whose notebook section reads saved files directly. This cleanup additionally removed `ppo_endgame_diagnostics.py`. Its sole remaining state-signature helper now lives in `test_environment.py`, and its two test callers import it there. Its tests now inspect the saved checkpoint, probe evidence, and validation logs. Shot-search tests now verify the saved search evidence and actual simulator replay. Existing behavioral tests remain available.

Episode summarization moved from `train_ppo.py` into `evaluate.py`. The old PPO import remains compatible through re-export. The all-peg DQN can now import evaluation without importing PPO training.

The bundled `local/PegglePy-main` source is retained in full to preserve original gameplay. Its adapter loads `vectors.py`, `ball.py`, and `collision.py` and supplies headless settings and peg objects. Its coordinates come from `levels/Level 1.lvl`.

## Minimum DQN Model 2 playback files

Keep the relative paths below when assembling a smaller code submission.

- `levels/dqn_all_pegs.py`
- `levels/config_rl.py`
- `levels/peggle_env.py`
- `levels/evaluate.py`
- `levels/visualize_data.py`
- `local/PegglePy-main/local/vectors.py`
- `local/PegglePy-main/local/ball.py`
- `local/PegglePy-main/local/collision.py`
- `local/PegglePy-main/levels/Level 1.lvl`
- `models/dqn_all_pegs_best.zip`
- `requirements.txt` for installing dependencies

The `levels` directory works as a Python namespace package from the project root. Playback creates its own output directory. It needs the installed packages from `requirements.txt`, including Pillow supplied with Matplotlib. The saved SB3 checkpoint is necessary for playback. It is a model artifact.

To include the full notebook, keep all its referenced historical result folders, compatibility modules, the original orange-DQN files, and the remaining model checkpoints. Continuing prior training additionally needs the saved replay buffers and original history. This minimal playback list does not promise to reproduce all training experiments or notebook sections.

## Final comparison

`final_comparison.csv` uses the historical results shown below, with an explicit seed set and action method in each row. These evaluations are descriptive comparisons across their recorded procedures. Scoring-enabled PPO also received additional training and reward changes.

| Model | Inputs | Free balls | Greedy clearance | Mean stochastic clearance | Wins |
|---|---|---|---|---|---|
| PPO V3 | 78 | No | 70/77 in ten shots | 65.39/77 | 0/100 |
| PPO Improved D | 155 | Yes | 76/77 in eleven shots | 75.90/77 | 0/100 |
| DQN Model 2 | 78 | No | 77/77 in ten shots | 74.71/77 | 73/100 |

The eight-shot result came from the bounded simulator search. The learned DQN's greedy result remains ten shots. The search established the shortest sequence found within its 20,000-candidate budget.

## Verification

For this cleanup, the baseline and final suite each passed 60 tests, with five training tests excluded. Tests of deleted experimental orchestration were replaced with saved-artifact and replay checks. No learning or extensive sequence search was run during cleanup.

The original orange-DQN evaluation is checked against its original per-episode, per-shot and arrangement logs in memory. It must reproduce 92.60% orange clearance and 12/100 wins. The main model configurations, original menu choices, saved eight-shot replay, notebook imports, and protected-file checksums are checked separately.

Legacy training commands described in earlier README or notebook development sections document the original experiments. The cleaned experimental compatibility modules expose only their retained helpers. Use the demonstration commands above for the final presentation.


## Search method retained for the report

The saved 20,000-candidate search is documented by `results/dqn_shot_efficiency/search_candidates.csv`. Every row records the candidate ID, generation method, proposed angles, actual angles played, clearance, and best shot count so far. The two full clears have saved starting colors, shot logs, screenshots and animations in `solutions/candidate_12466/` and `solutions/candidate_12574/`. The first used eight shots and hit peg 32 on shot three.

The original search used color seed 50042, mutation seed 592 plus candidate ID, a pool of 32 high-clearance candidates per length, and a forward beam of 60 distinct hit states. The first 5,000 trials tested deletions, one-angle changes, swaps, original prefixes, and random mutations. The extension swept possible final angles after strong prefixes, expanded the beam one shot at a time over all 21 legal angles, then returned to mutations. The remaining budget emphasized lengths shorter than the current best. Each proposed sequence reset the actual environment, kept the maximum at ten, and stopped when its sequence ended or all pegs were cleared.

The search can be understood as this procedure. It is documentation for the completed experiment.

```text
Start with the verified ten-shot DQN sequence.
For each candidate up to the fixed budget
    propose a deletion, replacement, swap, mutation or beam expansion
    reset the original Level 1 environment with seed 50042
    fire the proposed angles until success or the sequence ends
    record the actual shot results
    retain promising candidates and distinct cleared-peg states
    if a shorter full clear is found
        replay it from another fresh reset
        save its angles, shot log and starting arrangement
Continue testing shorter lengths within the remaining budget.
```

`play_sequence` in `dqn_shot_efficiency.py` retains the actual simulator evaluation step. The saved candidate CSV supplies the exact historical proposals for replay. The cleanup does not retain the completed reward-continuation or optimization orchestrators. The DQN playback still obtains each action from its trained neural network.

## Current Python-file audit

| Python file | Purpose | Keep or remove | Reason |
|---|---|---|---|
| levels\config_rl.py | Model settings, observation choices and Level 1 coordinate loading | Keep | Shared implementation required by retained code |
| levels\dqn_all_pegs.py | Main all-peg DQN training, checkpoint loading, evaluation and playback | Keep | Required demonstration or learning entry point |
| levels\dqn_shot_efficiency.py | Verified simulator sequence replay, with fresh-reset helper | Keep | Required demonstration or learning entry point |
| levels\evaluate.py | Original orange-DQN evaluation and shared episode summaries | Keep | Shared implementation required by retained code |
| levels\peggle_env.py | Shared Gymnasium observations, physics adapter, rewards and shot simulation | Keep | Shared implementation required by retained code |
| levels\ppo_v4_further.py | Historical notebook A?D settings including Improved D | Keep | Shared implementation required by retained code |
| levels\ppo_v4_improved.py | Historical notebook A/B settings and checkpoint verification | Keep | Shared implementation required by retained code |
| levels\ppo_v4_shot_order.py | Historical notebook F settings and regression ranking | Keep | Shared implementation required by retained code |
| levels\ppo_version4.py | Read-only historical notebook replay helper | Keep | Shared implementation required by retained code |
| levels\ppo_version5.py | Historical notebook angle configuration and regression ranking | Keep | Shared implementation required by retained code |
| levels\selected_models.py | Three saved report models, configured and demonstrated individually | Keep | Required demonstration or learning entry point |
| levels\test_dqn_all_pegs.py | Regression checks for dqn all pegs | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_dqn_shot_efficiency.py | Regression checks for dqn shot efficiency | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_environment.py | Regression checks for environment | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_ppo_121_angles.py | Regression checks for ppo 121 angles | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_ppo_v4_further.py | Regression checks for ppo v4 further | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_ppo_v4_improved.py | Regression checks for ppo v4 improved | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_ppo_v4_shot_order.py | Regression checks for ppo v4 shot order | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_ppo_version4.py | Regression checks for ppo version4 | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\test_ppo_version5.py | Regression checks for ppo version5 | Keep | Preserves behavioral and saved-checkpoint verification |
| levels\train_dqn.py | Original orange-DQN training and checkpoint explanation | Keep | Shared implementation required by retained code |
| levels\train_ppo.py | Shared PPO implementation and preserved notebook training imports | Keep | Shared implementation required by retained code |
| levels\visualize_data.py | Board extraction, plots and actual trajectory recordings | Keep | Shared implementation required by retained code |
| play.py | Original two-option menu and unchanged checkpoint selection | Keep | Notebook cells 32 import this module (zero-based). Required demonstration or learning entry point |
| levels/ppo_endgame_diagnostics.py | State comparison helper used only by tests | Removed this cleanup | Moved unchanged state_signature into existing test_environment.py. Updated its two test imports. No notebook or runtime caller. |
| levels/ppo_121_angles.py | Completed 121-angle experiment trainer | Already removed | Historical notebook reads saved CSVs. Saved checkpoints and all research evidence remain. |
