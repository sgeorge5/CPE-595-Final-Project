# CPE 595 Peggle DQN Project

This project provides three trained DQN policies on the original PegglePy Level 1 board, plus their source training procedures.

| Model | Objective | Inputs | Selected pretrained checkpoint |
|---|---|---|---|
| DQN Model 1 | Clear the 25 orange pegs | 155 | `models/level1_improved/best_model.zip` |
| Standard all-peg DQN | Clear all 77 pegs | 78 | `models/dqn_all_pegs_best.zip` |
| Demonstration-assisted DQN | Clear all 77 pegs | 78 | `models/dqn_eight_shot_experiment/best_new.zip` |

All three models use 21 launch angles, a maximum of ten shots, the original physics and no free balls. The three selected checkpoints are included for reproducible saved-policy playback. Experimental checkpoints and replay buffers are excluded from the current repository tree.

## Install and run

Use Python 3.12. Create or activate a Python environment, then run these commands from the project root.

```text
python -m pip install -r requirements.txt
python play.py
```

First choose the objective.

```text
1. Standard DQN - Clear orange pegs
2. Standard DQN - Clear all 77 pegs
3. Demonstration-assisted DQN - Clear all 77 pegs
```

Then choose the action.

```text
1. Use existing pretrained model
2. Train a new model from scratch
3. Back to model selection
```

Pretrained playback loads exactly the corresponding checkpoint listed above and uses deterministic neural-network predictions. A missing or incompatible checkpoint produces an error; use Back to return to model source selection. This option never starts training automatically.

Fresh training creates a unique empty directory under `models/retrained/model1/run_*`, `model2/run_*` or `model3/run_*`. Each request starts from new random network weights. It saves checkpoints, validation results and the selected new model in that run directory. After training, the program prints an actual 100-game evaluation, saves its CSVs in the same run directory, and plays the newly selected policy in the chosen execution mode. A previous run is never reused by this menu choice. Historical checkpoints are preserved.

Training can take considerably longer than playback. A fresh model may perform differently from the historical checkpoints. The program reports its measured outcome and does not assume an eight-shot win or the original orange-clearance result.

## Visual and headless execution

After choosing a model and its source, choose `1` for visual execution, `2` for headless execution, or `3` to go back one menu. For example, `2`, `1`, `2` runs the existing standard all-peg DQN headlessly; `3`, `1`, `1` watches the existing demonstration-assisted DQN live. Choose source `2` instead to start fresh training.

Visual playback shows the original board and the actual recorded ball trajectory after each shot is simulated. A side panel shows the angle, clearance, shots remaining and outcome. No saved CSV, GIF or historical result is needed. Matplotlib uses an interactive backend only when visual mode is explicitly selected. If no interactive backend is available, choose headless execution from the menu.

After choosing Model 1, its source and its visual/headless execution mode, choose the playback arrangement:

```text
1. Random orange peg arrangement
2. Previously verified winning arrangement (seed 50119)
3. Enter a custom seed
4. Back
```

Random mode assigns the 25 orange pegs using a new seed; custom mode reproduces the entered seed. Back returns to the execution-mode menu before loading or training anything. Every game prints its actual seed. All actions remain deterministic DQN predictions. The same checkpoint and seed give identical visual and headless results. Models 2 and 3 retain their original fixed playback behavior. Formal evaluation and checkpoint-selection seeds are unchanged.

Option 2 is labeled in the terminal and visual panel as a **selected demonstration case, not a randomly generated test game**: seed **50119**, episode 78 of `results/level1_improved/evaluation_episodes.csv`, cleared **25/25 in seven shots**. Seed **50045** also cleared **25/25 in eight shots** and can be replayed with the custom-seed option. These selected examples are not unbiased evaluations or guarantees for newly trained policies. Model 1's historical average remains **92.60% clearance, 23.15/25 orange pegs and 12 wins out of 100 games**. Repeated late -60-degree shots also occurred in the historical evaluation.

The optional shortcuts `python play.py --winning-arrangement` (seed 50119) and `python play.py --seed 50045` bypass the arrangement question for Model 1. Seeds are stored in the source or supplied by the user, so playback does not require historical CSV files. These shortcuts do not affect Models 2 and 3.

Visual training observes the actual training environment, including epsilon-greedy exploratory actions. By default, episode 1 and every tenth episode are animated. Change `RENDER_EVERY_N_EPISODES = 10` near the top of `levels/visualize_data.py` to change that interval. Every episode still trains normally. The panel and terminal show measured progress, rewards, epsilon, successes and recent average clearance. Between animated episodes, the window indicates that training continues without animation.

The demonstration-assisted path labels all five stages: initial standard DQN learning, demonstration collection, supervised action-preference updates, demonstration-assisted DQN learning and checkpoint evaluation. Demonstration shots are explicitly labeled as data; supervised updates and headless validation display their stage rather than unrelated gameplay.

Visual training is slower because it draws trajectories and waits for animation. Headless training is recommended for full runs. Headless playback prints shot-by-shot results without opening a window or browser, recording trajectories or generating files. Headless training retains the existing budgets, settings and checkpoint selection, prints periodic progress, and saves new outputs only in the unique run directory.

In the visual window, Space pauses/resumes animation, `+` speeds it up and `-` slows it down. These controls change display speed only. Closing the window continues execution headlessly. At the end of visual playback, close the window to exit. No additional Python packages are required; the live window uses Matplotlib's existing interactive backend (normally TkAgg).

## What the models learned

Model 1's saved 45,000-step checkpoint historically averaged 92.60% orange clearance, or 23.15 of 25 targets, with 12 wins out of 100 games. Its selected compact training settings and original reward remain in `train_dqn.py` and `config_rl.py`.

The independent standard all-peg DQN (menu option 2) uses its original 90,000-step checkpoint and greedily clears 77/77 in ten shots, without demonstrations.

The demonstration-assisted policy (menu option 3, historically called Model 2) uses its selected 110,000-step checkpoint and greedily clears all 77 pegs in eight shots. Its learned angles are `[-6, 12, -30, -60, -54, 0, -42, -42]`. Its saved matched epsilon-greedy evaluation at epsilon 0.05 recorded 78/100 wins, including 69 eight-shot wins and nine nine-shot wins.

The improved model used demonstration-assisted DQN training. Its initialization came from an earlier all-peg DQN trained for 90,000 steps. Eight actual simulator transitions supplied 1,000 supervised action-preference updates. A fresh replay buffer contained 100 copies of those transitions, with another copy added every 32 gameplay steps during standard SB3 DQN training. The run completed 25,000 additional steps and selected the checkpoint after 20,000 additional steps using validation games. The network chooses its own actions during inference.

For a fresh Model 2 run, `train_dqn.py` recreates the initial 90,000-step all-peg stage from random weights before demonstration-assisted learning. It generates the eight demonstration transitions directly through the simulator, without needing historical checkpoints or search CSVs. Model 2's reward is `5 * newly_hit_pegs - 2`, plus `100 + 20 * actual_unused_shots` on completion. Model 1 retains its original orange-only reward. Full fresh-training reproduction of the historical metrics was not performed during this final submission update.

## Other commands

```text
python -m levels.demonstation_assisted_dqn play
python -m levels.demonstation_assisted_dqn play --no-browser
python -m levels.evaluate orange
python -m levels.evaluate all
python -m pytest levels -q
```

The first two commands use the selected pretrained Model 2 without training. Evaluation commands print results in memory. Playback creates a new `results/source_playback/game_*` folder, preserving earlier recordings.

The lower-level training APIs and `python -m levels.train_dqn orange` / `all` remain available for source-only experiments. Their default output is `models/source_only/`. Use the fresh-training menu option for a guaranteed new run directory on every request.

## Files and historical research

`play.py` contains model, source and execution-mode menus. `train_dqn.py` trains the three policies. `demonstation_assisted_dqn.py` loads and plays the assisted policy and constructs demonstration transitions. Its optional viewer argument displays demonstration collection without changing the data. `evaluate.py`, `peggle_env.py`, `config_rl.py` and `visualize_data.py` provide evaluation, the shared environment, configuration and rendering. The two `test_*.py` files check observations, rewards, physics, checkpoints, menus and isolated fresh training.

The runtime also needs `local/PegglePy-main/local/vectors.py`, `ball.py`, `collision.py`, and `local/PegglePy-main/levels/Level 1.lvl`, in their existing locations. The full original PegglePy source remains available locally. Its separate graphical game may require its own optional dependencies. The DQN menu uses optional live Matplotlib visualization; existing notebook recording helpers remain available.

Historical DQN/PPO checkpoints, CSVs, figures, search evidence and notebook outputs are preserved. The independent simulator search and the demonstration-assisted trained DQN are distinct stages of the research. Fixed Level 1 geometry and limited recovery performance restrict conclusions about new levels. Eight shots is the shortest clear found in our experiments, with no mathematical optimality proof.

`main.ipynb` contains the recorded research results. Removed PPO scripts mean some historical code cells require their archived Git revision to rerun. Direct archived imports occur in one-based cells 33, 44, 55, 64, 74 and 84; later dependent cells need those setup cells. Existing outputs remain readable. The final appended eight-shot DQN section reads saved results and includes a read-only optional checkpoint check.

`requirements.txt` records the compatible package versions tested for this submission. Short isolated training smoke tests verify initialization, actual optimizer updates, checkpoint reload and playback. They do not establish that a full fresh run will reproduce historical performance.
