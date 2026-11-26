# CPE 595 Peggle DQN Project

This project provides two trained DQN policies on the original PegglePy Level 1 board, plus their source training procedures.

| Model | Objective | Inputs | Selected pretrained checkpoint |
|---|---|---|---|
| DQN Model 1 | Clear the 25 orange pegs | 155 | `models/level1_improved/best_model.zip` |
| DQN Model 2 | Clear all 77 pegs | 78 | `models/dqn_eight_shot_experiment/best_new.zip` |

Both models use 21 launch angles, a maximum of ten shots, the original physics and no free balls. The two selected checkpoints are included for reproducible saved-policy playback. Experimental checkpoints and replay buffers are excluded from the submission commit.

## Install and run

Use Python 3.12. Create or activate a Python environment, then run these commands from the project root.

```text
python -m pip install -r requirements.txt
python play.py
```

First choose the objective.

```text
1. DQN Model 1 - Clear all orange pegs
2. DQN Model 2 - Clear all 77 pegs
```

Then choose the action.

```text
1. Use existing pretrained model
2. Train a new model from scratch
3. Back to model selection
```

Pretrained playback loads exactly the corresponding checkpoint listed above, runs its neural-network predictions and displays an animated recording of actual gameplay. A missing or incompatible checkpoint produces an error and returns to the action menu. This option never starts training automatically.

Fresh training creates a unique empty directory under `models/retrained/model1/run_*` or `models/retrained/model2/run_*`. Each request starts from new random network weights. It saves checkpoints, validation results and the selected new model in that run directory. After training, the program prints an actual 100-game evaluation, saves its CSVs in the same run directory, and demonstrates the newly trained policy. A previous run is never reused by this menu choice. Historical checkpoints are preserved.

Training can take considerably longer than playback. A fresh model may perform differently from the historical checkpoints. The program reports its measured outcome and does not assume an eight-shot win or the original orange-clearance result.

## What the models learned

Model 1's saved 45,000-step checkpoint historically averaged 92.60% orange clearance, or 23.15 of 25 targets, with 12 wins out of 100 games. Its selected compact training settings and original reward remain in `train_dqn.py` and `config_rl.py`.

Model 2's selected 110,000-step checkpoint greedily clears all 77 pegs in eight shots. Its learned angles are `[-6, 12, -30, -60, -54, 0, -42, -42]`. Its saved matched epsilon-greedy evaluation at epsilon 0.05 recorded 78/100 wins, including 69 eight-shot wins and nine nine-shot wins.

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

`play.py` contains the two menus. `train_dqn.py` trains both models. `demonstation_assisted_dqn.py` loads and plays Model 2 and constructs demonstration transitions. `evaluate.py`, `peggle_env.py`, `config_rl.py` and `visualize_data.py` provide evaluation, the shared environment, configuration and rendering. The two `test_*.py` files check observations, rewards, physics, checkpoints, menus and isolated fresh training.

The runtime also needs `local/PegglePy-main/local/vectors.py`, `ball.py`, `collision.py`, and `local/PegglePy-main/levels/Level 1.lvl`, in their existing locations. The full original PegglePy source remains available locally. Its separate graphical game may require its own optional dependencies. The final DQN interface uses Matplotlib recordings.

Historical DQN/PPO checkpoints, CSVs, figures, search evidence and notebook outputs are preserved. The independent simulator search and the demonstration-assisted trained DQN are distinct stages of the research. Fixed Level 1 geometry and limited recovery performance restrict conclusions about new levels. Eight shots is the shortest clear found in our experiments, with no mathematical optimality proof.

`main.ipynb` contains the recorded research results. Removed PPO scripts mean some historical code cells require their archived Git revision to rerun. Direct archived imports occur in one-based cells 33, 44, 55, 64, 74 and 84; later dependent cells need those setup cells. Existing outputs remain readable. The final appended eight-shot DQN section reads saved results and includes a read-only optional checkpoint check.

`requirements.txt` records the compatible package versions tested for this submission. Short isolated training smoke tests verify initialization, actual optimizer updates, checkpoint reload and playback. They do not establish that a full fresh run will reproduce historical performance.
