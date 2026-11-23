For the cleaned final submission, use [FINAL_PROJECT_GUIDE.md](FINAL_PROJECT_GUIDE.md). It lists the current demonstration commands, required DQN files, and the code audit. The experiment commands below describe the development history. Completed experiment entry points have been reduced to notebook compatibility helpers.

# CPE 595 Peggle Reinforcement Learning

Open a terminal in the project folder, where `play.py` and `main.ipynb` are located. The scripts use paths relative to this folder. Keep `levels/`, `local/PegglePy-main/`, `models/`, and `results/` in place.

Install the dependencies with Python 3.12, preferably in a virtual environment:

```text
python -m pip install -r requirements.txt
python play.py
```

Enter **1** for DQN, which targets the 25 orange pegs on Level 1, or **2** for PPO, which targets all 77 pegs. The menu loads the trained checkpoints. Open `main.ipynb` with the same Python environment to view the reports and gameplay demonstrations. Its default settings load the saved results.

The first three PPO versions, their comparison tables, and graphs are preserved in `results/ppo_all_pegs/versions/`. Their selected ten-shot checkpoint remains the fallback when Version 4 is unavailable. The earlier tuning results are recorded in `previous_experiments.csv`.

| PPO version | Training reached | Selected checkpoint | Deterministic pegs | Mean over 100 sampled games | Sampled wins |
|---|---:|---:|---:|---:|---:|
| 1: Initial | 25,088 | 15,360 | 64/77 | 59.42 | 0/100 |
| 2: Entropy 0.02 | 50,176 | 45,568 | 69/77 | 61.58 | 0/100 |
| 3: Longer training | 150,016 | 105,472 | 70/77 | 65.39 | 0/100 |

All three comparisons use ten shots and the same fixed layout. Version 3 continued from Version 2 in two stages. Its 146,432-step checkpoint had a higher sampled mean of 69.45 but cleared 69 pegs deterministically; its selection follows the deterministic rule. The selected checkpoint's best sampled game cleared 74 pegs. Across Version 3's training and checkpoint evaluations, the highest recorded clearance was 76; none of those games cleared all 77.

A separate fresh 15-shot experiment reached 50,176 steps, cleared 68 pegs deterministically, and averaged 65.73 pegs with zero wins in 100 sampled games. This uses a different shot budget and is reported separately. Successful-game shot statistics are N/A for these PPO evaluations.

Open `main.ipynb` for the full comparison, training curves, checkpoint results, shot logs, and gameplay images. It loads saved results by default. The original DQN remains at 92.60% orange clearance and 12/100 wins.

Optional commands for tests and new training runs:

```text
python -m pytest levels/test_environment.py -q
python -m levels.train_dqn
python -m levels.train_ppo
```

The PPO training command starts a new continuation from Version 2 toward 150,000 total timesteps. Edit the run name and target in `levels/train_ppo.py` first if needed; existing runs are protected from overwriting. The notebook and `play.py` load saved models by default.

## PPO Version 4

Version 4 uses `RLConfig(goal="all", free_balls=True)`. The default remains `free_balls=False`, so DQN and PPO Versions 1–3 retain their original behavior. DQN still has no scoring or free balls.

Only Version 4 uses the new 155-value observation: 77 orange flags, 77 hit flags, and remaining balls divided by 241. This differs from the DQN input ordering. The conservative scale 241 permits more than ten remaining balls without clipping. The same 77 coordinates and physics are used.

Simplified Peggle-style scoring sums blue (10) and orange (100) base values multiplied by the post-hit Fever multiplier, then multiplies by the number of newly hit pegs in that shot. Multipliers are 1, 2, 3, 5, and 10 as orange counts fall; 10x remains after the oranges are cleared. Only newly hit pegs score, and point thresholds reset each shot. Scores of 25,000, 75,000, and 125,000 earn one, two, or three balls respectively. PegglePy includes additional scoring mechanics that this assignment does not reproduce.

The learning reward remains `5 * new_pegs - 2`, plus 100 for completion. Every shot fired is counted, including shots earned through scoring. This version starts with ten balls and does not have the original strict ten-shot limit. The input space and rules both change, so a comparison cannot isolate the effect of free balls.

Version 4 outputs are kept separately in `results/ppo_version4/`. The menu loads its verified selected checkpoint when available and otherwise falls back to Version 3. Neither the menu nor the notebook trains automatically. The appended Version 4 section is independent of the preserved earlier notebook cells.

Version 4 reached 100,352 steps across three stages, starting fresh. The selected checkpoint is also at 100,352. It clears 74/77 deterministically in 11 actual shots, with one earned ball. Its 100 sampled games average 71.20 pegs (92.47%), 11.02 shots, and 1.02 free balls, with zero wins. Its recorded best sampled example is 76/77 in 11 shots, seed 50076. Successful-game evaluation shot statistics are N/A.

The training log records one 77-peg completion at step 94,373 in 11 shots, using one free ball. That intermediate policy and its per-shot action trace were not saved. This training-only completion is therefore distinguished from the selected checkpoint's zero evaluation wins, and the demonstration does not claim to reproduce it.

Only the new Version 4 notebook cells were executed for this update. The preceding 42 cells and their outputs, the earlier models/results, and all original Level 1 source data remain unchanged. DQN's in-memory replay of the original 100 seeded games still matches every saved episode, shot, reward, and orange assignment: 92.60% clearance and 12 wins.

For a new, separately named 25,000-step Version 4 run:

```text
python -m levels.ppo_version4
```

Edit the run name in that script before repeating a run. Existing outputs are protected from overwriting.

## PPO Version 5

Version 5 tests 41 angles spaced 3° apart, from −60° to +60°. It uses Version 4's original Level 1 geometry, scoring, free balls, 155 inputs, PPO rewards, and training settings. Version 5 starts with a fresh policy because its action space has changed.

The appended notebook section reads the Version 5 results from `results/ppo_version5/`. The checkpoint table uses 100 validation games per checkpoint. The final comparison uses a separate set of 100 seeds and verifies that Versions 4 and 5 see the same orange assignments. The deterministic game, sampled mean, and best sampled game are reported separately. Any training wins are saved with their policy, starting colors, and action sequence.

Version 5 reached 100,352 steps. The selected 91,136-step checkpoint cleared all 77 pegs once in sampled validation, using seed 40107, 11 shots, and one earned ball. Its final matched evaluation cleared 56 pegs deterministically and averaged 64.65 pegs with zero wins in 100 sampled games. Version 4 averaged 71.20 pegs on the same sampled seeds, so menu option 2 retains Version 4. The notebook includes the verified validation win as a separate replay.

## PPO Version 4 Improved

Experiments A and B each resume the original 100,352-step Version 4 checkpoint. Both use its 21 angles, original board, physics, 155 inputs, and free-ball rules. A keeps the original PPO reward. B adds two reward points per newly hit peg when the shot begins with at most ten pegs remaining.

The report selects checkpoints on 50 sampled validation games, then compares them with the original Version 4 on 100 separate matched test seeds. Every successful game has a shot log and its starting colors. Training wins also include a policy snapshot. Results are under `results/ppo_v4_improved/` and the appended notebook section loads them directly.

Both experiments reached 199,680 total steps. On the new matched test seeds, original V4 averaged 70.80 pegs, A averaged 75.83, and B averaged 74.85. All three had zero wins. A selected the 156,672-step checkpoint by mean validation clearance. B selected the 125,952-step checkpoint after one validation win in ten shots. The validation rule chose B for menu option 2 before the final test. The notebook replays B's validation win and labels it separately from the final evaluation.

## PPO V4 Further Improvements

The late-game diagnostics sampled 100 games per saved A and B policy. Peg 32 remained in 192 of those 200 games. The 21-angle probe could not hit it when it was the only peg left, but the agents sometimes hit it earlier while other pegs were still on the board. The diagnostic logs and board plot are in `results/ppo_v4_further/diagnostics/`.

C and D each continue the same 125,952-step Improved B checkpoint for 99,840 additional steps. C increases the completion bonus from 100 to 300. D keeps the B reward and reduces the PPO entropy coefficient from 0.02 to 0.005. The 21 angles, 155 inputs, board, physics, starting balls, and free-ball rules are shared.

The new reports and exact winning action logs live in `results/ppo_v4_further/`. Each candidate checkpoint is ranked on 100 validation seeds. All four A–D policies are then compared on a separate set of 100 matched final seeds. The notebook loads saved reports and replays, and the menu uses the validation selection.

Both trials reached 225,792 steps. C selected the 197,632-step checkpoint and D selected the 166,912-step checkpoint. On the new final seeds, A/B/C/D averaged 75.86/74.70/75.95/75.97 pegs respectively, with zero wins for each. D recorded two training-only wins in 11 shots. Option 2 loads `models/ppo_v4_improved_D_best.zip`, which won the validation tie on mean clearance. The one-ball probes point to investigating shot order before peg 32 is isolated. Experiment E was deferred.

## PPO V4 Improved F

Exact replays of four prior full clears found that three hit peg 32 with more than ten pegs remaining. In all four, the same angle missed when peg 32 was isolated. F tests a +10 learning reward for a new hit on peg 32 when the shot starts with more than ten pegs left. Peggle points and free balls follow the existing rules.

F and the continued-training control each load the same 166,912-step Improved D checkpoint and add 99,840 steps. The control keeps D's reward. Checkpoints use 50 validation games. The saved D policy, the control, and F are evaluated on the same 100 held-out seeds after validation choices are frozen. New logs, figures, and replay evidence are in `results/ppo_v4_improved_F/`.

Both continuations trained to 266,752 steps and selected the 177,152-step checkpoint. On the matched test, D averaged 75.91 pegs, the control 75.99, and F 76.00. Each had zero wins and left peg 32 in every final game. F hit peg 32 on three validation runs across its checkpoints, all on the same seed. Option 2 retains Improved D because F's 0.12-peg validation gain fell below the predeclared 0.25-peg margin. The notebook includes the four historical winning trajectories as an analysis table and replays F's recorded 76/77 test game.

## PPO — 121-Angle Resolution

The new policy uses all integer angles from −60° through +60°. It starts from scratch with the original V4 gameplay rules, 155 input values, reward, and two 64-unit layers. The six saved states with only peg 32 remaining each received 121 diagnostic shots. All 726 missed in those specific states.

Run these commands in the project root using the project Python environment.

```text
python -m levels.ppo_121_angles probe
python -m levels.ppo_121_angles train --steps 100000
python -m levels.ppo_121_angles report
```

The training command also resumes an interrupted run. `--steps` sets the total budget, including steps already saved. Stable-Baselines3 uses 512-step rollouts, so a 100,000-step request completes at 100,352. The largest allowed target is 200,000, which completes at 199,680. The model and training history are saved about every 10,000 steps in `models/ppo_121_angles/` and `results/ppo_121_angles/`. The latter contains `latest_checkpoint.csv` and the per-checkpoint validation results. The checkpoint restores the policy and saved optimizer state. A resumed process starts a fresh environment and may sample a different training trajectory. Any work after the latest committed checkpoint may be repeated.

`report` freezes checkpoint and menu selection on 50 validation seeds before evaluating the four policies on 100 matched held-out seeds. The notebook loads the resulting plots, tables and separate deterministic and sampled replays. The reports also keep any training, validation and final wins separate. Repeating the report command uses any completed held-out game logs already saved.

The run completed 100,352 timesteps and selected its final checkpoint. It cleared 54/77 deterministically and averaged 65.27/77 across 100 sampled games, with a best of 75/77 and zero wins. The two full clears during training were replayed and verified. On the same held-out seeds, V4 averaged 71.40 with one win, V5 averaged 64.18 with zero wins, and Improved D averaged 75.90 with zero wins. Improved D averaged 76.00 on validation versus 65.56 for the new policy, so menu option 2 remains on Improved D.
