# Simulator search and three-model restoration

The complete historical search source was recovered from local unreachable Git blob
`91f71148aa52d72dee77038917216ea0d834766d` (39,844 bytes). The committed version at
`a1511cc:levels/dqn_shot_efficiency.py` contained only replay functionality.

## What was restored

`levels/simulator_search.py` is a simplified search-only adaptation of the recovered
source, not a claim to reproduce its exact historical candidate order. It reuses:

- Forward beam search with beam width **60**, recovered from the original source.
- Ranking by more cleared pegs, then earlier candidate number, while retaining
  distinct hit-status states at each depth.
- Pools of **32** promising sequences per length.
- Shot deletion, angle replacement, shot swaps, and random mutation.
- A **20,000 unique-candidate maximum**, board seed **50042**, and search seed **592**.

The beam starts from `[[]]`. It never imports or queries a DQN/PPO policy and never
seeds itself with a historical winning sequence. The recovered historical program
also seeded edits from a ten-shot DQN sequence; that dependency and its RL
continuation code were removed. Python's `random` replaces the old NumPy mutation
RNG, and the beam runs before local edits. These changes make the implementation
understandable and independent, but its results/order must not be attributed to
an exact rerun of the old 20,000-candidate experiment.

The search imports `RLConfig` from `levels.config_rl` and `PeggleEnv` from
`levels.peggle_env`. It uses the same original 77-peg Level 1, 21 angles spaced by
six degrees from -60 to +60, ten available shots, 78-value game observations,
no free balls/scoring, gravity .045, launch force 5, velocity cap 10, and dt 1.
No physics, geometry, or observation changes were made.

Complete solutions take precedence over partial clearance, with fewer shots
preferred among full clears. After discovering an eight-shot full clear, remaining
local/mutation trials use at most seven shots, without reducing the environment's
ten-shot allowance. Duplicate proposals are skipped; verification replays are
counted separately. Beam pruning and finite mutation retries do not prove global
optimality.

## Actual search verification

Both historical eight-shot winners reproduced **77/77** through fresh resets:

1. `-6 12 -30 -60 -54 0 -42 -42`
2. `0 -36 0 -54 -18 -12 -18 -24`

These sequences are read from the historical CSV only by the separate `--verify`
function. They are not search starting points.

Only a **63-candidate smoke search**, beam width 2 and depth limit 3, was run.
It evaluated 21 one-shot and 42 two-shot candidates, with no duplicates or budget
overrun. Its best actual result was **43/77 in two shots**, angles `0 -36`, incomplete.
The full 20,000-candidate search was **not** run. Historical results remain unchanged.
New search runs are saved in unique `results/simulator_search_reconstructed/run_*`
folders, including settings, candidate logs, per-length clearance, progress,
and verified solution traces when a full clear is found.

## Three-model gameplay menu

Run `python -B play.py` from the Final Project folder:

| Choice | Model | Existing checkpoint | Verified behavior |
|---|---|---|---|
| 1 | Standard DQN, orange objective | `models/level1_improved/best_model.zip` | Original 92.60% orange clearance and 12/100 wins reproduced exactly |
| 2 | Independent standard DQN, all 77 pegs | `models/dqn_all_pegs_best.zip` | Original 90k checkpoint: 77/77 in ten greedy shots; 73/100 epsilon-.05 wins, mean 74.71/77 |
| 3 | Demonstration-assisted DQN, all 77 pegs | `models/dqn_eight_shot_experiment/best_new.zip` | 77/77 in eight greedy shots; original 78/100 epsilon-.05 result reproduced unchanged |

The existing secondary menu remains pretrained / fresh training / back. All six
pretrained/fresh routes were verified. Pretrained mode never starts training.
Fresh mode always creates a unique new run folder and displays measured results;
it does not promise to reproduce historical performance.

The simulator search is a planning algorithm, not a trained ZIP model. The
third model is a neural DQN trained with demonstrations; its saved policy still
chooses its gameplay actions using its Q-network.

## Training code reused

- Choice 1 reuses `train_model_one()` in `levels/train_dqn.py` unchanged.
- Choice 2 uses the small new `train_standard_all_pegs()` wrapper around the
  existing `train_initial_all_pegs()` learning loop. No separate large
  `dqn_all_pegs.py` was recreated. The wrapper requires a fresh empty output
  folder and does not load historical weights or demonstrations.
- Choice 3 reuses `train_model_two()` unchanged. Its legacy function name remains
  even though it is now menu choice 3. Its prerequisite training, behavior
  cloning, replay mixing, and completion reward procedure are preserved.

`train_initial_all_pegs()` now accepts an optional checkpoint-evaluation hook;
the assisted trainer does not supply it and retains its original default 90k
prerequisite. Independent training uses the original unwrapped reward:
`5 * new_pegs_cleared - 2 + 100 * success`.

The original all-peg settings were checked against Git history: MlpPolicy,
[64,64], learning rate .0003, gamma .99, replay 20k, batch 64, warmup 1000,
train frequency 4, one gradient step, target update every 500, epsilon 1 -> .05
with absolute decay 60k. Fresh standard training defaults to 100k steps.

The original validation procedure was recovered from `a1511cc:levels/dqn_all_pegs.py`:
50 fixed-epsilon .05 games starting at seed 45042, with a separate greedy game.
Selection ranks greedy full clear, validation wins, greedy clearance, mean
clearance, greedy winning shots, then average successful shots. Final test seeds
91042..91141 are separate from selection. The standard evaluator reuses summary
logic while bypassing the demonstration helper and its additional reward.
The assisted evaluator's default output was checked against both original saved
game and shot CSVs and remains identical.

## Tests and preservation

- **38 existing tests passed; 1 deselected.** The unchanged legacy fresh-training
  test assumes choice 2 is demonstration-assisted and expects its stage files.
  It was not run, since that outdated mapping could launch a full standard run.
  The new three-model source-only and menu checks cover its replaced routing.
- Source-only smokes used 256 orange steps, 1104 independent standard steps,
  and 1104 initial + 1104 assisted steps with eight cloning updates, two
  validation games and 552-step intervals. These are engineering smoke results,
  not research performance claims.
- All historical checkpoint loads were blocked during fresh-training tests.
  Demonstration helpers were additionally blocked during standard training.
  Weight updates, optimizer states, checkpoint reloads and own-run replay were
  verified. All temporary smoke model ZIP/replay files and pytest artifacts
  were removed; no submission ZIP or project copy was created.
- Greedy actions from all three existing policies were checked against Q argmax.
- The standard policy's 100 sampled games reproduce original game rows exactly;
  all original orange summary/game/shot/arrangement rows also match exactly.
- Hashes cover **4,943 pre-existing project files** (excluding Git/dependencies).
  Exactly the three authorized source files below changed; there are **zero
  unexpected changes**. Existing tests, notebooks, all DQN/PPO weights, historical
  research results, assets, game physics, configuration, README, dependency files,
  and the preceding standard-DQN optimization study remain unchanged.
- `train_model_one`, `train_model_two`, `clone_actions`, `insert_demonstration`,
  and `EpisodeLog` retain their original source-function hashes.
- Nothing was committed or pushed.

Modified files: `play.py`, `levels/train_dqn.py`, `levels/evaluate.py`.
The only new Python file is `levels/simulator_search.py`.
Verification logs and preservation manifests are in `restoration_checks/`.

## Commands

Use the existing virtual environment from the Final Project folder:

```text
.venv/Scripts/python.exe -B play.py
.venv/Scripts/python.exe -B -m levels.simulator_search --verify
.venv/Scripts/python.exe -B -m levels.simulator_search --max-candidates 63 --beam-width 2 --max-depth 3
```

To run the full adapted search yourself:

```text
.venv/Scripts/python.exe -B -m levels.simulator_search
```

Independent standard training from scratch (new empty output folder):

```text
.venv/Scripts/python.exe -B -m levels.train_dqn standard --steps 100000 --output models/retrained/standard_manual_new
```

Read-only original standard evaluation:

```text
.venv/Scripts/python.exe -B -m levels.evaluate standard --episodes 100
```

The previous `train_dqn all` and `evaluate all` commands retain their
**demonstration-assisted** meaning; `standard` is the independent all-peg route.

Source size: simulator_search.py has 320 lines, compared with 830 lines in the existing optimization experiment.
