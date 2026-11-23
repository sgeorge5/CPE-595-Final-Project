# DQN eight-shot learning experiment

Both new experiments retain the original 78 observations, 21 angles, ten-shot maximum and physics.

Experiment A continued the original 90,000-step model with a fresh replay buffer. Its reward is 5 per new peg, minus 2 per shot, plus 100 + 20 times the actual unused shots on completion. Training exploration was epsilon 0.05. The maximum successful eight-shot return is 509, versus 465 for a ten-shot clear. An incomplete episode earns at most 378 before discounting, so a complete clear is preferred in total undiscounted return.

Experiment B used 1,000 supervised cross-entropy updates on the eight actual demonstrated observations and discrete actions. These updates directly trained the existing Q-network's action preferences, after which its target network was synchronized. A fresh optimizer and fresh replay buffer were used for normal SB3 DQN learning. The buffer initially contained 100 copies of the eight transitions and received another copy every 32 gameplay steps. SB3 sampled all replay entries uniformly. The demo fraction is approximately 20% after the buffer fills. Evaluation always uses network predictions, with explicit epsilon-greedy perturbations for stochastic testing.

After the first 5,000-step demonstration checkpoint lost the eight-shot clear, an additional bounded rehearsal variation was started. It also used 20 supervised updates on the same eight examples every 128 environment steps, alongside ordinary SB3 DQN updates. Its budget was 10,000 additional steps. All variations and their validation results are retained. Rehearsal uses a separate supervised optimizer, whose state is reinitialized if that run resumes.

Only eight demonstrated state-action pairs are available. Reproducing those actions is evidence of demonstration-assisted policy learning on one fixed board. The recovery probes test forced alternative actions after 0, 2 or 4 reference-prefix shots. They are controlled intermediate-state probes, not unseen physical maps.

Validation used 50 games on seeds 47042–47091. Selection prioritized wins, winning shot count, greedy completion, greedy shot count, then mean clearance. Checkpoint choices were frozen before testing on seeds 93042–93141, at epsilon 0.05. Colors do not affect the no-scoring dynamics, observations or reward. Rewards under the two reward functions have different scales.

| Model | Greedy pegs | Greedy shots | Test wins | Wins in <=8 | Wins in 9 | Wins in 10 |
|---|---|---|---|---|---|---|
| original | 77/77 | 10 | 63/100 | 0 | 0 | 63 |
| reward | 77/77 | 10 | 65/100 | 0 | 0 | 65 |
| demonstration | 77/77 | 8 | 78/100 | 69 | 9 | 0 |
| rehearsal | 77/77 | 8 | 78/100 | 69 | 9 | 0 |

Selected checkpoints

- original uses `models/dqn_all_pegs_best.zip`.
- reward uses `models/dqn_eight_shot_experiment\reward\checkpoint_105000.zip`.
  Its full run completed 50,000 additional environment steps. The selected checkpoint is a separate validation decision.
- demonstration uses `models/dqn_eight_shot_experiment\demonstration\checkpoint_110000.zip`.
  Its full run completed 25,000 additional environment steps. The selected checkpoint is a separate validation decision.
- rehearsal uses `models/dqn_eight_shot_experiment\rehearsal\checkpoint_95000.zip`.
  Its full run completed 10,000 additional environment steps. The selected checkpoint is a separate validation decision.

Greedy action sequences

- original fired `0 36 36 -18 6 -60 -18 -60 -18 -24` degrees. Its own reward totaled 465; the original environment reward for that same gameplay is 465.
- reward fired `0 36 36 -18 6 -60 -18 -60 -18 -24` degrees. Its own reward totaled 465; the original environment reward for that same gameplay is 465.
- demonstration fired `-6 12 -30 -60 -54 0 -42 -42` degrees. Its own reward totaled 509; the original environment reward for that same gameplay is 469.
- rehearsal fired `-6 12 -30 -60 -54 0 -42 -42` degrees. Its own reward totaled 509; the original environment reward for that same gameplay is 469.

Recovery probes

- original completed 1/63 forced-prefix probes, including 0 in at most eight shots.
- reward completed 1/63 forced-prefix probes, including 0 in at most eight shots.
- demonstration completed 15/63 forced-prefix probes, including 4 in at most eight shots.
- rehearsal completed 15/63 forced-prefix probes, including 4 in at most eight shots.

Interpretation

A saved DQN Q-network reproduced an eight-shot full clear after reinforcement learning. Its training included the successful simulator demonstration. The result supports a demonstration-assisted learning claim on the fixed board. It does not show that reinforcement learning independently discovered this sequence or establish an optimal shot count.
Reward-only learning cleared 77/77 in 10 greedy shots. The terminal bonus alone did not establish the target if its greedy shot count remains above eight. Reward magnitudes should not be compared as scores of policy quality.
The demonstration-assisted policy improved completion frequency and shot efficiency on this matched test. Its mean peg count can still be lower because unsuccessful exploratory games sometimes lose many pegs. The rehearsal variation did not improve these completion counts over the selected demonstration-plus-replay model. One training seed and a single physical layout limit conclusions about generalization.

Independent reproduction commands

- `python -m levels.dqn_eight_shot_experiment train --experiment reward --steps 50000`
- `python -m levels.dqn_eight_shot_experiment train --experiment demonstration --steps 25000`
- `python -m levels.dqn_eight_shot_experiment train --experiment rehearsal --steps 10000`
- `python -m levels.dqn_eight_shot_experiment report`

The finished experiment's selection is frozen. For a new independent training replication, choose fresh ROOT and MODELS paths in this experiment module before starting. Original project settings and checkpoint paths need no edits. A fresh-process load of the selected policy reproduced its greedy clear in the original unwrapped environment, with each action verified against the Q-network argmax.

The baseline is the untouched Model 2. Historical 73/100 and 60/100 results use different seeds and remain in their original files. The new comparison is stored separately.

The current cleaned-code regression baseline is 60 passing tests with five training tests excluded. The older 66-test shot-search suite covered orchestration later removed in the code cleanup. New experiment tests verify the isolated terminal bonus and true demonstration transitions.

Run `python -m levels.dqn_eight_shot_experiment play` to load the validation-selected new policy. Add `--no-browser` for file-only playback. Training commands resume their latest checkpoint and replay buffer. Exact environment and random-number progression are not restored. New commands run independently of the notebook and the original menu.

## Final verification

The current regression suite passed 63 tests, with five historical training tests excluded. The baseline before this additive experiment was 60 passing tests; three new reward/transition/ranking tests account for the increase. All 3,655 pre-existing files matched their recorded SHA-256 checksums, including the entire notebook, model checkpoints, original Python implementations, physics, prior reports and results. The preservation_checks.csv file records these checks. The original orange-DQN evaluation matched every historical row, including 92.60% clearance and 12/100 wins. The original all-peg DQN reproduced its exact ten-shot winning angles, and the original PPO menu and checkpoints still loaded. Git status shows only the new experiment module, its tests, and its isolated results directory.
