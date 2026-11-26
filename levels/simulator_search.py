
import argparse
import csv
import os
import random
import tempfile

from levels.config_rl import RLConfig
from levels.peggle_env import PeggleEnv

ANGLES = list(range(-60, 61, 6))
MAX_SHOTS = 10
TOTAL_PEGS = 77
BEAM_WIDTH = 60
POOL_SIZE = 32
MAX_CANDIDATES = 20_000
BOARD_SEED = 50_042
SEARCH_SEED = 592
RESULTS = "results/simulator_search_reconstructed"
HISTORICAL_SOLUTIONS = "results/dqn_shot_efficiency/search_solutions.csv"


def create_environment():
    config = RLConfig(goal="all", max_shots=10, free_balls=False, n_actions=21)
    env = PeggleEnv(config)
    assert len(config.coordinates) == TOTAL_PEGS
    assert env.observation_space.shape == (78,) and env.action_space.n == 21
    assert list(config.angles_degrees) == ANGLES and config.max_shots == MAX_SHOTS
    assert not config.free_balls and not config.endgame_bonus and not config.early_peg_bonus
    assert (config.dt, config.gravity, config.launch_force, config.max_ball_velocity) == (1, .045, 5, 10)
    assert (config.peg_reward, config.shot_penalty, config.completion_bonus) == (5, 2, 100)
    return env


def angle_text(sequence):
    return " ".join(str(angle) for angle in sequence)


def evaluate_sequence(env, sequence):
    if len(sequence) > MAX_SHOTS or any(angle not in ANGLES for angle in sequence):
        raise ValueError("Use at most ten angles from -60 through 60 in steps of six.")
    _, info = env.reset(seed=BOARD_SEED)
    assert info["shots_used"] == 0 and info["pegs_cleared"] == 0
    assert info["orange_initial"] == 25 and env.shots_remaining == MAX_SHOTS
    trace = []
    for angle in sequence:
        before = [peg.isHit for peg in env.pegs]
        action = ANGLES.index(angle)
        _, reward, done, truncated, info = env.step(action)
        hit_ids = [i for i, peg in enumerate(env.pegs) if peg.isHit and not before[i]]
        assert len(hit_ids) == info["new_pegs_cleared"]
        assert "shot_score" not in info and "free_balls_earned" not in info
        trace.append(dict(shot=info["shots_used"], angle_degrees=angle, action=action,
                          new_pegs_cleared=info["new_pegs_cleared"], pegs_cleared=info["pegs_cleared"],
                          pegs_remaining=info["pegs_remaining"], shots_remaining=info["shots_remaining"],
                          reward=reward, hit_ids=angle_text(hit_ids), success=info["success"]))
        if done or truncated:
            break
    sequence = [shot["angle_degrees"] for shot in trace]
    return dict(sequence=sequence, pegs_cleared=info["pegs_cleared"], shots_used=len(trace),
                success=info["success"], trace=trace,
                hit_signature=angle_text(i for i, peg in enumerate(env.pegs) if peg.isHit))


def verify_sequence(env, sequence):
    first = evaluate_sequence(env, sequence)
    second = evaluate_sequence(env, sequence)
    assert first == second
    return second


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidate(env, sequence, method, rows, seen, solutions, budget, folder):
    key = tuple(sequence)
    if key in seen or len(rows) >= budget:
        return None
    seen.add(key)
    result = evaluate_sequence(env, sequence)
    row = dict(candidate=len(rows)+1, method=method, proposed_shots=len(sequence),
               proposed_angles=angle_text(sequence), played_angles=angle_text(result["sequence"]),
               pegs_cleared=result["pegs_cleared"], shots_used=result["shots_used"],
               success=result["success"], replay_verified=False,
               new_pegs_by_shot=angle_text(s["new_pegs_cleared"] for s in result["trace"]),
               pegs_after=angle_text(s["pegs_cleared"] for s in result["trace"]),
               hit_ids_by_shot=";".join(s["hit_ids"] for s in result["trace"]),
               hit_signature=result["hit_signature"])
    if result["success"]:
        shortest_before = min((r["shots_used"] for r in solutions), default=MAX_SHOTS+1)
        checked = verify_sequence(env, result["sequence"])
        assert checked == result and checked["pegs_cleared"] == TOTAL_PEGS
        row["replay_verified"] = True
        solutions.append(row.copy())
        location = os.path.join(folder, "solutions", "candidate_" + str(row["candidate"]))
        write_csv([row], os.path.join(location, "game.csv"))
        write_csv(result["trace"], os.path.join(location, "shots.csv"))
        write_csv(solutions, os.path.join(folder, "search_solutions.csv"))
        if row["shots_used"] < shortest_before:
            print(f"New verified full clear: {row['shots_used']} shots, "
                  f"candidate {row['candidate']}: {row['played_angles']}", flush=True)
    rows.append(row)
    if len(rows) % 100 == 0 or result["success"]:
        save_results(rows, solutions, budget, folder)
        print(f"Evaluated {len(rows):,}/{budget:,} candidates; "
              f"best clearance {max(r['pegs_cleared'] for r in rows)}/77.", flush=True)
    return dict(sequence=result["sequence"], pegs_cleared=result["pegs_cleared"],
                success=result["success"], candidate=row["candidate"], hit_signature=result["hit_signature"])


def add_to_pool(pools, result):
    if result is None:
        return
    pool = pools[len(result["sequence"])]
    pool.append(result)
    pool.sort(key=lambda row: (-row["pegs_cleared"], row["candidate"]))
    del pool[POOL_SIZE:]


def sequence_edits(sequence):
    edits = []
    for i in range(len(sequence)):
        edits.append((sequence[:i] + sequence[i+1:], "Remove shot"))
        for angle in ANGLES:
            if angle != sequence[i]:
                changed = sequence.copy()
                changed[i] = angle
                edits.append((changed, "Replace angle"))
        for j in range(i+1, len(sequence)):
            changed = sequence.copy()
            changed[i], changed[j] = changed[j], changed[i]
            edits.append((changed, "Swap shots"))
    return edits


def random_candidate(pools, maximum_length, rng):
    length = maximum_length if rng.random() < .7 else rng.randint(1, maximum_length)
    sources = pools[length] or pools.get(length+1, [])
    if not sources or rng.random() < .05:
        return [rng.choice(ANGLES) for _ in range(length)], "Fresh random sequence"
    sequence = rng.choice(sources[:12])["sequence"].copy()
    while len(sequence) > length:
        del sequence[rng.randrange(len(sequence))]
    while len(sequence) < length:
        sequence.insert(rng.randrange(len(sequence)+1), rng.choice(ANGLES))
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(length)
        if rng.random() < .32:
            j = rng.randrange(length)
            sequence[i], sequence[j] = sequence[j], sequence[i]
        else:
            sequence[i] = rng.choice(ANGLES)
    return sequence, "Random mutation"


def save_results(rows, solutions, budget, folder):
    if not rows:
        return
    write_csv(rows, os.path.join(folder, "search_candidates.csv"))
    by_length = []
    for length in range(1, MAX_SHOTS+1):
        group = [r for r in rows if r["proposed_shots"] == length]
        if group:
            by_length.append(dict(proposed_shots=length, candidates=len(group),
                                  best_clearance=max(r["pegs_cleared"] for r in group),
                                  full_clears=sum(r["success"] for r in group)))
    write_csv(by_length, os.path.join(folder, "search_by_length.csv"))
    if solutions:
        best = min(solutions, key=lambda row: (row["shots_used"], row["candidate"]))
    else:
        best = max(rows, key=lambda row: (row["pegs_cleared"], -row["shots_used"]))
    write_csv([best], os.path.join(folder, "best_sequence.csv"))
    write_csv([dict(candidates_tested=len(rows), candidate_budget=budget,
                    best_clearance=best["pegs_cleared"], full_clear=best["success"],
                    shortest_verified=best["shots_used"] if solutions else "",
                    verification_replays=2*len(solutions))], os.path.join(folder, "search_progress.csv"))


def search(env, beam_width=BEAM_WIDTH, max_candidates=MAX_CANDIDATES, max_depth=MAX_SHOTS):
    if beam_width < 1 or not 1 <= max_candidates <= MAX_CANDIDATES or not 1 <= max_depth <= MAX_SHOTS:
        raise ValueError("Use positive beam width, 1..20000 candidates, and depth 1..10.")
    os.makedirs(RESULTS, exist_ok=True)
    folder = tempfile.mkdtemp(prefix="run_", dir=RESULTS)
    write_csv([dict(implementation="Restored/adapted; not an exact historical rerun",
                    source_blob="91f71148aa52d72dee77038917216ea0d834766d", beam_width=beam_width,
                    pool_size=POOL_SIZE, candidate_budget=max_candidates, max_depth=max_depth,
                    board_seed=BOARD_SEED, search_seed=SEARCH_SEED,
                    ranking="clearance descending, candidate ascending; distinct hit states",
                    initial_candidates="empty sequence; no DQN or historical solutions")],
              os.path.join(folder, "search_settings.csv"))
    rows, solutions, seen = [], [], set()
    pools = {length: [] for length in range(1, MAX_SHOTS+1)}
    rng = random.Random(SEARCH_SEED)
    try:
        candidates = [[]]
        for depth in range(1, max_depth+1):
            if solutions and depth >= min(r["shots_used"] for r in solutions):
                break  
            expanded = []
            for sequence in candidates:
                for angle in ANGLES:
                    result = evaluate_candidate(env, sequence+[angle], "Forward beam", rows, seen,
                                                solutions, max_candidates, folder)
                    if result is not None:
                        add_to_pool(pools, result)
                        if not result["success"]:
                            expanded.append(result)
                    if len(rows) >= max_candidates:
                        break
                if len(rows) >= max_candidates:
                    break
            if not expanded:
                break
            expanded.sort(key=lambda row: (-row["pegs_cleared"], row["candidate"]))
            candidates, states = [], set()
            for result in expanded:
                if result["hit_signature"] not in states:
                    states.add(result["hit_signature"])
                    candidates.append(result["sequence"])
                    if len(candidates) == beam_width:
                        break
            print(f"Beam depth {depth}: best {expanded[0]['pegs_cleared']}/77, "
                  f"{len(rows):,} candidates evaluated.", flush=True)
            save_results(rows, solutions, max_candidates, folder)
            if len(rows) >= max_candidates:
                break

        edits = []
        for length in range(1, max_depth+1):
            for result in pools[length][:2]:
                edits.extend(sequence_edits(result["sequence"]))
        for sequence, method in edits:
            maximum = min(max_depth, min((r["shots_used"]-1 for r in solutions), default=max_depth))
            if 1 <= len(sequence) <= maximum:
                result = evaluate_candidate(env, sequence, method, rows, seen, solutions, max_candidates, folder)
                add_to_pool(pools, result)
            if len(rows) >= max_candidates:
                break
        duplicates = 0
        while len(rows) < max_candidates:
            maximum = min(max_depth, min((r["shots_used"]-1 for r in solutions), default=max_depth))
            if maximum < 1:
                break
            sequence, method = random_candidate(pools, maximum, rng)
            result = evaluate_candidate(env, sequence, method, rows, seen, solutions, max_candidates, folder)
            if result is None:
                duplicates += 1
                if duplicates == 500:
                    break  
            else:
                duplicates = 0
                add_to_pool(pools, result)
    finally:
        save_results(rows, solutions, max_candidates, folder)
    best = (min(solutions, key=lambda r: (r["shots_used"], r["candidate"])) if solutions
            else max(rows, key=lambda r: (r["pegs_cleared"], -r["shots_used"])))
    print(f"Actual new-search result: {best['pegs_cleared']}/77, {best['shots_used']} shots, "
          f"success={best['success']}, angles {best['played_angles']}. Results: {folder}")
    print("Beam pruning and finite mutations do not prove a globally optimal solution.")
    return dict(best=best, candidates_tested=len(rows), output_dir=folder)


def verify_historical(env):
    with open(HISTORICAL_SOLUTIONS, newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    checked = []
    for row in saved:
        sequence = [int(angle) for angle in row["angle_sequence"].split()]
        result = verify_sequence(env, sequence)
        assert result["success"] and result["pegs_cleared"] == int(row["pegs_cleared"]) == TOTAL_PEGS
        assert result["shots_used"] == int(row["shots_used"]) == 8
        checked.append(result)
        print(f"Historical candidate {row['candidate']}: verified 77/77 in eight shots, "
              f"angles {angle_text(sequence)}.")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="replay historical winners only; no search")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--max-depth", type=int, default=MAX_SHOTS)
    args = parser.parse_args()
    env = create_environment()
    try:
        if args.verify:
            verify_historical(env)
        else:
            search(env, args.beam_width, args.max_candidates, args.max_depth)
    finally:
        env.close()


if __name__ == "__main__":
    main()
