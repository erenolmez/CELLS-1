from __future__ import annotations

import json, math, os, random, time
from functools import partial               
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from updated_grid_env import CellularNetworkEnv

# ╭─────────────────────────────────────────────────────────────┐
# │ 0.  Module‑level evaluator  (pickle‑able)                   │
# ╰─────────────────────────────────────────────────────────────┘
# This is what every worker process runs.  It *must* be top‑level so the
#   multiprocessing spawn‑strategy on Windows can import it.

def _evaluate_individual(
    individual: np.ndarray,
    *,
    rows: int,
    cols: int,
    total_users: int,
    movement_mode: str,
    steps: int,
) -> float:
    """Return average reward over <steps> for a candidate grid."""
    env = CellularNetworkEnv(
        rows=rows,
        cols=cols,
        total_users=total_users,
        movement_mode=movement_mode,
    )
    env.antenna_grid = individual.copy()
    env.place_users()

    total_r = 0.0
    for _ in range(steps):
        env.move_users_markov_chain()
        _, fails, reds = env.check_coverage()
        total_r += env.calculate_reward(fails, reds)
    return total_r / steps

# ╭─────────────────────────────────────────────────────────────┐
# │ 1. Helper — create a zone‑seeded individual                 │
# ╰─────────────────────────────────────────────────────────────┘

def _create_zone_seeded_individual(
    rows: int,
    cols: int,
    zones: List[List[Tuple[int, int]]],
    antenna_range: Tuple[int, int],
) -> np.ndarray:
    """Seed grid with 3‑4 antennas on zone cores + 1‑2 on neighbour ring."""
    grid = np.zeros((rows, cols), dtype=int)

    # core cells
    for zone in zones:
        for r, c in zone:
            grid[r, c] = np.random.randint(3, 5)

    # neighbour ring
    neighbours = set()
    for zone in zones:
        for r, c in zone:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbours.add((nr, nc))
    zone_cells = {cell for zone in zones for cell in zone}
    neighbours.difference_update(zone_cells)
    for r, c in neighbours:
        grid[r, c] = np.random.randint(1, 3)
    return np.clip(grid, *antenna_range)

# ╭─────────────────────────────────────────────────────────────┐
# │ 2. GA main entry                                            │
# ╰─────────────────────────────────────────────────────────────┘

def run_ga(
    *,
    # grid / env ----------------------------------------------------------------
    rows: int = 20,
    cols: int = 20,
    total_users: int = 50_000,
    antenna_capacity: int = 300,
    movement_mode: str = "attraction",  # or "markov"
    # GA hyper‑params ------------------------------------------------------------
    pop_size: Optional[int] = None,  # default = logical cores
    generations: int = 250,
    base_mutation_rate: float = 0.10,
    density_boost_start_gen: int = 50,
    block_size: Optional[int] = None,  # GA block crossover
    logging: bool = True,
) -> None:
    """Parallel GA with sane defaults — backward‑compatible signature."""

    # ─── derived constants ────────────────────────────────────
    pop_size = pop_size or cpu_count()
    block_size = block_size or max(2, min(6, rows // 4))

    antenna_range = (0, 4) if movement_mode == "attraction" else (0, 3)
    coverage_baseline = max(rows * cols, math.ceil(total_users / antenna_capacity))
    max_antennas_allowed = int(coverage_baseline * 0.625)

    np.random.seed()  # each process gets its own RNG state
    random.seed()

    # ─── helper fns INSIDE main proc (not sent to workers) ───
    def _mutate(individual: np.ndarray, mut_rate: float) -> np.ndarray:
        mutant = individual.copy()
        for r in range(rows):
            for c in range(cols):
                if random.random() < mut_rate:
                    delta = random.choice([-1, 1])
                    mutant[r, c] = np.clip(mutant[r, c] + delta, *antenna_range)
        return mutant

    def _crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        child = p1.copy()
        for r in range(0, rows, block_size):
            for c in range(0, cols, block_size):
                if random.random() < 0.5:
                    child[r : r + block_size, c : c + block_size] = p2[
                        r : r + block_size,
                        c : c + block_size,
                    ]
        return child

    # local evaluator (can return density) — stays inside main process
    def _evaluate_local(individual: np.ndarray, steps: int, return_density=False):
        env = CellularNetworkEnv(
            rows=rows,
            cols=cols,
            total_users=total_users,
            movement_mode=movement_mode,
        )
        env.antenna_grid = individual.copy()
        env.place_users()

        total_r = 0.0
        density_acc = np.zeros((rows, cols), dtype=float) if return_density else None

        for _ in range(steps):
            env.move_users_markov_chain()
            if return_density:
                density_acc += env.user_grid
            _, fails, reds = env.check_coverage()
            total_r += env.calculate_reward(fails, reds)
        avg_r = total_r / steps
        if return_density:
            return avg_r, density_acc / steps  # type: ignore
        return avg_r

    # --- pool‑safe evaluators -----------------------------------
    # — use functools.partial instead of lambdas
    eval_2190 = partial(
        _evaluate_individual,
        rows=rows,
        cols=cols,
        total_users=total_users,
        movement_mode=movement_mode,
        steps=2_190,
    )
    eval_8760 = partial(
        _evaluate_individual,
        rows=rows,
        cols=cols,
        total_users=total_users,
        movement_mode=movement_mode,
        steps=8_760,
    )

    # ─── population init ────────────────────────────────────────
    population: List[np.ndarray] = []
    seed_file = Path("best_antenna_grid.npy")

    if seed_file.exists():
        seed_grid = np.load(seed_file)
        if movement_mode == "attraction":
            dummy_env = CellularNetworkEnv(
                rows=rows,
                cols=cols,
                total_users=total_users,
                movement_mode=movement_mode,
            )
            zones = [dummy_env.zones[z] for z in ("home", "school", "work", "city_center")]
            overlay = _create_zone_seeded_individual(rows, cols, zones, antenna_range)
            augmented = np.clip(seed_grid + overlay, *antenna_range)
            # trim excess antennas
            excess = max(0, augmented.sum() - max_antennas_allowed)
            if excess:
                flat = augmented.flatten()
                for idx in np.argsort(flat)[::-1]:
                    if excess == 0:
                        break
                    if flat[idx] > 0:
                        flat[idx] -= 1
                        excess -= 1
                augmented = flat.reshape((rows, cols))
            population.append(augmented)
            while len(population) < pop_size:
                population.append(_mutate(augmented, base_mutation_rate))
        else:
            population.append(seed_grid)
            while len(population) < pop_size:
                population.append(_mutate(seed_grid, base_mutation_rate))
    else:
        if movement_mode == "attraction":
            dummy_env = CellularNetworkEnv(
                rows=rows,
                cols=cols,
                total_users=total_users,
                movement_mode=movement_mode,
            )
            zones = [dummy_env.zones[z] for z in ("home", "school", "work", "city_center")]
            population.append(_create_zone_seeded_individual(rows, cols, zones, antenna_range))
        while len(population) < pop_size:
            population.append(
                np.random.randint(antenna_range[0], antenna_range[1] + 1, size=(rows, cols))
            )

    # ─── GA main loop ───────────────────────────────────────────
    gen_log: List[dict] = []
    last_best_fitness: Optional[float] = None
    last_improvement_gen = 0

    with Pool(pop_size) as pool:
        fitnesses = list(tqdm(pool.imap(eval_2190, population), total=pop_size, desc="Initial Eval"))

        for gen in range(generations):
            print(f"\n📈 Generation {gen + 1}/{generations}")
            mut_rate = base_mutation_rate * (1 - gen / generations)

            best_idx = int(np.argmax(fitnesses))

            # density‑driven reinforcement (late stage, attraction mode)
            if movement_mode == "attraction" and gen >= density_boost_start_gen:
                best_fit, density = _evaluate_local(population[best_idx], 8_760, return_density=True)
                threshold = np.percentile(density, 95)
                hotspot = np.argwhere(density >= threshold)
                modified = population[best_idx].copy()
                for r, c in hotspot:
                    modified[r, c] = min(modified[r, c] + 1, antenna_range[1])
                modified_fit = _evaluate_local(modified, 8_760)
                if modified_fit > fitnesses[best_idx]:
                    population[best_idx] = modified
                    fitnesses[best_idx] = modified_fit
                    print("  ✅ Reinforced grid accepted.")
                else:
                    print("  ❌ Reinforced grid rejected.")
            best_fit = float(fitnesses[best_idx])

            # stagnation detection → temp mutation boost
            if last_best_fitness is None or best_fit > last_best_fitness:
                last_best_fitness = best_fit
                last_improvement_gen = gen
            elif gen - last_improvement_gen >= 10:
                print("⚠️  Stagnation – temporary mutation boost.")
                mut_rate *= 1.5

            # selection & reproduction
            ranked = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
            parents = ranked[: pop_size // 2]
            new_pop = parents.copy()
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(parents, 2)
                child = _mutate(_crossover(p1, p2), mut_rate)
                new_pop.append(child)
            population = new_pop

            evaluator = eval_8760 if gen >= (generations - 50) else eval_2190
            fitnesses = list(tqdm(pool.imap(evaluator, population), total=pop_size, desc=f"Gen {gen + 1}"))
            best_idx = int(np.argmax(fitnesses))
            best_ind = population[best_idx]
            print(
                f"  🏅 Best reward: {fitnesses[best_idx]:.4f} | 📡 Antennas: {best_ind.sum()}"
            )
            if logging:
                gen_log.append(
                    {
                        "generation": gen + 1,
                        "reward": float(fitnesses[best_idx]),
                        "antennas": int(best_ind.sum()),
                    }
                )

    # save artefacts --------------------------------------------------
    np.save("best_antenna_grid.npy", best_ind)
    if logging:
        with open("ga_progress_log.json", "w") as f:
            json.dump(gen_log, f, indent=2)
    print("\n✅ GA complete – best grid saved to best_antenna_grid.npy")

# ╭─────────────────────────────────────────────────────────────┐
# │ 3. CLI fallback                                             │
# ╰─────────────────────────────────────────────────────────────┘

if __name__ == "__main__":
    run_ga()
