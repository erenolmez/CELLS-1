import numpy as np
import random
import os
import time
from multiprocessing import Pool, cpu_count
from updated_grid_env import CellularNetworkEnv
from tqdm import tqdm

# =========================
# Parallel Genetic Algorithm for CELLS
# =========================
# Grid size: 20x20
# Users simulated: 50000
# Duration per individual: 3 months (2190 steps) in early gens, 1 year (8760 steps) in final 50
# Antenna range per cell: [0, 3] ⟶  will be dynamically set
# Parallel CPU workers = POP_SIZE = number of logical cores
# =========================

ROWS = 20
COLS = 20
TOTAL_USERS = 50000
NUM_CORES = cpu_count()
POP_SIZE = NUM_CORES
GENERATIONS = 250
# MUTATION_RATE = 0.1
BASE_MUTATION_RATE = 0.1
BLOCK_SIZE = max(2, min(6, ROWS // 4))
MOVEMENT_MODE = "attraction"
DENSITY_BOOST_START_GEN = 50

if MOVEMENT_MODE == "attraction":
    ANTENNA_RANGE = (0, 4)  # allow more antennas in dynamic hotspots
else:
    ANTENNA_RANGE = (0, 3)

def create_zone_seeded_individual(zones):
    """
    Assign more antennas to zone cells and slightly fewer to their neighbors.
    """
    grid = np.zeros((ROWS, COLS), dtype=int)

    # --- Core placement (zones) ---
    for zone in zones:
        for r, c in zone:
            grid[r, c] = np.random.randint(3, 5)  # 3–4 antennas for hotspot core

    # --- Neighbor placement ---
    neighbors = set()
    for zone in zones:
        for r, c in zone:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        neighbors.add((nr, nc))

    # Exclude core zone cells
    zone_cells = {cell for zone in zones for cell in zone}
    neighbors.difference_update(zone_cells)

    for r, c in neighbors:
        grid[r, c] = np.random.randint(1, 3)  # 1–2 antennas in neighbor ring

    return np.clip(grid, *ANTENNA_RANGE)

def create_individual():
    return np.random.randint(ANTENNA_RANGE[0], ANTENNA_RANGE[1] + 1, size=(ROWS, COLS))

# def mutate(individual):
#     mutant = individual.copy()
#     for r in range(ROWS):
#         for c in range(COLS):
#             if random.random() < MUTATION_RATE:
#                 delta = random.choice([-1, 1])
#                 mutant[r, c] = np.clip(mutant[r, c] + delta, *ANTENNA_RANGE)
#     return mutant

def mutate(individual, mutation_rate):
    mutant = individual.copy()
    for r in range(ROWS):
        for c in range(COLS):
            if random.random() < mutation_rate:
                delta = random.choice([-1, 1])
                mutant[r, c] = np.clip(mutant[r, c] + delta, *ANTENNA_RANGE)
    return mutant

# def crossover(parent1, parent2):
#     mask = np.random.rand(ROWS, COLS) > 0.5
#     child = np.where(mask, parent1, parent2)
#     return child

def crossover(parent1, parent2):
    child = parent1.copy()
    for r in range(0, ROWS, BLOCK_SIZE):
        for c in range(0, COLS, BLOCK_SIZE):
            if random.random() < 0.5:
                child[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = parent2[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
    return child

# def evaluate(individual, steps):
def evaluate(individual, steps, return_density=False):
    env = CellularNetworkEnv(rows=ROWS, cols=COLS, total_users=TOTAL_USERS, movement_mode=MOVEMENT_MODE)
    env.antenna_grid = individual.copy()
    env.place_users()

    total_reward = 0.0

    if return_density:
        density_accum = np.zeros((ROWS, COLS), dtype=float)
    
    for _ in range(steps):
        env.move_users_markov_chain()
        if return_density:
            density_accum += env.user_grid
        _, failures, redirects = env.check_coverage()
        reward = env.calculate_reward(failures, redirects)
        total_reward += reward

    # return total_reward / steps

    avg_reward = total_reward / steps
    if return_density:
        return avg_reward, density_accum / steps
    else:
        return avg_reward

def evaluate_2190(ind):
    return evaluate(ind, 2190)

def evaluate_8760(ind):
    return evaluate(ind, 8760)

def run_ga():
    start_time = time.time()
    print(f"Using {NUM_CORES} CPU cores for parallel GA.")
    print(f"🚦 Movement mode: {MOVEMENT_MODE.upper()} — initializing population accordingly.")
    population = []

    # === Seed from previous best if exists ===
    if os.path.exists("best_antenna_grid.npy"):
        seed_grid = np.load("best_antenna_grid.npy")

        if MOVEMENT_MODE == "attraction":
            print("🔁 Augmenting saved best grid with zone-based reinforcement.")
            dummy_env = CellularNetworkEnv(rows=ROWS, cols=COLS, total_users=TOTAL_USERS, movement_mode=MOVEMENT_MODE)
            zones = [dummy_env.zones["home"], dummy_env.zones["school"], dummy_env.zones["work"], dummy_env.zones["city_center"]]

            # Overlay reinforcement
            overlay = create_zone_seeded_individual(zones)
            augmented = np.clip(seed_grid + overlay, *ANTENNA_RANGE)
            MAX_ANTENNAS_ALLOWED = 250
            if augmented.sum() > MAX_ANTENNAS_ALLOWED:
                excess = augmented.sum() - MAX_ANTENNAS_ALLOWED
                flat = augmented.flatten()
                sorted_idx = np.argsort(flat)[::-1]  # highest antennas first
                for idx in sorted_idx:
                    if excess == 0:
                        break
                    if flat[idx] > 0:
                        flat[idx] -= 1
                        excess -= 1
                augmented = flat.reshape((ROWS, COLS))
            print(f"⚖️ Clamped seed+overlay from {seed_grid.sum() + overlay.sum()} to {augmented.sum()} antennas.")
            population.append(augmented)

            while len(population) < POP_SIZE:
                population.append(mutate(augmented, BASE_MUTATION_RATE))

        else:
            population.append(seed_grid)
            while len(population) < POP_SIZE:
                population.append(mutate(seed_grid, BASE_MUTATION_RATE))

    else:
        if MOVEMENT_MODE == "attraction":
            print("⚙️ Using zone-seeded initialization for attraction movement.")
            # Generate first zone-seeded individual
            dummy_env = CellularNetworkEnv(rows=ROWS, cols=COLS, total_users=TOTAL_USERS, movement_mode=MOVEMENT_MODE)
            zones = [dummy_env.zones["home"], dummy_env.zones["school"], dummy_env.zones["work"], dummy_env.zones["city_center"]]
            population.append(create_zone_seeded_individual(zones))
            while len(population) < POP_SIZE:
                population.append(create_individual())
        else:
            population = [create_individual() for _ in range(POP_SIZE)]


    last_best_fitness = None
    last_improvement_gen = 0

    with Pool(NUM_CORES) as pool:
        fitnesses = list(tqdm(pool.imap(evaluate_2190, population), total=POP_SIZE, desc="Initial Eval"))

        for gen in range(GENERATIONS):
            print(f"\n📈 Generation {gen+1}/{GENERATIONS}")

            # Annealing-style adaptive mutation rate
            mutation_rate = BASE_MUTATION_RATE * (1 - gen / GENERATIONS)

            # === Stagnation Detection ===
            best_idx = np.argmax(fitnesses)
            # best_fit = fitnesses[best_idx]

            # 🔁 Reinforce antennas at high-traffic cells in late-stage attraction mode
            #     → tracks density from top candidate and boosts it
            if MOVEMENT_MODE == "attraction" and gen >= DENSITY_BOOST_START_GEN:
                best_fit, density = evaluate(population[best_idx], 8760, return_density=True)

                # Identify top 5% hottest cells
                threshold = np.percentile(density, 95)
                hotspot_coords = np.argwhere(density >= threshold)

                # ✅ Work on a copy, not in-place
                modified = population[best_idx].copy()
                # 🌟 Final global clip (in case any values were pushed beyond range)
                for r, c in hotspot_coords:
                    modified[r, c] = min(modified[r, c] + 1, ANTENNA_RANGE[1])

                modified_fit = evaluate(modified, 8760)

                # ✅ Accept if improved
                if modified_fit > fitnesses[best_idx]:
                    population[best_idx] = modified
                    fitnesses[best_idx] = modified_fit
                    print("  ✅ Reinforced antenna grid accepted.")
                else:
                     best_fit = fitnesses[best_idx]
                     print("  ❌ Reinforced grid did not improve fitness.")
            else:
                best_fit = fitnesses[best_idx]
                
            if last_best_fitness is None or best_fit > last_best_fitness:
                last_best_fitness = best_fit
                last_improvement_gen = gen
            elif gen - last_improvement_gen >= 10:
                print("⚠️ Stagnation detected — boosting mutation temporarily.")
                mutation_rate *= 1.5

            ranked = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
            num_survivors = POP_SIZE // 2
            parents = ranked[:num_survivors]
            new_pop = parents.copy()

            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(parents, 2)
                child = crossover(p1, p2)
                # child = mutate(child)
                child = mutate(child, mutation_rate)
                new_pop.append(child)

            population = new_pop
            evaluator = evaluate_8760 if gen >= 200 else evaluate_2190
            fitnesses = list(tqdm(pool.imap(evaluator, population), total=POP_SIZE, desc=f"Generation {gen+1}"))
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            best_ind = population[best_idx]
            num_antennas = best_ind.sum()
            print(f"  🏅 Best reward this gen: {best_fit:.4f} | 📡 Antennas used: {num_antennas}")

        print("\n✅ Best Antenna Grid:")
        print(best_ind)
        np.save("best_antenna_grid.npy", best_ind)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"\n⏱️ Total runtime: {int(mins)} min {int(secs)} sec")

if __name__ == "__main__":
    run_ga()
