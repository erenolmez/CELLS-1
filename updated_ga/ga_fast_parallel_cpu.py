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
# Antenna range per cell: [0, 3]
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
ANTENNA_RANGE = (0, 3)
BLOCK_SIZE = max(2, min(6, ROWS // 4))

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

def evaluate(individual, steps):
    env = CellularNetworkEnv(rows=ROWS, cols=COLS, total_users=TOTAL_USERS)
    env.antenna_grid = individual.copy()
    env.place_users()

    total_reward = 0.0
    for _ in range(steps):
        env.move_users_markov_chain()
        _, failures, redirects = env.check_coverage()
        reward = env.calculate_reward(failures, redirects)
        total_reward += reward

    return total_reward / steps

def evaluate_2190(ind):
    return evaluate(ind, 2190)

def evaluate_8760(ind):
    return evaluate(ind, 8760)

def run_ga():
    start_time = time.time()
    print(f"Using {NUM_CORES} CPU cores for parallel GA.")
    population = []

    # === Seed from previous best if exists ===
    if os.path.exists("best_antenna_grid.npy"):
        seed_grid = np.load("best_antenna_grid.npy")
        population.append(seed_grid)
        while len(population) < POP_SIZE:
            # population.append(mutate(seed_grid))
            population.append(mutate(seed_grid, BASE_MUTATION_RATE))
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
