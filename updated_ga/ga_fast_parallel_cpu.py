import numpy as np
import random
import os
from multiprocessing import Pool, cpu_count
from env_fast import CellularNetworkEnv
from tqdm import tqdm

# =========================
# Parallel Genetic Algorithm for CELLS
# =========================
# Grid size: 20x20
# Users simulated: 50000
# Duration per individual: 1 month (720 steps)
# Antenna range per cell: [0, 2]
# Parallel CPU workers = POP_SIZE = number of logical cores
# =========================

ROWS = 20
COLS = 20
TOTAL_USERS = 50000
NUM_CORES = cpu_count()
POP_SIZE = NUM_CORES
GENERATIONS = 250
MUTATION_RATE = 0.1
STEPS_PER_EVAL = 2190
ANTENNA_RANGE = (0, 3)

def create_individual():
    return np.random.randint(ANTENNA_RANGE[0], ANTENNA_RANGE[1] + 1, size=(ROWS, COLS))

def mutate(individual):
    mutant = individual.copy()
    for r in range(ROWS):
        for c in range(COLS):
            if random.random() < MUTATION_RATE:
                delta = random.choice([-1, 1])
                mutant[r, c] = np.clip(mutant[r, c] + delta, *ANTENNA_RANGE)
    return mutant

def crossover(parent1, parent2):
    mask = np.random.rand(ROWS, COLS) > 0.5
    child = np.where(mask, parent1, parent2)
    return child

def evaluate(individual):
    env = CellularNetworkEnv(rows=ROWS, cols=COLS, total_users=TOTAL_USERS)
    env.antenna_grid = individual.copy()
    env.place_users()

    total_failures = 0
    total_redirects = 0
    total_reward = 0.0

    for step in range(STEPS_PER_EVAL):
        env.move_users_markov_chain()
        _, failures, redirects = env.check_coverage()
        reward = env.calculate_reward(failures, redirects)
        total_failures += failures
        total_redirects += redirects
        total_reward += reward

    avg_reward = total_reward / STEPS_PER_EVAL
    return avg_reward

def run_ga():
    print(f"Using {NUM_CORES} CPU cores for parallel GA.")
    population = [create_individual() for _ in range(POP_SIZE)]

    with Pool(NUM_CORES) as pool:
        fitnesses = list(tqdm(pool.imap(evaluate, population), total=POP_SIZE, desc="Initial Eval"))

        for gen in range(GENERATIONS):
            print(f"\n📈 Generation {gen+1}/{GENERATIONS}")

            # === CHANGED: top 50% truncation selection ===
            ranked = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
            num_survivors = POP_SIZE // 2
            parents = ranked[:num_survivors]
            new_pop = parents.copy()

            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(parents, 2)
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)

            population = new_pop
            fitnesses = list(tqdm(pool.imap(evaluate, population), total=POP_SIZE, desc=f"Generation {gen+1}"))
            best_fit = max(fitnesses)
            print(f"  🏅 Best reward this gen: {best_fit:.4f}")

        best_idx = np.argmax(fitnesses)
        best_ind = population[best_idx]
        print("\n✅ Best Antenna Grid:")
        print(best_ind)
        np.save("best_antenna_grid.npy", best_ind)

if __name__ == "__main__":
    run_ga()
