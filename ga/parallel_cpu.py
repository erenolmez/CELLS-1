from __future__ import annotations
import torch as th
import numpy as np
import random, time
from datetime import timedelta
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import os

NUM_WORKERS = os.cpu_count() or 4

device = "cpu"
print(f"Running on: CPU ({NUM_WORKERS} workers)")

# ──── Environment Definition ────
class CellularNetworkEnvTorch:
    rows, cols = 6, 6
    num_cells  = rows * cols

    def __init__(self, total_users: int = 5000, antenna_capacity: int = 300, time_step: int = 60):
        self.total_users     = total_users
        self.antenna_capacity = antenna_capacity
        self.time_step       = time_step
        self.coverage_rad    = 1
        self.max_antennas    = total_users // antenna_capacity
        self._build_neighbor_map()
        self.reset()

    def _build_neighbor_map(self):
        nbrs = []
        for r in range(self.rows):
            for c in range(self.cols):
                lst = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            lst.append(nr * self.cols + nc)
                nbrs.append(lst)
        self.nbr_map = nbrs

    def reset(self):
        self.antenna_grid = th.zeros((self.rows, self.cols), dtype=th.int8, device=device)
        counts = np.random.multinomial(self.total_users, [1 / self.num_cells] * self.num_cells).reshape(self.rows, self.cols)
        self.car_grid = th.tensor(counts, dtype=th.int32, device=device)
        self.sim_time_h = 0

    def _p_move(self) -> float:
        return 0.4 * (self.time_step / 60)

    def move_users(self):
        p_move = self._p_move()
        new_grid = th.zeros_like(self.car_grid)
        for idx in range(self.num_cells):
            r, c = divmod(idx, self.cols)
            users = int(self.car_grid[r, c].item())
            if users == 0:
                continue
            nbrs = self.nbr_map[idx]
            k = len(nbrs)
            probs = [1 - p_move] + [p_move / k] * k if k else [1.0]
            counts = np.random.multinomial(users, probs)
            new_grid[r, c] += counts[0]
            for j, cnt in enumerate(counts[1:]):
                nr, nc = divmod(nbrs[j], self.cols)
                new_grid[nr, nc] += cnt
        self.car_grid = new_grid

    def coverage_stats(self) -> Tuple[int, int]:
        failures = 0
        redirects = 0
        for r in range(self.rows):
            for c in range(self.cols):
                users = int(self.car_grid[r, c].item())
                if users == 0:
                    continue
                remaining = users
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            cap = int(self.antenna_grid[nr, nc]) * self.antenna_capacity
                            if cap and remaining:
                                served = min(cap, remaining)
                                remaining -= served
                                if (nr, nc) != (r, c):
                                    redirects += served
                        if remaining == 0:
                            break
                failures += remaining
        return failures, redirects

    def step(self):
        self.move_users()
        self.sim_time_h += 1

ROWS, COLS = 6, 6
GRID_SIZE  = ROWS * COLS
POP_SIZE   = 60
GENERATIONS = 120
TOUR_K     = 3
CX_PROB    = 0.7
MUT_PROB   = 0.05
EP_STEPS   = 240
W_FAIL, W_RED, W_ANT = 3.0, 0.1, 0.5
MAX_ANTENNAS = CellularNetworkEnvTorch().max_antennas

def bits_to_tensor(bits: List[int]) -> th.Tensor:
    return th.tensor(bits, dtype=th.int8, device=device).reshape((ROWS, COLS))

def random_layout() -> List[int]:
    bits = [0] * GRID_SIZE
    for i in random.sample(range(GRID_SIZE), MAX_ANTENNAS):
        bits[i] = 1
    return bits

def repair(bits: List[int]):
    ones = [i for i, b in enumerate(bits) if b]
    zeros = [i for i, b in enumerate(bits) if not b]
    if len(ones) > MAX_ANTENNAS:
        for i in random.sample(ones, len(ones) - MAX_ANTENNAS):
            bits[i] = 0
    elif len(ones) < MAX_ANTENNAS:
        for i in random.sample(zeros, MAX_ANTENNAS - len(ones)):
            bits[i] = 1

def crossover(p1, p2):
    if random.random() > CX_PROB:
        return p1[:], p2[:]
    pt = random.randint(1, GRID_SIZE - 2)
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    repair(c1); repair(c2)
    return c1, c2

def mutate(bits):
    for i in range(GRID_SIZE):
        if random.random() < MUT_PROB:
            bits[i] ^= 1
    repair(bits)

def fitness(bits: List[int]) -> Tuple[float, int, int, int]:
    env = CellularNetworkEnvTorch()
    env.antenna_grid = bits_to_tensor(bits)
    for _ in range(EP_STEPS):
        env.step()
    fails, reds = env.coverage_stats()
    ant_cost = int(th.sum(env.antenna_grid).item())
    score = W_FAIL * fails + W_RED * reds + W_ANT * ant_cost
    return score, fails, reds, ant_cost

if __name__ == "__main__":
    # Place the entire GA loop inside this block
    pop = [random_layout() for _ in range(POP_SIZE)]
    best_bits, best_score  = None, float('inf')
    best_triplet = None
    start = time.time()

    for g in range(GENERATIONS):
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            scored = list(pool.map(fitness, pop))

        for ind, (score, f, r, a) in zip(pop, scored):
            if score < best_score:
                best_bits, best_score = ind[:], score
                best_triplet = (f, r, a)

        gen_time = time.time() - start
        avg_per_gen = gen_time / (g + 1)
        eta = timedelta(seconds=int(avg_per_gen * (GENERATIONS - g - 1)))
        elapsed = timedelta(seconds=int(gen_time))

        print(f"Gen {g:03d}  best={best_score:7.2f}  "
              f"fails={best_triplet[0]}  red={best_triplet[1]}  ant={best_triplet[2]}  "
              f"[Elapsed: {elapsed}, ETA: {eta}]")

        new_pop = []
        while len(new_pop) < POP_SIZE:
            contenders = random.sample(list(zip(pop, scored)), TOUR_K)
            p1 = min(contenders, key=lambda t: t[1][0])[0]
            contenders = random.sample(list(zip(pop, scored)), TOUR_K)
            p2 = min(contenders, key=lambda t: t[1][0])[0]
            c1, c2 = crossover(p1, p2)
            mutate(c1); mutate(c2)
            new_pop.extend([c1, c2])
        pop = new_pop[:POP_SIZE]

    elapsed = time.time() - start
    print("\n========== GA Parallel CPU Complete ==========")
    print(f"Elapsed time : {elapsed/60:.1f} min")
    print(f"Best score   : {best_score:.2f}")
    print(f"Failures     : {best_triplet[0]}")
    print(f"Redirects    : {best_triplet[1]}")
    print(f"Antenna count: {best_triplet[2]}")
    print("Best layout grid (1 = antenna):")
    print(np.array(bits_to_tensor(best_bits).cpu()))

