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
    rows, cols = 20, 20
    num_cells  = rows * cols

    def __init__(self, total_users: int = 50000, antenna_capacity: int = 300, time_step: int = 60):
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
        antenna_load = th.zeros((self.rows, self.cols), dtype=th.int32, device=device)
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(self.rows):
            for c in range(self.cols):
                users = int(self.car_grid[r, c].item())
                if users == 0:
                    continue

                remaining = users
                saturated_seen = False

                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                        continue

                    cap = int(self.antenna_grid[nr, nc].item()) * self.antenna_capacity
                    used = int(antenna_load[nr, nc].item())
                    free = max(0, cap - used)

                    if free == 0 and cap > 0:
                        saturated_seen = True
                        continue

                    if remaining == 0:
                        break

                    served = min(free, remaining)
                    antenna_load[nr, nc] += served
                    remaining -= served

                    if saturated_seen:
                        redirects += served

                failures += remaining
        return failures, redirects

    def step(self):
        self.move_users()
        self.sim_time_h += 1

# ──── GA Setup ────
ROWS, COLS = 20, 20
GRID_SIZE  = ROWS * COLS
POP_SIZE   = 60
GENERATIONS = 120
TOUR_K     = 3
CX_PROB    = 0.7
MUT_PROB   = 0.05
EP_STEPS   = 240
W_FAIL, W_RED, W_ANT = 2.0, 0.1, 0.5
MAX_PER_CELL = 2

def bits_to_tensor(bits: List[int]) -> th.Tensor:
    return th.tensor(bits, dtype=th.int8, device=device).reshape((ROWS, COLS))

def random_layout() -> List[int]:
    return [random.randint(0, MAX_PER_CELL) for _ in range(GRID_SIZE)]

def repair(bits: List[int]):
    for i in range(len(bits)):
        bits[i] = max(0, min(MAX_PER_CELL, bits[i]))

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
            delta = random.choice([-1, 1])
            bits[i] = max(0, min(MAX_PER_CELL, bits[i] + delta))
    repair(bits)

def fitness(bits: List[int]) -> Tuple[float, int, int, int, float, float]:
    env = CellularNetworkEnvTorch()
    env.antenna_grid = bits_to_tensor(bits)
    total_failures = 0
    total_redirects = 0
    for _ in range(EP_STEPS):
        env.step()
        f, r = env.coverage_stats()
        total_failures += f
        total_redirects += r
    ant_cost = int(th.sum(env.antenna_grid).item())
    score = W_FAIL * total_failures + W_RED * total_redirects + W_ANT * ant_cost
    return score, total_failures, total_redirects, ant_cost, total_failures / EP_STEPS, total_redirects / EP_STEPS

if __name__ == "__main__":
    pop = [random_layout() for _ in range(POP_SIZE)]
    best_bits, best_score  = None, float('inf')
    best_triplet = None
    start = time.time()

    for g in range(GENERATIONS):
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            scored = list(pool.map(fitness, pop))

        for ind, (score, f, r, a, avgf, avgr) in zip(pop, scored):
            if score < best_score:
                best_bits, best_score = ind[:], score
                best_triplet = (f, r, a, avgf, avgr)

        gen_time = time.time() - start
        avg_per_gen = gen_time / (g + 1)
        eta = timedelta(seconds=int(avg_per_gen * (GENERATIONS - g - 1)))
        elapsed = timedelta(seconds=int(gen_time))

        print(f"Gen {g:03d}  best={best_score:7.2f}  "
              f"fails={best_triplet[0]}  red={best_triplet[1]}  ant={best_triplet[2]}  "
              f"avg_fail={best_triplet[3]:.1f}  avg_red={best_triplet[4]:.1f}  "
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
    print(f"Avg Failures : {best_triplet[3]:.1f}")
    print(f"Avg Redirects: {best_triplet[4]:.1f}")
    print("Best layout grid (num antennas per cell):")
    print(np.array(bits_to_tensor(best_bits).cpu()))