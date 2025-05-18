from __future__ import annotations
import random, time, os
from typing import List, Tuple
import numpy as np
import torch as th
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

# ─────────────────────────────────────────────
# 1.  Hyper‑parameters
# ─────────────────────────────────────────────
ROWS, COLS     = 20, 20
GRID_SIZE      = ROWS * COLS
MAX_PER_CELL   = 5
EP_STEPS       = 720
T_INIT         = 10.0
T_MIN          = 0.05
ALPHA          = 0.95
ITER_PER_TEMP  = 100
W_FAIL, W_RED, W_ANT = 2.0, 0.1, 0.5
SEED = 42
random.seed(SEED); np.random.seed(SEED)
NUM_WORKERS = os.cpu_count() or 4

device = "cpu"

# ─────────────────────────────────────────────
# 2.  Environment for SA (shared logic)
# ─────────────────────────────────────────────
class CellularNetworkEnvTorch:
    rows, cols = ROWS, COLS
    num_cells  = rows * cols

    def __init__(self, total_users: int = 50000, antenna_capacity: int = 300, time_step: int = 60):
        self.total_users     = total_users
        self.antenna_capacity = antenna_capacity
        self.time_step       = time_step
        self.coverage_rad    = 1
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

# ─────────────────────────────────────────────
# 3.  SA encoding
# ─────────────────────────────────────────────
def random_layout() -> List[int]:
    return [random.randint(0, MAX_PER_CELL) for _ in range(GRID_SIZE)]

def neighbour(bits: List[int]) -> List[int]:
    new = bits[:]
    i = random.randint(0, GRID_SIZE - 1)
    delta = random.choice([-1, 1])
    new[i] = max(0, min(MAX_PER_CELL, new[i] + delta))
    return new

def bits_to_grid(bits: List[int]) -> th.Tensor:
    return th.tensor(bits, dtype=th.int8, device=device).reshape((ROWS, COLS))

def fitness(bits: List[int]) -> Tuple[float, float, float]:
    env = CellularNetworkEnvTorch()
    env.antenna_grid = bits_to_grid(bits)
    total_failures = 0
    total_redirects = 0
    for _ in range(EP_STEPS):
        env.step()
        f, r = env.coverage_stats()
        total_failures += f
        total_redirects += r
    ant_cost = int(th.sum(env.antenna_grid).item())
    score = W_FAIL * total_failures + W_RED * total_redirects + W_ANT * ant_cost
    avg_fail = total_failures / EP_STEPS
    avg_red = total_redirects / EP_STEPS
    return score, avg_fail, avg_red

# ─────────────────────────────────────────────
# 4.  Simulated Annealing loop
# ─────────────────────────────────────────────
start_time = time.time()
current_bits = random_layout()
current_fit, current_avg_fail, current_avg_red = fitness(current_bits)
best_bits, best_fit = current_bits[:], current_fit
best_avg_fail, best_avg_red = current_avg_fail, current_avg_red
T = T_INIT
iteration = 0
print("Starting SA...")

while T > T_MIN:
    for _ in range(ITER_PER_TEMP):
        iteration += 1
        cand_bits = neighbour(current_bits)
        cand_fit, cand_avg_fail, cand_avg_red = fitness(cand_bits)
        delta     = cand_fit - current_fit

        if delta <= 0 or random.random() < np.exp(-delta / T):
            current_bits, current_fit = cand_bits, cand_fit
            current_avg_fail, current_avg_red = cand_avg_fail, cand_avg_red
            if cand_fit < best_fit:
                best_bits, best_fit = cand_bits[:], cand_fit
                best_avg_fail, best_avg_red = cand_avg_fail, cand_avg_red

    print(f"T={T:4.2f}  iter={iteration:5d}  "
          f"current={current_fit:7.2f}  best={best_fit:7.2f}  "
          f"avg_fail={current_avg_fail:.2f}  avg_red={current_avg_red:.2f}")
    T *= ALPHA

elapsed = time.time() - start_time

# ─────────────────────────────────────────────
# 5.  Output result
# ─────────────────────────────────────────────
env_best = CellularNetworkEnvTorch()
env_best.antenna_grid = bits_to_grid(best_bits)
for _ in range(EP_STEPS):
    env_best.step()
best_fails, best_reds = env_best.coverage_stats()
best_ant_count = int(th.sum(env_best.antenna_grid).item())

print("\n========== SA Complete ==========")
print(f"Elapsed time      : {elapsed/60:.1f} min")
print(f"Best score        : {best_fit:.2f}")
print(f"Total Failures    : {best_fails}")
print(f"Total Redirects   : {best_reds}")
print(f"Antenna Count     : {best_ant_count}")
print(f"Avg Failures/Step : {best_avg_fail:.2f}")
print(f"Avg Redirects/Step: {best_avg_red:.2f}")
print("Best layout grid (num antennas per cell):")
print(env_best.antenna_grid.cpu().numpy())