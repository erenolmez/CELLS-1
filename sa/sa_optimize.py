"""
sa_optimize.py
==============

Simulated‑Annealing optimiser for static antenna placement on the 6×6
CellularNetworkEnv grid.  The objective combines failures, redirects and
antenna cost after a 10‑day simulation.

Algorithm outline
-----------------
1.  Start from a random layout with MAX_ANTENNAS ones.
2.  At each temperature T:
      • propose a neighbour by relocating / flipping an antenna
      • run the simulation for EP_STEPS (= 10 days) and get fitness
      • accept if Δ≤0  OR  rand() < exp(-Δ / T)
3.  Gradually cool T ← α·T until Tmin.
4.  Return the best layout encountered.

You can change weights, cooling schedule or neighbour rules easily.
"""

from __future__ import annotations
import random, time
from typing import List, Tuple

import numpy as np
from grid_env import CellularNetworkEnv


# ─────────────────────────────────────────────
# 1.  Hyper‑parameters
# ─────────────────────────────────────────────
MAX_ANTENNAS   = CellularNetworkEnv().max_antennas  # 16  (change if you like)
EP_STEPS       = 240        # 10 days × 24 h
T_INIT         = 10.0       # starting temperature
T_MIN          = 0.05       # stop when T < T_MIN
ALPHA          = 0.95       # cooling rate (T ← α·T)
ITER_PER_TEMP  = 100        # SA inner loop at each temperature

# Fitness weights
W_FAIL   = 3.0
W_RED    = 0.1
W_ANT    = 0.5

SEED = 42
random.seed(SEED); np.random.seed(SEED)


# ─────────────────────────────────────────────
# 2.  Encoding helpers
# ─────────────────────────────────────────────
ROWS, COLS = 6, 6
GRID_SIZE  = ROWS * COLS

def bits_to_grid(bits: List[int]) -> np.ndarray:
    return np.array(bits, dtype=int).reshape((ROWS, COLS))

def random_layout() -> List[int]:
    """Exactly MAX_ANTENNAS ones, the rest zeros."""
    bits = [0] * GRID_SIZE
    for idx in random.sample(range(GRID_SIZE), MAX_ANTENNAS):
        bits[idx] = 1
    return bits

def neighbour(bits: List[int]) -> List[int]:
    """Randomly move one antenna to an empty cell (keeps count constant)."""
    ones  = [i for i,b in enumerate(bits) if b]
    zeros = [i for i,b in enumerate(bits) if not b]
    new   = bits[:]
    out   = random.choice(ones)
    inp   = random.choice(zeros)
    new[out]  = 0
    new[inp]  = 1
    return new

# ─────────────────────────────────────────────
# 3.  Fitness function (lower is better)
# ─────────────────────────────────────────────
def fitness(bits: List[int]) -> float:
    env = CellularNetworkEnv()
    env.antenna_grid = bits_to_grid(bits)
    for _ in range(EP_STEPS):
        env.step(None)                   # layout fixed, users move
    _, fails, reds = env.check_coverage()
    ant_cost = np.sum(env.antenna_grid)
    return (W_FAIL * fails +
            W_RED  * reds  +
            W_ANT  * ant_cost)

# ─────────────────────────────────────────────
# 4.  Simulated Annealing main loop
# ─────────────────────────────────────────────
start_time = time.time()

current_bits = random_layout()
current_fit  = fitness(current_bits)
best_bits, best_fit = current_bits[:], current_fit

T = T_INIT
iteration = 0
print("Starting SA…")
while T > T_MIN:
    for _ in range(ITER_PER_TEMP):
        iteration += 1
        cand_bits = neighbour(current_bits)
        cand_fit  = fitness(cand_bits)
        delta     = cand_fit - current_fit

        if delta <= 0 or random.random() < np.exp(-delta / T):
            current_bits, current_fit = cand_bits, cand_fit
            if cand_fit < best_fit:
                best_bits, best_fit = cand_bits[:], cand_fit

    print(f"T={T:4.2f}  iter={iteration:5d}  "
          f"current={current_fit:7.2f}  best={best_fit:7.2f}")
    T *= ALPHA

elapsed = time.time() - start_time

# Evaluate best layout components
env_best = CellularNetworkEnv()
env_best.antenna_grid = bits_to_grid(best_bits)
for _ in range(EP_STEPS):
    env_best.step(None)
_, best_fails, best_reds = env_best.check_coverage()
best_ant_count = np.sum(env_best.antenna_grid)

# ─────────────────────────────────────────────
# 5.  Output result
# ─────────────────────────────────────────────
print("\n========== SA Complete ==========")
print(f"Elapsed            : {elapsed/60:.1f} min")
print(f"Best fitness score : {best_fit:.2f}")
print(f"Failures           : {best_fails}")
print(f"Redirects          : {best_reds}")
print(f"Antenna count      : {best_ant_count}")
print("Best layout grid (1 = antenna):")
print(bits_to_grid(best_bits))
