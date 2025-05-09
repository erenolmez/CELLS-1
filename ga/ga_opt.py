"""
ga_optimize.py
==============

Static antenna‑placement optimiser using a simple Genetic Algorithm
on top of CellularNetworkEnv.  One chromosome = 36‑bit vector; exactly
`max_antennas` bits must be 1.  Fitness = failures after simulating
10 days (240 env steps).

Usage:
    python ga_optimize.py
"""

from __future__ import annotations
import random, itertools, time
from typing import List

import numpy as np
from grid_env import CellularNetworkEnv


# ─────────────────────────────────────────────
# 1.  GA hyper‑parameters
# ─────────────────────────────────────────────
POP_SIZE        = 60
GENERATIONS     = 120
TOURNAMENT_K    = 3
CX_PROB         = 0.7     # crossover probability
MUT_PROB        = 0.05    # per‑bit mutation probability

EP_STEPS        = 240     # 10 days  × 24 h
SEED            = 42
random.seed(SEED); np.random.seed(SEED)


# ─────────────────────────────────────────────
# 2.  Environment constants
# ─────────────────────────────────────────────
env_proto = CellularNetworkEnv()
GRID_SIZE       = env_proto.rows * env_proto.cols        # 36
MAX_ANTENNAS    = env_proto.max_antennas                 # 16


# ─────────────────────────────────────────────
# 3.  Helper functions
# ─────────────────────────────────────────────
def random_chromosome() -> List[int]:
    """Binary vector with exactly MAX_ANTENNAS ones."""
    bits = [0] * GRID_SIZE
    ones = random.sample(range(GRID_SIZE), MAX_ANTENNAS)
    for idx in ones:
        bits[idx] = 1
    return bits

def repair(chromo: List[int]) -> None:
    """Ensure chromosome has exactly MAX_ANTENNAS ones (in‑place)."""
    ones = [i for i,b in enumerate(chromo) if b]
    zeros = [i for i,b in enumerate(chromo) if not b]
    if len(ones) > MAX_ANTENNAS:          # too many 1s → flip extras to 0
        flip = random.sample(ones, len(ones) - MAX_ANTENNAS)
        for i in flip: chromo[i] = 0
    elif len(ones) < MAX_ANTENNAS:        # too few 1s → flip some 0s to 1
        flip = random.sample(zeros, MAX_ANTENNAS - len(ones))
        for i in flip: chromo[i] = 1

def crossover(p1: List[int], p2: List[int]) -> tuple[List[int], List[int]]:
    if random.random() > CX_PROB:
        return p1[:], p2[:]
    point = random.randint(1, GRID_SIZE-2)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    repair(c1); repair(c2)
    return c1, c2

def mutate(chromo: List[int]) -> None:
    for i in range(GRID_SIZE):
        if random.random() < MUT_PROB:
            chromo[i] ^= 1   # flip bit
    repair(chromo)

def layout_to_grid(bits: List[int]) -> np.ndarray:
    """Convert 36‑bit vector to 6 × 6 antenna_grid (int)."""
    grid = np.array(bits, dtype=int).reshape((env_proto.rows, env_proto.cols))
    return grid

def fitness(bits: List[int]) -> tuple[int,int]:
    """Return (failures, redirects) after a 10‑day simulation."""
    env = CellularNetworkEnv()
    env.antenna_grid = layout_to_grid(bits)
    # run simulation
    for _ in range(EP_STEPS):
        env.step(None)          # None = no‑op (antenna layout fixed)
    _, fails, reds = env.check_coverage()
    return fails, reds          # tuple so GA can tiebreak

# ─────────────────────────────────────────────
# 4.  GA main loop
# ─────────────────────────────────────────────
pop = [random_chromosome() for _ in range(POP_SIZE)]
best_bits   = None
best_score  = (float('inf'), float('inf'))

start = time.time()
for gen in range(GENERATIONS):

    # ---- evaluate population
    scores = [fitness(ind) for ind in pop]
    for bits, score in zip(pop, scores):
        if score < best_score:
            best_bits, best_score = bits[:], score

    print(f"Gen {gen:03d}: best failures={best_score[0]:5d} "
          f"redirects={best_score[1]:5d}")

    # ---- selection (tournament)
    new_pop = []
    while len(new_pop) < POP_SIZE:
        contenders = random.sample(list(zip(pop, scores)), TOURNAMENT_K)
        parent1 = min(contenders, key=lambda t: t[1])[0]
        contenders = random.sample(list(zip(pop, scores)), TOURNAMENT_K)
        parent2 = min(contenders, key=lambda t: t[1])[0]

        # ---- crossover & mutation
        child1, child2 = crossover(parent1, parent2)
        mutate(child1); mutate(child2)
        new_pop.extend([child1, child2])

    pop = new_pop[:POP_SIZE]

elapsed = time.time() - start
print("\n=============  GA Complete  =============")
print(f"Elapsed time: {elapsed/60:.1f} min")
print(f"Best failures : {best_score[0]}")
print(f"Best redirects: {best_score[1]}")

# ---- pretty‑print best layout
best_grid = layout_to_grid(best_bits)
print("\nBest antenna grid (1 = antenna):")
print(best_grid)
