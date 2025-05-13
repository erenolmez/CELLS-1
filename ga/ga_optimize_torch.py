# ga_optimize_torch.py
import torch as th
import numpy as np
import random, time
from typing import List

# ─────────────────────────────────────────
# 1. Hyper‑parameters & device
# ─────────────────────────────────────────
ROWS, COLS   = 6, 6
GRID_SIZE    = ROWS * COLS
POP_SIZE     = 64
GENERATIONS  = 120
EP_STEPS     = 240

TOTAL_USERS, ANT_CAPACITY, COVERAGE_RADIUS = 5000, 300, 1

min_by_capacity = TOTAL_USERS / ANT_CAPACITY
min_by_coverage = GRID_SIZE / ((2 * COVERAGE_RADIUS + 1) ** 2)
INIT_ANTENNAS   = int(min_by_capacity)
MAX_ANTENNAS    = int(np.ceil(max(min_by_capacity, min_by_coverage) * 1.3))

W_FAIL, W_RED, W_ANT = 3.0, 0.1, 0.5
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
print(f"🚀 Batched GA | pop={POP_SIZE} gens={GENERATIONS} device={DEVICE}")

# ─────────────────────────────────────────
# 2. helper utilities
# ─────────────────────────────────────────
def random_layout() -> List[int]:
    """Bit‑vector with exactly INIT_ANTENNAS ones."""
    bits = [0] * GRID_SIZE
    for idx in random.sample(range(GRID_SIZE), INIT_ANTENNAS):
        bits[idx] = 1
    return bits

def bits_to_tensor(batch_bits: List[List[int]]) -> th.Tensor:
    """(B, 36) → (B, 6, 6)"""
    return th.tensor(batch_bits, dtype=th.int8, device=DEVICE).view(-1, ROWS, COLS)

def generate_users(batch: int) -> th.Tensor:
    counts = np.random.multinomial(TOTAL_USERS, [1 / GRID_SIZE] * GRID_SIZE, size=batch)
    return th.tensor(counts.reshape(batch, ROWS, COLS), dtype=th.int32, device=DEVICE)

# ─────────────────────────────────────────
# 3. movement (unchanged)
# ─────────────────────────────────────────
_neighbor_offsets = [(dr, dc) for dr in (-1, 0, 1)
                     for dc in (-1, 0, 1) if dr or dc]

@th.no_grad()
def step(car: th.Tensor, p_move: float = 0.4) -> th.Tensor:
    new = th.zeros_like(car)
    move_prob = p_move / 8.0
    for r in range(ROWS):
        for c in range(COLS):
            users = car[:, r, c]
            valid = [(dr, dc) for dr, dc in _neighbor_offsets
                     if 0 <= r + dr < ROWS and 0 <= c + dc < COLS]
            stay_p = (1 - p_move) + (8 - len(valid)) * move_prob
            stay   = (users.float() * stay_p).int()
            share  = ((users - stay).float() * move_prob).int()
            new[:, r, c] += stay
            for dr, dc in valid:
                new[:, r + dr, c + dc] += share
    return new

# ─────────────────────────────────────────
# 4. coverage with redirect‑after‑saturation
# ─────────────────────────────────────────
OFFSETS = [                             ### CHANGED ###
    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

@th.no_grad()
def coverage(car: th.Tensor, ant: th.Tensor):
    """
    Returns failures & redirects (B,).
    Redirect counted only after a saturated antenna was hit first.
    """                                ### CHANGED ###
    B = car.shape[0]
    fail = th.zeros(B, dtype=th.int32, device=DEVICE)
    red  = th.zeros_like(fail)

    cap_left = (ant * ANT_CAPACITY).clone().to(th.int32)

    for r in range(ROWS):
        for c in range(COLS):
            rem = car[:, r, c].clone()
            sat_seen = th.zeros(B, dtype=th.bool, device=DEVICE)

            for dr, dc in OFFSETS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue

                free = cap_left[:, nr, nc]
                sat_seen |= (free == 0) & (ant[:, nr, nc] > 0)   # full antenna met
                served = th.minimum(free, rem)
                cap_left[:, nr, nc] -= served
                rem -= served

                red += served * sat_seen.int()                   # redirect rule

                if (rem == 0).all():
                    break

            fail += rem
    return fail, red                                             ### CHANGED ###

# ─────────────────────────────────────────
# 5. fitness
# ─────────────────────────────────────────
def evaluate_batch(bits):
    B   = len(bits)
    ant = bits_to_tensor(bits)
    car = generate_users(B)
    cum_fail = th.zeros(B, dtype=th.int32, device=DEVICE)
    cum_red  = th.zeros_like(cum_fail)

    for _ in range(EP_STEPS):
        car = step(car)
        f, r = coverage(car, ant)
        cum_fail += f
        cum_red  += r

    ant_cnt = ant.view(B, -1).sum(dim=1)
    score   = (W_FAIL * cum_fail + W_RED * cum_red + W_ANT * ant_cnt).float()
    return (score.cpu().tolist(), cum_fail.cpu().tolist(),
            cum_red.cpu().tolist(), ant_cnt.cpu().tolist())

# ─────────────────────────────────────────
# 6. GA main loop (identical logic)
# ─────────────────────────────────────────
random.seed(42); np.random.seed(42)
pop = [random_layout() for _ in range(POP_SIZE)]
best_bits, best_score = None, float('inf')
start = time.time()

for gen in range(GENERATIONS):
    scores, fails, reds, ants = evaluate_batch(pop)

    idx = int(np.argmin(scores))
    if scores[idx] < best_score:
        best_score = scores[idx]
        best_bits  = pop[idx][:]
        best_tuple = (fails[idx], reds[idx], ants[idx])

    eta = (time.time() - start) / (gen + 1) * (GENERATIONS - gen - 1)
    print(f"Gen {gen:03d} | best={best_score:7.1f} "
          f"(F={best_tuple[0]} R={best_tuple[1]} A={best_tuple[2]}) "
          f"| ETA {eta/60:.1f}m")

    ranked = [x for _, x in sorted(zip(scores, pop))]
    survivors = ranked[:POP_SIZE // 4]
    offspring = []
    while len(offspring) < POP_SIZE - len(survivors):
        p1, p2 = random.sample(survivors, 2)
        pt = random.randint(1, GRID_SIZE - 2)
        child = p1[:pt] + p2[pt:]

        for i in range(GRID_SIZE):
            if random.random() < 0.02:
                child[i] ^= 1

        ones  = [i for i, b in enumerate(child) if b]
        zeros = [i for i, b in enumerate(child) if not b]
        if len(ones) > MAX_ANTENNAS:
            for i in random.sample(ones, len(ones) - MAX_ANTENNAS):
                child[i] = 0
        elif len(ones) < INIT_ANTENNAS:
            for i in random.sample(zeros, INIT_ANTENNAS - len(ones)):
                child[i] = 1

        offspring.append(child)

    pop = survivors + offspring

print("\n🎉 GA‑GPU batch complete")
print(f"Best score : {best_score:.1f}")
print(f"Best layout:\n{np.array(best_bits).reshape(ROWS, COLS)}")
