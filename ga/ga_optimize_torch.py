# ga_optimize_torch.py
import torch as th
import numpy as np
import random, time
from typing import List
from collections import defaultdict

# ─────────────────────────────────────────
# 1. Hyper‑parameters & GPU setup
# ─────────────────────────────────────────
ROWS, COLS  = 6, 6
GRID_SIZE   = ROWS * COLS
POP_SIZE    = 64
GENERATIONS = 120
EP_STEPS    = 240

# --- antenna maths consistent with CellularNetworkEnv ------------ ### CHANGED ###
TOTAL_USERS     = 5000
ANT_CAPACITY    = 300
COVERAGE_RADIUS = 1

min_by_capacity  = TOTAL_USERS / ANT_CAPACITY
min_by_coverage  = GRID_SIZE / ((2 * COVERAGE_RADIUS + 1) ** 2)   # cells / footprint
INIT_ANTENNAS    = int(min_by_capacity)                            # 16
MAX_ANTENNAS     = int(np.ceil(max(min_by_capacity, min_by_coverage) * 1.3))  # 22
# ------------------------------------------------------------------ ### CHANGED ###

W_FAIL, W_RED, W_ANT = 3.0, 0.1, 0.5
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
print(f"🚀 Batched GA  |  pop={POP_SIZE}  gens={GENERATIONS}  device={DEVICE}")

# ─────────────────────────────────────────
# 2. Helper utilities
# ─────────────────────────────────────────
def random_layout() -> List[int]:
    """Create one random antenna‑bit list with exactly INIT_ANTENNAS ones."""  # ### CHANGED ###
    bits = [0] * GRID_SIZE
    for idx in random.sample(range(GRID_SIZE), INIT_ANTENNAS):                # ### CHANGED ###
        bits[idx] = 1
    return bits

def bits_to_tensor(batch_bits: List[List[int]]) -> th.Tensor:
    """(B, 36) → (B, 6, 6) int8 tensor on DEVICE"""
    return th.tensor(batch_bits, dtype=th.int8, device=DEVICE).view(-1, ROWS, COLS)

def generate_users(batch_size: int) -> th.Tensor:
    """Return (B, 6, 6) initial user grid for each layout (multinomial)."""
    counts = np.random.multinomial(TOTAL_USERS, [1/GRID_SIZE]*GRID_SIZE, size=batch_size)
    return th.tensor(counts.reshape(batch_size, ROWS, COLS), dtype=th.int32, device=DEVICE)

# ─────────────────────────────────────────
# 3. Batched movement & coverage functions
# ─────────────────────────────────────────
_neighbor_offsets = [(dr, dc) for dr in (-1,0,1) for dc in (-1,0,1) if not (dr==dc==0)]

@th.no_grad()
def step(car: th.Tensor, p_move: float = 0.4) -> th.Tensor:
    """
    Vectorised Markov movement matching env logic:
    • intent to move is p_move
    • each of the 8 directions gets p_move/8
    • if a direction is off‑grid, its share is added to the stay probability
    """  # ### CHANGED ###
    B = car.shape[0]
    new = th.zeros_like(car)
    move_prob = p_move / 8.0

    for r in range(ROWS):
        for c in range(COLS):
            users = car[:, r, c]

            # valid neighbour list
            valid_dirs = [(dr, dc) for dr, dc in _neighbor_offsets
                          if 0 <= r+dr < ROWS and 0 <= c+dc < COLS]
            missing = 8 - len(valid_dirs)

            stay_prob  = (1.0 - p_move) + missing * move_prob
            stay_users = (users.float() * stay_prob).int()
            moved_users = users - stay_users
            per_dir = (moved_users.float() * move_prob).int()

            new[:, r, c] += stay_users
            for dr, dc in valid_dirs:
                new[:, r+dr, c+dc] += per_dir
    return new  # ### CHANGED ###

@th.no_grad()
def coverage(car: th.Tensor, ant: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Return (failures, redirects) vectors of shape (B,)."""
    B = car.shape[0]
    fail = th.zeros(B, dtype=th.int32, device=DEVICE)
    red  = th.zeros_like(fail)
    for r in range(ROWS):
        for c in range(COLS):
            users = car[:, r, c]
            remaining = users.clone()
            for dr, dc in _neighbor_offsets + [(0,0)]:  # include self last
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    cap    = ant[:, nr, nc] * ANT_CAPACITY
                    served = th.minimum(cap, remaining)
                    remaining -= served
                    if (dr != 0 or dc != 0):
                        red += served
                if (remaining == 0).all():
                    break
            fail += remaining
    return fail, red

# ─────────────────────────────────────────
# 4. Fitness (batched)
# ─────────────────────────────────────────
def evaluate_batch(batch_bits: List[List[int]]):
    B = len(batch_bits)
    ant = bits_to_tensor(batch_bits)
    car = generate_users(B)
    cum_fail = th.zeros(B, dtype=th.int32, device=DEVICE)
    cum_red  = th.zeros_like(cum_fail)
    for _ in range(EP_STEPS):
        car = step(car)            # move 1 hour
        f, r = coverage(car, ant)  # failures/redirects this hour
        cum_fail += f
        cum_red  += r
    ant_cnt = ant.view(B, -1).sum(dim=1)
    score   = (W_FAIL*cum_fail + W_RED*cum_red + W_ANT*ant_cnt).float()
    return score.cpu().tolist(), cum_fail.cpu().tolist(), cum_red.cpu().tolist(), ant_cnt.cpu().tolist()

# ─────────────────────────────────────────
# 5. Genetic Algorithm main loop
# ─────────────────────────────────────────
random.seed(42)
np.random.seed(42)

pop = [random_layout() for _ in range(POP_SIZE)]
best_bits, best_score = None, float('inf')
start = time.time()

for gen in range(GENERATIONS):
    scores, fails, reds, ants = evaluate_batch(pop)

    # ---- logging -------------------------------------------------
    gen_best_idx = int(np.argmin(scores))
    if scores[gen_best_idx] < best_score:
        best_score = scores[gen_best_idx]
        best_bits  = pop[gen_best_idx][:]
        best_tuple = (fails[gen_best_idx], reds[gen_best_idx], ants[gen_best_idx])
    elapsed = time.time() - start
    eta = elapsed / (gen+1) * (GENERATIONS - gen - 1)
    print(f"Gen {gen:03d} | best={best_score:7.2f} "
          f"(F={best_tuple[0]} R={best_tuple[1]} A={best_tuple[2]}) "
          f"| ⏱ {elapsed/60:.1f}m ETA {eta/60:.1f}m")
    # ---------------------------------------------------------------

    # Selection: keep top 25%, rest offspring
    ranked = [x for _, x in sorted(zip(scores, pop))]
    survivors = ranked[:POP_SIZE // 4]
    offspring = []
    while len(offspring) < POP_SIZE - len(survivors):
        p1, p2 = random.sample(survivors, 2)
        pt = random.randint(1, GRID_SIZE - 2)
        child = p1[:pt] + p2[pt:]

        # mutate bit flip
        for i in range(GRID_SIZE):
            if random.random() < 0.02:
                child[i] ^= 1

        # ---- repair antenna count (range INIT_ANTENNAS … MAX_ANTENNAS) --- ### CHANGED ###
        ones  = [i for i, b in enumerate(child) if b]
        zeros = [i for i, b in enumerate(child) if not b]
        if len(ones) > MAX_ANTENNAS:
            for i in random.sample(ones, len(ones) - MAX_ANTENNAS):
                child[i] = 0
        elif len(ones) < INIT_ANTENNAS:
            for i in random.sample(zeros, INIT_ANTENNAS - len(ones)):
                child[i] = 1
        # ------------------------------------------------------------------- ### CHANGED ###
        offspring.append(child)

    pop = survivors + offspring

print("\n🎉 GA‑GPU batch complete")
print(f"Best score : {best_score:.2f}")
print(f"Best layout (1=antenna):\n{np.array(best_bits).reshape(ROWS, COLS)}")
