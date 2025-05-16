import torch as th
import numpy as np
import random, time

# ─────────────────────────────────────────
# 1. Hyperparameters
# ─────────────────────────────────────────
ROWS, COLS = 6, 6
GRID_SIZE = ROWS * COLS
EP_STEPS = 240
T_INIT = 10.0
T_MIN = 0.05
ALPHA = 0.95
ITER_PER_TEMP = 100
MAX_PER_CELL = 5
W_FAIL, W_RED, W_ANT = 2.0, 0.4, 1.2

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ─────────────────────────────────────────
# 2. Movement and coverage (from GA)
# ─────────────────────────────────────────
_neighbor_offsets = [(dr, dc) for dr in (-1,0,1) for dc in (-1,0,1) if not (dr==dc==0)]

def step(car: th.Tensor, p_move: float = 0.4) -> th.Tensor:
    B = car.shape[0]
    new = th.zeros_like(car)
    move_prob = p_move / 8.0

    for r in range(ROWS):
        for c in range(COLS):
            users = car[:, r, c]
            valid_dirs = [(dr, dc) for dr, dc in _neighbor_offsets if 0 <= r+dr < ROWS and 0 <= c+dc < COLS]
            missing = 8 - len(valid_dirs)
            stay_prob = (1.0 - p_move) + missing * move_prob
            stay_users = (users.float() * stay_prob).int()
            moved_users = users - stay_users
            per_dir = (moved_users.float() * move_prob).int()
            new[:, r, c] += stay_users
            for dr, dc in valid_dirs:
                new[:, r+dr, c+dc] += per_dir
    return new

def coverage(car: th.Tensor, ant: th.Tensor, capacity: int = 300) -> tuple[th.Tensor, th.Tensor]:
    B = car.shape[0]
    fail = th.zeros(B, dtype=th.int32, device=device)
    red  = th.zeros_like(fail)
    antenna_load = th.zeros_like(car)

    for r in range(ROWS):
        for c in range(COLS):
            users = car[:, r, c]
            remaining = users.clone()
            saturated_seen = th.zeros(B, dtype=th.bool, device=device)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        cap = ant[:, nr, nc] * capacity
                        free = th.clamp(cap - antenna_load[:, nr, nc], min=0)
                        saturated_seen |= ((free == 0) & (cap > 0))
                        served = th.minimum(free, remaining)
                        antenna_load[:, nr, nc] += served
                        remaining -= served
                        red += served * saturated_seen.int()
                        if (remaining == 0).all():
                            break
                if (remaining == 0).all():
                    break
            fail += remaining
    return fail, red

# ─────────────────────────────────────────
# 3. Encoding + Fitness
# ─────────────────────────────────────────
def random_layout() -> th.Tensor:
    return th.randint(0, MAX_PER_CELL+1, (1, ROWS, COLS), device=device)

def neighbor(layout: th.Tensor) -> th.Tensor:
    new = layout.clone()
    r, c = random.randint(0, ROWS-1), random.randint(0, COLS-1)
    delta = random.choice([-1, 1])
    new[0, r, c] = th.clamp(new[0, r, c] + delta, 0, MAX_PER_CELL)
    return new

def fitness(layout: th.Tensor) -> tuple[float, int, int, int, float, float]:
    B = layout.shape[0]
    car = generate_users(B)
    ant = layout
    total_fail = th.zeros(B, dtype=th.int32, device=device)
    total_red  = th.zeros_like(total_fail)
    for _ in range(EP_STEPS):
        car = step(car)
        f, r = coverage(car, ant)
        total_fail += f
        total_red  += r
    ant_cnt = ant.view(B, -1).sum(dim=1)
    score = (W_FAIL*total_fail + W_RED*total_red + W_ANT*ant_cnt).float()
    avg_fail = total_fail.float() / EP_STEPS
    avg_red  = total_red.float() / EP_STEPS
    return score.item(), total_fail.item(), total_red.item(), ant_cnt.item(), avg_fail.item(), avg_red.item()

def generate_users(batch_size: int) -> th.Tensor:
    counts = np.random.multinomial(5000, [1/(ROWS*COLS)]*(ROWS*COLS), size=batch_size)
    return th.tensor(counts.reshape(batch_size, ROWS, COLS), dtype=th.int32, device=device)

# ─────────────────────────────────────────
# 4. Simulated Annealing Loop
# ─────────────────────────────────────────
start = time.time()
current = random_layout()
current_score, *_ = fitness(current)
best = current.clone()
best_score = current_score
T = T_INIT
iteration = 0
print("Starting Torch-SA...")

while T > T_MIN:
    for _ in range(ITER_PER_TEMP):
        iteration += 1
        cand = neighbor(current)
        cand_score, *_ = fitness(cand)
        delta = cand_score - current_score
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current = cand.clone()
            current_score = cand_score
            if cand_score < best_score:
                best = cand.clone()
                best_score = cand_score
    print(f"T={T:.2f}  iter={iteration:5d}  current={current_score:7.2f}  best={best_score:7.2f}")
    T *= ALPHA

elapsed = time.time() - start
_, f, r, a, af, ar = fitness(best)
print("\n========== Torch-SA Complete ==========")
print(f"Elapsed time : {elapsed/60:.1f} min")
print(f"Best score   : {best_score:.2f}")
print(f"Failures     : {f}")
print(f"Redirects    : {r}")
print(f"Antenna count: {a}")
print(f"Avg failures : {af:.1f}")
print(f"Avg redirects: {ar:.1f}")
print("Best layout grid (antennas per cell):")
print(best[0].int().cpu().numpy())