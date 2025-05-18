from __future__ import annotations
import random, time, statistics
from typing import List, Tuple
import numpy as np
from grid_env import CellularNetworkEnv

# ╭────────────────────── GA hyper‑parameters ┍
POP_SIZE           = 60
GENERATIONS        = 120
TOURNAMENT_K       = 3
CX_PROB            = 0.7
MUT_PROB           = 0.05        # bit‑flip probability per gene
MAX_PER_CELL       = 2           # 0,1,2 antennas allowed in each cell
EP_STEPS           = 720       # 10 simulated days
SEED               = 42
random.seed(SEED); np.random.seed(SEED)

# ╭────────────────────── Fitness weights ┍
LAMBDA_FAILURE     = 3.0
LAMBDA_REDIRECT    = 0.10
LAMBDA_ANTENNA     = 2.0

# ╭────────────────────── Environment constants ┍
proto_env    = CellularNetworkEnv(rows=20, cols=20, total_users=50000, antenna_capacity=300, time_step=60)
GRID_SIZE    = proto_env.rows * proto_env.cols   # 36

# ╭────────────────────── GA primitives ┍
Chromosome = List[int]        # length‑36 list of ints (0‑2)

def random_chromosome() -> Chromosome:
    return [random.randint(0, MAX_PER_CELL) for _ in range(GRID_SIZE)]

def crossover(p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    if random.random() > CX_PROB:
        return p1[:], p2[:]
    cut = random.randint(1, GRID_SIZE-2)
    return p1[:cut]+p2[cut:], p2[:cut]+p1[cut:]

def mutate(ch: Chromosome) -> None:
    for i in range(GRID_SIZE):
        if random.random() < MUT_PROB:
            ch[i] = random.randint(0, MAX_PER_CELL)

def chromo_to_grid(ch: Chromosome) -> np.ndarray:
    return np.array(ch, dtype=int).reshape((proto_env.rows, proto_env.cols))

def eval_layout(ch: Chromosome) -> Tuple[float,int,int,int]:
    """Return (fitness, failures, redirects, antenna_count)."""
    env = CellularNetworkEnv()
    env.antenna_grid = chromo_to_grid(ch)
    env.place_users()
    for _ in range(EP_STEPS):
        env.move_users_markov_chain()
    _, fails, reds = env.check_coverage()
    ant_cnt = np.sum(env.antenna_grid)
    fitness = LAMBDA_FAILURE*fails + LAMBDA_REDIRECT*reds + LAMBDA_ANTENNA*ant_cnt
    return fitness, fails, reds, ant_cnt

# ╭────────────────────── GA main loop ┍
pop: List[Chromosome] = [random_chromosome() for _ in range(POP_SIZE)]
best_ch, best_fit, best_tuple = None, float('inf'), None
start = time.time()

for gen in range(GENERATIONS):
    evaluated = [eval_layout(ind) for ind in pop]
    fits = [t[0] for t in evaluated]

    # Track best
    gen_best_idx = int(np.argmin(fits))
    if fits[gen_best_idx] < best_fit:
        best_fit   = fits[gen_best_idx]
        best_ch    = pop[gen_best_idx][:]
        best_tuple = evaluated[gen_best_idx]

    # ━ Detailed generation report ━
    mean_fail   = statistics.mean(t[1] for t in evaluated)
    mean_redir  = statistics.mean(t[2] for t in evaluated)
    mean_cost   = statistics.mean(t[3] for t in evaluated)
    print(f"Gen {gen:03d} | bestF={evaluated[gen_best_idx][1]:4d} "
        f"bestR={evaluated[gen_best_idx][2]:4d} A={evaluated[gen_best_idx][3]:3d} "
        f"| meanF={mean_fail:6.1f} meanR={mean_redir:6.1f} Aµ={mean_cost:4.1f}")


    # ━ Selection (tournament) and reproduction ━
    new_pop: List[Chromosome] = []
    while len(new_pop) < POP_SIZE:
        # Parent selection
        p1 = min(random.sample(pop, TOURNAMENT_K), key=lambda ind: eval_layout(ind)[0])
        p2 = min(random.sample(pop, TOURNAMENT_K), key=lambda ind: eval_layout(ind)[0])
        # Crossover & mutation
        c1, c2 = crossover(p1, p2)
        mutate(c1); mutate(c2)
        new_pop.extend([c1, c2])
    pop = new_pop[:POP_SIZE]

# ╭────────────────────── GA finished ┍
print("\n=========== GA Complete ===========")
print(f"Elapsed: {(time.time()-start)/60:.1f} min")
print(f"Best fitness   : {best_fit:.2f}")
print(f"Failures       : {best_tuple[1]}")
print(f"Redirects      : {best_tuple[2]}")
print(f"Antenna count  : {best_tuple[3]}")
print("Best layout grid (antenna counts):")
print(chromo_to_grid(best_ch))
