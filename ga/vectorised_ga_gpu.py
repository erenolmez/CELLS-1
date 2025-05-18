import torch
import math            # ①  <-- new
import time
import random
import statistics
from typing import List, Tuple

# ╭────────────────────── GA hyper-parameters ┍
POP_SIZE     = 60
GENERATIONS  = 120
TOURNAMENT_K = 3
CX_PROB      = 0.7
MUT_PROB     = 0.05
MAX_PER_CELL = 2
EP_STEPS     = 240
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_USERS  = 50000
ANT_CAPACITY = 300
torch.manual_seed(SEED)
random.seed(SEED)

# ╭────────────────────── Fitness weights ┍
LAMBDA_FAILURE  = 2.00
LAMBDA_REDIRECT = 0.10
LAMBDA_ANTENNA  = 1.00

# ╭────────────────────── Vectorized Environment ┍
class VectorizedCellularNetworkEnv:
    def __init__(self, batch_size, rows=20, cols=20,
                 total_users=50000, antenna_capacity=300,
                 time_step=60, device=None):
        self.batch_size = batch_size
        self.rows = rows
        self.cols = cols
        self.total_users = total_users
        self.antenna_capacity = antenna_capacity
        self.time_step = time_step
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.antenna_grid = torch.zeros(
            (batch_size, rows, cols), dtype=torch.int32, device=self.device)
        self.user_pos = None  # (B, U, 2)

    def set_antenna_grids(self, antenna_grids: torch.Tensor):
        self.antenna_grid = antenna_grids.to(device=self.device)

    def place_users(self):
        rows = torch.randint(
            0, self.rows, (self.batch_size, self.total_users), device=self.device)
        cols = torch.randint(
            0, self.cols, (self.batch_size, self.total_users), device=self.device)
        self.user_pos = torch.stack((rows, cols), dim=2)  # (B, U, 2)

    def move_users_markov_chain(self):
        delta = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
                             device=self.device)  # stay, N, S, W, E
        actions = torch.randint(
            0, 5, (self.batch_size, self.total_users), device=self.device)
        move = delta[actions]
        new_pos = self.user_pos + move
        new_pos[..., 0].clamp_(0, self.rows - 1)
        new_pos[..., 1].clamp_(0, self.cols - 1)
        self.user_pos = new_pos

    def check_coverage(self):
        B, U, _ = self.user_pos.shape
        coverage = torch.zeros(
            (B, self.rows, self.cols), dtype=torch.int32, device=self.device)

        row_idx = self.user_pos[..., 0]
        col_idx = self.user_pos[..., 1]
        for b in range(B):
            coverage[b].index_put_(
                (row_idx[b], col_idx[b]),
                torch.ones(U, dtype=torch.int32, device=self.device),
                accumulate=True)

        antenna_capacity_grid = self.antenna_grid * self.antenna_capacity
        covered = torch.minimum(coverage, antenna_capacity_grid)
        total_covered = covered.sum(dim=(1, 2))
        fails = torch.tensor(self.total_users, device=self.device).expand(
            B) - total_covered
        redirects = torch.sum(coverage > antenna_capacity_grid, dim=(1, 2))
        return total_covered, fails, redirects


# ╭────────────────────── Grid/GA settings ┍
GRID_ROWS = 20
GRID_COLS = 20
GRID_SIZE = GRID_ROWS * GRID_COLS

# ╭────────────────────── GA functions ┍
def random_population(pop_size: int) -> torch.Tensor:
    """
    Build an initial population in which **every chromosome has exactly
    the minimum number of antennas needed to (theoretically) serve all
    users** and each cell is either 0 or 1 antenna (no 2's yet).

    Later mutation/crossover can still move counts up to MAX_PER_CELL.
    """
    min_antennas = int(math.ceil(TOTAL_USERS / ANT_CAPACITY))   # ②
    pop = torch.zeros(pop_size, GRID_SIZE, dtype=torch.int32, device=DEVICE)
    idx = torch.arange(GRID_SIZE, device=DEVICE)

    for i in range(pop_size):
        chosen = idx[torch.randperm(GRID_SIZE)[:min_antennas]]
        pop[i, chosen] = 1                                      # set to one antenna
    return pop


def crossover_batch(p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand(p1.shape, device=DEVICE) < 0.5
    c1 = torch.where(mask, p1, p2)
    c2 = torch.where(mask, p2, p1)
    return c1, c2


def mutate_batch(pop: torch.Tensor) -> torch.Tensor:
    mutation_mask = torch.rand(pop.shape, device=DEVICE) < MUT_PROB
    random_genes = torch.randint(0, MAX_PER_CELL + 1, pop.shape, device=DEVICE)
    mutated = torch.where(mutation_mask, random_genes, pop)
    return mutated.clamp_(0, MAX_PER_CELL)        # ③ keep 0…2


def eval_layout_batch(pop: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grids = pop.view(-1, GRID_ROWS, GRID_COLS)
    env = VectorizedCellularNetworkEnv(
        batch_size=grids.shape[0],
        rows=GRID_ROWS,
        cols=GRID_COLS,
        total_users=TOTAL_USERS,
        antenna_capacity=ANT_CAPACITY,
        device=DEVICE
    )
    env.set_antenna_grids(grids)
    env.place_users()

    fails_total = torch.zeros(grids.shape[0], dtype=torch.float32, device=DEVICE)
    reds_total = torch.zeros(grids.shape[0], dtype=torch.float32, device=DEVICE)

    for _ in range(EP_STEPS):
        env.move_users_markov_chain()
        _, fails, reds = env.check_coverage()
        fails_total += fails.float()
        reds_total += reds.float()

    fails_mean = fails_total / EP_STEPS
    reds_mean = reds_total / EP_STEPS
    ant_cnt = torch.sum(grids, dim=(1, 2)).float()

    failure_penalty = fails_mean / TOTAL_USERS
    redirect_penalty = reds_mean / TOTAL_USERS
    antenna_cost = ant_cnt / (MAX_PER_CELL * GRID_SIZE)

    reward = 1.0 - (2.0 * failure_penalty + 0.1 * redirect_penalty + 1.0 * antenna_cost)
    reward = torch.clamp(reward, 0.0, 1.0)

    fitness = -reward
    return fitness, fails_mean, reds_mean, ant_cnt


def tournament_selection(pop: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
    indices = torch.randint(0, pop.size(0), (TOURNAMENT_K,), device=DEVICE)
    selected = pop[indices]
    selected_fitness = fitness[indices]
    return selected[torch.argmin(selected_fitness)]


# ╭────────────────────── Main GA loop ┍
pop = random_population(POP_SIZE)
best_ch = None
best_fit = float('inf')
best_tuple = None
start = time.time()

for gen in range(GENERATIONS):
    fitness, fails, reds, ant_cnts = eval_layout_batch(pop)
    gen_best_idx = torch.argmin(fitness).item()

    if fitness[gen_best_idx] < best_fit:
        best_fit = fitness[gen_best_idx].item()
        best_ch = pop[gen_best_idx].clone()
        best_tuple = (best_fit, fails[gen_best_idx].item(),
                      reds[gen_best_idx].item(), ant_cnts[gen_best_idx].item())

    print(f"Gen {gen:03d} | meanF={fails.mean():6.1f} "
          f"meanR={reds.mean():6.1f} A={int(ant_cnts[0].item())}")

    # Reproduction
    new_pop = []
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(pop, fitness)
        p2 = tournament_selection(pop, fitness)
        c1, c2 = crossover_batch(p1.unsqueeze(0), p2.unsqueeze(0))
        c1 = mutate_batch(c1)
        c2 = mutate_batch(c2)
        new_pop.append(c1.squeeze(0))
        new_pop.append(c2.squeeze(0))
    pop = torch.stack(new_pop[:POP_SIZE])

# ╭────────────────────── Final output ┍
print("\n=========== GA Complete ===========")
print(f"Elapsed: {(time.time() - start)/60:.1f} min")
print(f"Best fitness   : {best_tuple[0]:.2f}")
print(f"Failures       : {best_tuple[1]}")
print(f"Redirects      : {best_tuple[2]}")
print(f"Antenna count  : {best_tuple[3]}")
print("Best layout grid (antenna counts):")
print(best_ch.view(GRID_ROWS, GRID_COLS).int().cpu().numpy())
