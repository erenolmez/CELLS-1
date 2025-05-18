import torch
import time
import random
import statistics
from typing import List, Tuple

# Vectorized environment supporting batch simulation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorizedCellularNetworkEnv:
    def __init__(self, batch_size, rows=20, cols=20, total_users=50000, antenna_capacity=300, time_step=60):
        self.batch_size = batch_size
        self.rows = rows
        self.cols = cols
        self.total_users = total_users
        self.antenna_capacity = antenna_capacity
        self.time_step = time_step

        self.antenna_grid = torch.zeros((batch_size, rows, cols), dtype=torch.int32, device=device)
        self.user_pos = None

    def set_antenna_grids(self, antenna_grids: torch.Tensor):
        self.antenna_grid = antenna_grids.to(device=device)

    def place_users(self):
        rows = torch.randint(0, self.rows, (self.batch_size, self.total_users), device=device)
        cols = torch.randint(0, self.cols, (self.batch_size, self.total_users), device=device)
        self.user_pos = torch.stack((rows, cols), dim=2)

    def move_users_markov_chain(self):
        delta = torch.tensor([[0,0], [-1,0], [1,0], [0,-1], [0,1]], device=device)
        actions = torch.randint(0, 5, (self.batch_size, self.total_users), device=device)
        move = delta[actions]
        new_pos = self.user_pos + move
        new_pos[..., 0].clamp_(0, self.rows - 1)
        new_pos[..., 1].clamp_(0, self.cols - 1)
        self.user_pos = new_pos

    def check_coverage(self):
        B, U, _ = self.user_pos.shape
        coverage = torch.zeros((B, self.rows, self.cols), dtype=torch.int32, device=device)

        for b in range(B):
            row_idx = self.user_pos[b, :, 0]
            col_idx = self.user_pos[b, :, 1]
            coverage[b].index_put_((row_idx, col_idx), torch.ones(U, dtype=torch.int32, device=device), accumulate=True)

        capacity = self.antenna_grid * self.antenna_capacity
        served = torch.minimum(coverage, capacity)
        fails = self.total_users - served.sum(dim=(1,2))
        redirects = (coverage > capacity).sum(dim=(1,2))
        return fails, redirects

# GA Parameters
POP_SIZE = 60
GENERATIONS = 120
TOURNAMENT_K = 3
CX_PROB = 0.7
MUT_PROB = 0.05
MAX_PER_CELL = 2
EP_STEPS = 240
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

LAMBDA_FAILURE = 3.00
LAMBDA_REDIRECT = 0.10
LAMBDA_ANTENNA = 1.00

GRID_ROWS, GRID_COLS = 20, 20
GRID_SIZE = GRID_ROWS * GRID_COLS

# GA Functions
def random_population(pop_size: int) -> torch.Tensor:
    return torch.randint(0, MAX_PER_CELL + 1, (pop_size, GRID_SIZE), device=device)

def crossover_batch(p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand(p1.shape, device=device) < 0.5
    return torch.where(mask, p1, p2), torch.where(mask, p2, p1)

def mutate_batch(pop: torch.Tensor) -> torch.Tensor:
    mutation_mask = torch.rand(pop.shape, device=device) < MUT_PROB
    random_genes = torch.randint(0, MAX_PER_CELL + 1, pop.shape, device=device)
    return torch.where(mutation_mask, random_genes, pop)

def chromo_to_grid(ch: torch.Tensor) -> torch.Tensor:
    return ch.view(-1, GRID_ROWS, GRID_COLS)

def eval_layout_batch(pop: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grids = chromo_to_grid(pop)
    env = VectorizedCellularNetworkEnv(batch_size=grids.shape[0], rows=GRID_ROWS, cols=GRID_COLS)
    env.set_antenna_grids(grids)
    env.place_users()
    for _ in range(EP_STEPS):
        env.move_users_markov_chain()
    fails, reds = env.check_coverage()
    ant_cnt = grids.sum(dim=(1,2))
    fitness = LAMBDA_FAILURE * fails + LAMBDA_REDIRECT * reds + LAMBDA_ANTENNA * ant_cnt
    return fitness, fails, reds, ant_cnt

def tournament_selection(pop: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
    indices = torch.randint(0, pop.size(0), (TOURNAMENT_K,), device=device)
    selected = pop[indices]
    selected_fitness = fitness[indices]
    return selected[torch.argmin(selected_fitness)]

# GA Main Loop
pop = random_population(POP_SIZE)
best_ch, best_fit, best_tuple = None, float('inf'), None
start = time.time()

for gen in range(GENERATIONS):
    fitness, fails, reds, ant_cnts = eval_layout_batch(pop)
    gen_best_idx = torch.argmin(fitness).item()
    if fitness[gen_best_idx] < best_fit:
        best_fit = fitness[gen_best_idx].item()
        best_ch = pop[gen_best_idx].clone()
        best_tuple = (best_fit, fails[gen_best_idx].item(), reds[gen_best_idx].item(), ant_cnts[gen_best_idx].item())

    print(f"Gen {gen:03d} | bestF={fails[gen_best_idx]:4d} bestR={reds[gen_best_idx]:4d} A={ant_cnts[gen_best_idx]:3d} | "
        f"meanF={fails.float().mean():6.1f} meanR={reds.float().mean():6.1f} Aµ={ant_cnts.float().mean():4.1f}")

    new_pop = []
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(pop, fitness)
        p2 = tournament_selection(pop, fitness)
        c1, c2 = crossover_batch(p1.unsqueeze(0), p2.unsqueeze(0))
        new_pop.extend([mutate_batch(c1)[0], mutate_batch(c2)[0]])
    pop = torch.stack(new_pop[:POP_SIZE])

# Final Output
print("\n=========== GA Complete ===========")
print(f"Elapsed: {(time.time() - start)/60:.1f} min")
print(f"Best fitness   : {best_tuple[0]:.2f}")
print(f"Failures       : {best_tuple[1]}")
print(f"Redirects      : {best_tuple[2]}")
print(f"Antenna count  : {best_tuple[3]}")
print("Best layout grid (antenna counts):")
print(best_ch.view(GRID_ROWS, GRID_COLS).int().cpu().numpy())