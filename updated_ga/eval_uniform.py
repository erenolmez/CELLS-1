import numpy as np
from updated_grid_env import CellularNetworkEnv

# Load best antenna layout from GA
antenna_layout = np.load("best_uniform.npy")

# Set up 20×20 grid with 50,000 users, uniform random movement
env = CellularNetworkEnv(
    rows=20,
    cols=20,
    total_users=50000,
    movement_mode="uniform"
)

# Inject saved best antenna layout
env.antenna_grid = antenna_layout.copy()

# Run simulation for 1 year (8760 steps = hours)
steps = 8760
total_failures = 0
total_redirects = 0

for _ in range(steps):
    env.move_users_markov_chain()
    _, fails, reds = env.check_coverage()
    total_failures += fails
    total_redirects += reds
    env.sim_time_hours += 1

# Compute final stats
userwise_coverage = 100 * (1 - total_failures / (env.total_users * steps))

print("\n📊 Uniform-movement Evaluation:")
print(f"  Antennas deployed      : {int(env.antenna_grid.sum())}")
print(f"  Total Failures         : {total_failures}")
print(f"  Total Redirects        : {total_redirects}")
print(f"  Userwise Coverage Rate : {userwise_coverage:.2f}%")
