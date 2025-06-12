import numpy as np
from updated_grid_env import CellularNetworkEnv

# === Your GA-optimized best antenna layout ===
best_grid = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 2, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 1, 0, 3, 0, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 2, 3, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 0, 0, 1],
    [1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 1, 1, 0],
    [0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [2, 0, 0, 1, 3, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 2, 0, 0],
    [1, 0, 3, 0, 1, 1, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1, 0, 1, 2],
    [0, 1, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 0, 0],
    [3, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 3, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2],
    [0, 0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 0, 0, 0, 0, 2],
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 0]
], dtype=int)

# Create environment
env = CellularNetworkEnv(rows=20, cols=20, total_users=50000)
env.antenna_grid = best_grid
env.place_users()

# === Evaluate coverage stats over 1 year ===
env.evaluate_placement(steps=8760)

# === Animate user movement and coverage ===
# This uses the built-in method you mentioned
env.animate_coverage_quality(antenna_grid=env.antenna_grid, steps=240, interval=100, show=True, save=False)
