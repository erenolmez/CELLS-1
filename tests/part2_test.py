import numpy as np
from updated_grid_env import CellularNetworkEnv

# Load antenna layout
antenna_layout = np.load("best_antenna_grid.npy")

# Set up 20×20 grid with 50,000 users using attraction-based movement
env = CellularNetworkEnv(rows=20, cols=20, total_users=50000, movement_mode="uniform")

# Inject saved best antenna layout
env.antenna_grid = antenna_layout.copy()

# Simulate 30 days (720 hours)
env.animate_user_grid(
    steps=8760,
    interval=100,  # reduce interval to speed up playback
    save_path="user_movement_month.gif",
    show=True,
    save=False
)

env.animate_coverage_quality(
    steps=720,
    interval=100,
    save_path="coverage_month.gif",
    show=True,
    save=False
)
