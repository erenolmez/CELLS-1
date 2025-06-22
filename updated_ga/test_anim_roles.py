# test_anim_roles.py

from updated_grid_env import CellularNetworkEnv

# === Create attraction-mode env with default zones ===
env = CellularNetworkEnv(
    rows=20,
    cols=20,
    total_users=50000,
    movement_mode="attraction",
    time_step=60
)

# === Evaluate initial placement to populate grids ===
env.place_antennas()
env.place_users()

# === Run animations with roles ===
env.animate_user_grid(steps=10, interval=300, save_path="user_movement_roles.gif", show=True)
env.animate_coverage_quality(steps=10, interval=300, save_path="coverage_roles.gif", show=True)
