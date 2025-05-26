import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CellularNetworkEnv:
    # ────────────────────────────
    # INITIALISATION
    # ────────────────────────────
    def __init__(self,
                 rows=6, cols=6,
                 total_users=5000, antenna_capacity=300,
                 time_step=60):
        self.rows, self.cols = rows, cols
        self.total_users = total_users
        self.antenna_capacity = antenna_capacity
        self.time_step = time_step            # minutes/step
        self.sim_time_hours = 0

        self.coverage_radius = 1
        self.num_cells = rows * cols
        self.num_antennas = total_users // antenna_capacity

        self.car_grid = np.zeros((rows, cols), dtype=int)
        self.antenna_grid = np.zeros((rows, cols), dtype=int)

        self._compute_neighbor_map()
        self.place_users()
        self.place_antennas()                 # greedy after warm-up

    # ────────────────────────────
    # HELPER MAPS
    # ────────────────────────────
    def _compute_neighbor_map(self):
        self.neighbor_map = {
            (r, c): [(r + dr, c + dc)
                     for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                     if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]
            for r in range(self.rows) for c in range(self.cols)
        }

    def compute_dynamic_p_move(self):
        return 0.4 * (self.time_step / 60)     # simple linear rule

    def get_sim_time(self):
        d, h = divmod(self.sim_time_hours, 24)
        return f"Day {d}, {h:02d}:00"

    # ────────────────────────────
    # USERS & MOVEMENT
    # ────────────────────────────
    def place_users(self):
        flat = np.random.multinomial(self.total_users,
                                     [1 / self.num_cells] * self.num_cells)
        self.car_grid = flat.reshape(self.rows, self.cols)

    # ▼▼▼ ——— ONLY THIS FUNCTION HAS CHANGED ——— ▼▼▼
    def move_users_markov_chain(self):
        """
        Markov movement with fixed per-direction probability:
            – There are 8 possible move directions (N, NE, E, …).
            – Each valid direction gets p_move / 8.
            – Invalid directions (edge/corner) add their share back to the 'stay' prob.
        """
        p_move = self.compute_dynamic_p_move()
        new_grid = np.zeros_like(self.car_grid)

        for r in range(self.rows):
            for c in range(self.cols):
                users = self.car_grid[r, c]
                if users == 0:
                    continue

                neighbors = self.neighbor_map[(r, c)]
                k = len(neighbors)               # valid moves (≤ 8)
                missing = 8 - k                  # off-grid directions
                stay_prob = 1.0 - p_move + missing * (p_move / 8)

                # multinomial: [stay] + one bucket per valid neighbor
                probs = [stay_prob] + [p_move / 8] * k
                moves = np.random.multinomial(users, probs)

                # add stayers
                new_grid[r, c] += moves[0]
                # add movers
                for idx, cnt in enumerate(moves[1:]):
                    nr, nc = neighbors[idx]
                    new_grid[nr, nc] += cnt

        self.car_grid = new_grid
    # ▲▲▲ ——— MOVEMENT MODEL ENDS HERE ——— ▲▲▲


    # ────────────────────────────
    # ANTENNAS (GREEDY)
    # ────────────────────────────
    def place_antennas(self, warmup_hours=6):
        print("\n⏳ Warm-up simulation for greedy placement …")
        density = np.zeros_like(self.car_grid, dtype=float)
        for _ in range(warmup_hours):
            self.move_users_markov_chain()
            density += self.car_grid

        top_idx = np.argsort(density.ravel())[::-1][:self.num_antennas]
        self.antenna_grid.fill(0)
        for idx in top_idx:
            r, c = divmod(idx, self.cols)
            self.antenna_grid[r, c] = 1

        print("📡 Antenna placement complete:")
        for idx in top_idx:
            r, c = divmod(idx, self.cols)
            print(f"  • antenna at ({r}, {c})")

    # ────────────────────────────
    # COVERAGE METRICS  (unchanged)
    # ────────────────────────────
    def calculate_stats(self):
        R = self.coverage_radius
        covered_mask = np.zeros_like(self.car_grid, dtype=bool)
        load = np.zeros_like(self.antenna_grid, dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                if self.antenna_grid[r, c]:
                    for dr in range(-R, R + 1):
                        for dc in range(-R, R + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                covered_mask[nr, nc] = True

        covered_users = failures = 0

        for r in range(self.rows):
            for c in range(self.cols):
                users = self.car_grid[r, c]
                if users == 0:
                    continue

                if not covered_mask[r, c]:
                    failures += users
                    continue

                attached = False
                for dr in range(-R, R + 1):
                    for dc in range(-R, R + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.antenna_grid[nr, nc]:
                            if load[nr, nc] + users <= self.antenna_capacity:
                                load[nr, nc] += users
                                covered_users += users
                                attached = True
                                break
                    if attached:
                        break
                if not attached:
                    failures += users

        cov_pct = 100 * covered_users / self.total_users
        cost = failures + self.num_antennas * 5
        return cov_pct, failures, cost

    # ────────────────────────────
    # VISUALISATION  (unchanged)
    # ────────────────────────────
    def visualize(self, hours=24, interval=500):
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        heat = ax.imshow(self.car_grid, cmap="YlOrRd",
                         vmin=0, vmax=self.car_grid.max())
        fig.colorbar(heat, cax=cax).set_label("Users per Cell")

        # ---------- antenna markers ----------
        ay, axx = np.where(self.antenna_grid == 1)
        antenna_scatter = ax.scatter(axx, ay, marker='^', s=120,
                                     color='blue', edgecolor='white',
                                     linewidth=0.7, zorder=3, label='Antenna')
        # --------------------------------------

        texts = [ax.text(j, i, f"{self.car_grid[i, j]}",
                         ha="center", va="center", fontsize=8)
                 for i in range(self.rows) for j in range(self.cols)]

        title = ax.set_title("Live Movement – Hour 0")

        def update(_):
            self.move_users_markov_chain()
            self.sim_time_hours += 1

            cov_pct, fails, cost = self.calculate_stats()
            print(f"H{self.sim_time_hours:3d} | "
                  f"Cov {cov_pct:5.2f}% | "
                  f"Fail {fails:5d} | "
                  f"Ant {self.num_antennas:2d} | "
                  f"Cost {cost:6.0f}")

            heat.set_data(self.car_grid)
            title.set_text(f"Live Movement – {self.get_sim_time()}")

            for i in range(self.rows):
                for j in range(self.cols):
                    texts[i * self.cols + j].set_text(str(self.car_grid[i, j]))

            return [heat] + texts + [antenna_scatter]

        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        plt.tight_layout()

        self.anim = FuncAnimation(fig, update, frames=hours,
                                  interval=interval, blit=False, repeat=False)
        plt.show()


# ────────────────────────────
# MAIN
# ────────────────────────────
if __name__ == "__main__":
    env = CellularNetworkEnv()
    env.visualize(hours=720, interval=300)

    
