import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CellularNetworkEnv:
    def __init__(self, rows=6, cols=6, total_users=5000, antenna_capacity=300, time_step=60):
        self.time_step = time_step
        self.sim_time_hours = 0
        self.rows = rows
        self.cols = cols
        self.coverage_radius = 1
        self.total_users = total_users
        self.antenna_capacity = antenna_capacity
        self.num_cells = self.rows * self.cols
        self.num_antennas = self.total_users // self.antenna_capacity
        self.max_antennas = self.num_antennas

        self.car_grid = np.zeros((self.rows, self.cols), dtype=int)
        self.antenna_grid = np.zeros((self.rows, self.cols), dtype=int)

        self._seed_value = None
        self._compute_neighbor_map()
        self.place_antennas()
        self.place_users()

    def compute_dynamic_p_move(self):
        return 0.4 * (self.time_step / 60)

    def get_sim_time(self):
        day = self.sim_time_hours // 24
        hour = self.sim_time_hours % 24
        return f"Day {day}, {hour:02d}:00"

    def _compute_neighbor_map(self):
        self.neighbor_map = {}
        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            neighbors.append((nr, nc))
                self.neighbor_map[(r, c)] = neighbors

    def place_antennas(self):
        self.antenna_grid = np.zeros((self.rows, self.cols), dtype=int)
        for _ in range(self.num_antennas):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            self.antenna_grid[r, c] += 1

    def place_users(self):
        flat_grid = np.random.multinomial(self.total_users, [1 / self.num_cells] * self.num_cells)
        self.car_grid = np.array(flat_grid).reshape((self.rows, self.cols))

    def move_users_markov_chain(self):
        p_move = self.compute_dynamic_p_move()
        new_car_grid = np.zeros((self.rows, self.cols), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                users = self.car_grid[r, c]
                if users == 0:
                    continue

                neighbors = self.neighbor_map[(r, c)]
                weights = [1 / (1 + self.car_grid[nr, nc]) for (nr, nc) in neighbors]
                total_weight = sum(weights)
                probs = [1.0 - p_move] + [(p_move * w / total_weight) for w in weights]

                move_counts = np.random.multinomial(users, probs)

                new_car_grid[r, c] += move_counts[0]
                for idx, cnt in enumerate(move_counts[1:]):
                    nr, nc = neighbors[idx]
                    new_car_grid[nr, nc] += cnt

        self.car_grid = new_car_grid

    def visualize_movement_heatmap(self, hours=24, interval=500):
        fig, ax = plt.subplots(figsize=(6, 6))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        heatmap = ax.imshow(self.car_grid, cmap="YlOrRd", vmin=0, vmax=self.car_grid.max())
        cbar = fig.colorbar(heatmap, cax=cax)
        cbar.set_label("Users per Cell")

        text_annotations = []
        for i in range(self.rows):
            for j in range(self.cols):
                text = ax.text(j, i, str(self.car_grid[i, j]), ha="center", va="center", fontsize=8)
                text_annotations.append(text)

        title = ax.set_title("Live Movement - Hour 0")

        def update(frame):
            self.move_users_markov_chain()
            self.sim_time_hours += 1

            total_users = np.sum(self.car_grid)
            assert total_users == self.total_users, f"User mismatch: {total_users}"
            print(f"Hour {self.sim_time_hours}: Total users = {total_users}")

            heatmap.set_data(self.car_grid)
            title.set_text(f"Live Movement - {self.get_sim_time()}")

            for i in range(self.rows):
                for j in range(self.cols):
                    idx = i * self.cols + j
                    text_annotations[idx].set_text(str(self.car_grid[i, j]))

            return [heatmap] + text_annotations

        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        plt.tight_layout()
        anim = FuncAnimation(fig, update, frames=hours, interval=interval, blit=False, repeat=False)
        plt.show()

# Run the simulation
if __name__ == "__main__":
    env = CellularNetworkEnv()
    env.visualize_movement_heatmap(hours=24, interval=500)