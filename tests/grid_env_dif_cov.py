# %%
import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
# %%
class CellularNetworkEnv(gym.Env):
    """ Gym environment for optimizing antenna placement and handling failures. """

    def __init__(self, rows=6, cols=6, total_users=5000, antenna_capacity=300, time_step=60, movement_mode="markov"):
        super(CellularNetworkEnv, self).__init__()
        self.movement_mode = movement_mode  # "markov" or "attraction"
        self.time_step = time_step  # real-time duration per step (in minutes)
        self.sim_time_hours = 0

        # Grid settings
        self.rows = rows
        self.cols = cols
        self.coverage_radius = 1  # Range of coverage for each antenna

        # State representation
        self.user_grid = np.zeros((self.rows, self.cols), dtype=int)  # Users per cell
        self.antenna_grid = np.zeros((self.rows, self.cols), dtype=int)  # 0 (No Antenna), 1 (Antenna)

        # Place antennas 
        self.total_users = total_users
        self.num_cells = self.rows * self.cols

        self.num_students = self.total_users // 2
        self.num_workers = self.total_users - self.num_students

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.rows),
            spaces.Discrete(self.cols),
            spaces.Discrete(2)  # 0 = add, 1 = remove
        ))

        self.antenna_capacity = antenna_capacity
        min_antenna_by_capacity = self.total_users / self.antenna_capacity
        min_antenna_by_coverage = self.num_cells / ((2 * self.coverage_radius + 1) ** 2)
        self.num_antennas = int(min_antenna_by_capacity)
        self.max_antennas = int(np.ceil(max(min_antenna_by_capacity, min_antenna_by_coverage) * 1.3))

        self.place_antennas()

        # Place users randomly
        self.place_users()

        self.observation_space = spaces.Box(low=0, high=5000, shape=(self.rows * self.cols * 2,), dtype=np.int32)

        self._seed_value = None
        self._compute_neighbor_map()

        self.zones = {
        "home":  [(17, 5), (18, 3), (16, 7)],
        "school": [(2, 16), (3, 14)],
        "work": [(2, 3), (3, 5), (1, 4)],
        "city_center": [(9, 9), (10, 10), (11, 11)]  # <-- new zone
        }
    
    def seed(self, seed=None):
        self._seed_value = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
      
    def compute_dynamic_p_move(self):
        """
        Movement probability scaled to match the real-time length of each step.
        Assumes 0.4 move probability per 60-minute step.
        """
        baseline_p_move = 0.4
        hour = self.sim_time_hours % 24
        if 23 <= hour or hour < 6:
            return baseline_p_move * 0.1  # reduced night movement
        return baseline_p_move * (self.time_step / 60)

    def get_sim_time(self):
        day = self.sim_time_hours // 24
        hour = self.sim_time_hours % 24
        return f"Day {day}, {hour:02d}:00"

    def get_active_zones(self, hour, user_type):
        day = self.sim_time_hours // 24
        is_weekend = (day % 7) in [5, 6]  # Saturday = 5, Sunday = 6
        is_saturday = (day % 7 == 5)

        active = []

        if user_type == 1:  # student
            if not is_weekend and 8 <= hour < 15:
                active += self.zones["school"]

        elif user_type == 2:  # worker
            if is_saturday and 10 <= hour < 14:
                active += self.zones["work"]
            elif not is_weekend and 9 <= hour < 18:
                active += self.zones["work"]

        # Weekend city center (for all types)
        if is_weekend and 11 <= hour < 20:
            active += self.zones["city_center"]

        # Evening pull toward home (stronger)
        if 18 <= hour < 23:
            active += self.zones["home"] * 2  # duplicate to boost home pull

        # Home always weakly attractive
        active += self.zones["home"]

        return active

    def attraction_weights(self, hour, neighbors, user_type):
        zones = self.get_active_zones(hour, user_type)
        weights = []
        for nr, nc in neighbors:
            dists = [abs(nr - zr) + abs(nc - zc) for (zr, zc) in zones]
            min_dist = min(dists)
            weights.append(1.0 / (1 + min_dist))
        weights = np.array(weights)
        return weights / weights.sum()
        
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

    def print_neighbor_grid(self):
        """
        Prints a grid where each cell contains the number of neighbors for that cell.
        Useful for visually verifying the neighbor map.
        """
        neighbor_grid = np.zeros((self.rows, self.cols), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                neighbor_grid[r, c] = len(self.neighbor_map[(r, c)])

        print("\nNeighbor Grid (Number of Neighbors per Cell):")
        print(neighbor_grid)

    def place_antennas(self):
        """Randomly distribute antennas across the grid (multiple per cell allowed)."""
        self.antenna_grid = np.zeros((self.rows, self.cols), dtype=int)

        for _ in range(self.num_antennas):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            self.antenna_grid[r, c] += 1

    # def place_users(self):
    #     """ Randomly distribute 5000 users across the grid. """
    #     # total_users = 5000
    #     # flat_grid = np.random.multinomial(total_users, [1/(self.rows*self.cols)]*(self.rows*self.cols))
    #     flat_grid = np.random.multinomial(self.total_users, [1 / self.num_cells] * self.num_cells)
    #     self.user_grid = np.array(flat_grid).reshape((self.rows, self.cols))
    #     # print(self.user_grid)

    # def place_users(self):
    #     # Randomly assign users to cells
    #     flat_student = np.random.multinomial(self.num_students, [1 / self.num_cells] * self.num_cells)
    #     flat_worker  = np.random.multinomial(self.num_workers,  [1 / self.num_cells] * self.num_cells)

    #     self.user_grid_student = np.array(flat_student).reshape((self.rows, self.cols))
    #     self.user_grid_worker  = np.array(flat_worker).reshape((self.rows, self.cols))
    #     self.user_grid = self.user_grid_student + self.user_grid_worker  # total user view

    def place_users(self):
        """Distribute users across the grid. Split into student/worker only if attraction-based movement is used."""
        flat = np.random.multinomial(self.total_users, [1 / self.num_cells] * self.num_cells)
        full_user_grid = np.array(flat).reshape((self.rows, self.cols))

        if self.movement_mode == "attraction":
            flat_stud = np.random.multinomial(self.num_students, [1 / self.num_cells] * self.num_cells)
            flat_work = np.random.multinomial(self.num_workers,  [1 / self.num_cells] * self.num_cells)

            self.user_grid_student = np.array(flat_stud).reshape((self.rows, self.cols))
            self.user_grid_worker  = np.array(flat_work).reshape((self.rows, self.cols))
            self.user_grid = self.user_grid_student + self.user_grid_worker

        else:
            self.user_grid = full_user_grid
            self.user_grid_student = None
            self.user_grid_worker = None

    def check_coverage(self): 
        """
        Returns covered_grid (0‑100 %), failures, redirects.
        Redirects count only when a user is forced to skip one or more
        *saturated* antennas (capacity = 0) before finding free capacity.
        Distributes users more evenly to antennas in passes.
        """
        covered_grid = np.zeros((self.rows, self.cols), dtype=int)
        failures = 0
        redirects = 0
        antenna_load = np.zeros((self.rows, self.cols), dtype=int)

        # Pre‑compute neighbour list ordered by Manhattan distance (local first)
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(self.rows):
            for c in range(self.cols):
                users_in_cell = self.user_grid[r, c]
                remaining_users = users_in_cell
                users_served = 0
                saturated_seen = False  # ← flag for redirect counting

                # For more homogeneous use: repeat over antenna list in passes
                max_passes = len(offsets)
                pass_index = 0

                # Split users equally over passes
                while remaining_users > 0 and pass_index < max_passes:
                    dr, dc = offsets[pass_index]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:

                        cap_here = self.antenna_grid[nr, nc] * self.antenna_capacity
                        free_here = max(0, cap_here - antenna_load[nr, nc])

                        if free_here == 0:
                            if cap_here > 0:
                                saturated_seen = True
                        else:
                            # Serve a fair share of users in this round
                            fair_share = int(np.ceil(users_in_cell / len(offsets)))
                            served = min(fair_share, remaining_users, free_here)
                            antenna_load[nr, nc] += served
                            remaining_users -= served
                            users_served += served
                            if saturated_seen:
                                redirects += served

                    pass_index += 1

                failures += remaining_users

                if users_in_cell > 0:
                    covered_grid[r, c] = int(100 * users_served / users_in_cell)

        return covered_grid, failures, redirects

    def move_users_markov_chain(self):
        if self.movement_mode == "markov":
            self._move_users_uniform_markov()
        elif self.movement_mode == "attraction":
            self._move_users_biased_attraction()
        else:
            raise ValueError(f"Unknown movement_mode: {self.movement_mode}")
        
    def _move_users_uniform_markov(self):
        """
        Markov movement using *one* multinomial draw per cell:
            category 0  – stay          (prob = 1 − p_move)
            categories 1…k – neighbours (prob = p_move / k each)
        For edge/corner cells, invalid directions contribute to stay probability.
        """

        p_move = self.compute_dynamic_p_move()
        new_user_grid = np.zeros((self.rows, self.cols), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                users = self.user_grid[r, c]
                if users == 0:
                    continue

                neighbors = self.neighbor_map[(r, c)]
                k = len(neighbors)
                missing = 8 - k
                stay_prob = 1.0 - p_move + (missing * (p_move / 8))
                move_probs = [stay_prob] + [p_move / 8] * k

                move_counts = np.random.multinomial(users, move_probs)

                # 1) stayers
                new_user_grid[r, c] += move_counts[0]

                # 2) movers to each neighbour
                for idx, count in enumerate(move_counts[1:]):
                    nr, nc = neighbors[idx]
                    new_user_grid[nr, nc] += count

        self.user_grid = new_user_grid   

    def _move_users_biased_attraction(self):
        p_move = self.compute_dynamic_p_move()
        new_student = np.zeros_like(self.user_grid_student)
        new_worker  = np.zeros_like(self.user_grid_worker)
        hour = self.sim_time_hours % 24

        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = self.neighbor_map[(r, c)]

                # --- Students ---
                n_stud = self.user_grid_student[r, c]
                if n_stud > 0:
                    weights = self.attraction_weights(hour, neighbors, user_type=1)
                    stay_prob = 1.0 - p_move
                    move_probs = [stay_prob] + list(p_move * weights)
                    move_counts = np.random.multinomial(n_stud, move_probs)

                    new_student[r, c] += move_counts[0]
                    for idx, count in enumerate(move_counts[1:]):
                        nr, nc = neighbors[idx]
                        new_student[nr, nc] += count

                # --- Workers ---
                n_work = self.user_grid_worker[r, c]
                if n_work > 0:
                    weights = self.attraction_weights(hour, neighbors, user_type=2)
                    stay_prob = 1.0 - p_move
                    move_probs = [stay_prob] + list(p_move * weights)
                    move_counts = np.random.multinomial(n_work, move_probs)

                    new_worker[r, c] += move_counts[0]
                    for idx, count in enumerate(move_counts[1:]):
                        nr, nc = neighbors[idx]
                        new_worker[nr, nc] += count

        self.user_grid_student = new_student
        self.user_grid_worker = new_worker
        self.user_grid = new_student + new_worker

    def update_after_user_movement(self):
            """ Update coverage and failures after user movement. """
            self.check_coverage()  # Recalculate coverage status
            self.render()  # Display updated grids
    
    def reset(self):
        """ Reset environment with deterministic random state if seeded. """
        if self._seed_value is not None:
            np.random.seed(self._seed_value)
            random.seed(self._seed_value)
        
        self.sim_time_hours = 0  # <-- Add this line to reset time
        self.place_antennas()
        self.place_users()
        return np.stack([self.user_grid, self.antenna_grid], axis=-1).reshape(-1)

    def calculate_reward(self, failures, redirects):
        total_users = self.total_users
        total_antennas = np.sum(self.antenna_grid)

        failure_penalty = failures / total_users  # Normalize [0, 1]
        redirect_penalty = redirects / total_users # Normalize [0, 1]
        antenna_cost = total_antennas / self.max_antennas  # Normalize [0, 1]

        reward = 1.0 - (2.5 * failure_penalty + 0.05 * redirect_penalty + 1.0 * antenna_cost)
        return reward

    def step(self, action):
        r, c, op = action

        if op == 0:
            if np.sum(self.antenna_grid) < self.max_antennas:
                self.antenna_grid[r, c] += 1
        elif op == 1:
            if self.antenna_grid[r, c] > 0:
                self.antenna_grid[r, c] -= 1

        # Simulate user movement
        self.move_users_markov_chain()

        # Evaluate coverage
        covered_grid, failures, redirects = self.check_coverage()
        reward = self.calculate_reward(failures, redirects)

        # ⏱️ 
        self.sim_time_hours += 1  # advance simulation time by 1 hour

        done = False

        obs = np.stack([self.user_grid, self.antenna_grid], axis=-1).reshape(-1)
        return obs, reward, done, {}

    def evaluate_placement(self, steps: int = 240):
        """
        Run the simulation for N steps and print:
        • Total + average failures
        • Total + average redirects
        • Userwise coverage rate (% of all users served)
        • Total antenna cost (antennas × steps)
        """
        total_failures = 0
        total_redirects = 0
        reward_accum = 0.0

        ant_count = int(np.sum(self.antenna_grid)) 

        for _ in range(steps):
            self.move_users_markov_chain()
            _, failures, redirects = self.check_coverage()
            total_failures += failures
            total_redirects += redirects
            reward_accum += self.calculate_reward(failures, redirects)

        avg_fail = total_failures / steps
        avg_red = total_redirects / steps
        avg_reward = reward_accum / steps
        episode_cost = 1.0 - avg_reward
        userwise_coverage = 100 * (1 - total_failures / (self.total_users * steps))

        print(f"\n📊 Evaluation over {steps} steps:")
        print(f"  Antennas deployed      : {ant_count}")
        print(f"  Total Failures         : {total_failures}")
        print(f"  Total Redirects        : {total_redirects}")
        print(f"  Avg Failures per step  : {avg_fail:.2f}")
        print(f"  Avg Redirects per step : {avg_red:.2f}")
        print(f"  Userwise Coverage Rate : {userwise_coverage:.2f}%")
        print(f"  Episode Cost (1-R)     : {episode_cost:.4f}")

    def render(self):
        """ Display the grid state with separate visuals for users, antennas, and coverage. """
        self.covered_grid, failures, redirects = self.check_coverage()
        print("\nUser Grid (Users per cell):")
        print(self.user_grid)
        print("\nAntenna Grid (Values = Antenna count per cell):")
        print(self.antenna_grid)
        print("\nCoverage Grid (1: Covered, 0: Not Covered):")
        print(self.covered_grid)
        print(f"\nFailures: {failures}, Redirects: {redirects}")

    def render_heatmaps(self):
        """Visualize user, antenna, and coverage grids as separate annotated heatmaps."""
        self.covered_grid, failures, redirects = self.check_coverage()

        def plot_single_heatmap(grid, title="Heatmap", cmap="plasma"):
            fig, ax = plt.subplots(figsize=(5, 5))
            heatmap = ax.imshow(grid, cmap=cmap, vmin=0, vmax=grid.max())
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    ax.text(j, i, grid[i, j], ha="center", va="center",
                            color="white" if grid[i, j] > grid.max() / 2 else "black")
            ax.set_title(title)
            fig.colorbar(heatmap, ax=ax)
            plt.tight_layout()
            plt.show()

        plot_single_heatmap(self.user_grid, title="User Grid (Users per Cell)", cmap="YlOrRd") 
        # plot_single_heatmap(self.antenna_grid, title="Antenna Grid (0 or 1)", cmap="viridis")
        # plot_single_heatmap(self.covered_grid, title="Coverage Grid (0 = Uncovered, 1 = Covered)", cmap="Greens")

    def animate_user_grid(self, steps=20, interval=500, save_path="user_movement.gif", show=True, save=True):
        fig, ax = plt.subplots(figsize=(6, 6))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        heatmap = ax.imshow(self.user_grid, cmap="YlOrRd", vmin=0, vmax=self.user_grid.max())
        fig.colorbar(heatmap, cax=cax).set_label("Users per Cell")
        title = ax.set_title("User Distribution (Step 0)")

        text_annotations = [
            ax.text(j, i, str(self.user_grid[i, j]), ha="center", va="center", fontsize=8)
            for i in range(self.rows) for j in range(self.cols)
        ]

        circles = []
        for i in range(self.rows):
            for j in range(self.cols):
                circle = Circle((j + 0.35, i - 0.35), radius=0.08, facecolor='gray', edgecolor='black')
                ax.add_patch(circle)
                circles.append(circle)

        def update(frame):
            self.move_users_markov_chain()
            covered_grid, _, _ = self.check_coverage()
            heatmap.set_data(self.user_grid)

            title.set_text(f"User Distribution ({self.get_sim_time()})")
            self.sim_time_hours += 1

            for i in range(self.rows):
                for j in range(self.cols):
                    idx = i * self.cols + j
                    text_annotations[idx].set_text(str(self.user_grid[i, j]))

            for idx, circle in enumerate(circles):
                r, c = divmod(idx, self.cols)
                coverage = covered_grid[r, c]
                color = 'green' if coverage == 100 else 'red' if coverage == 0 else 'orange'
                circle.set_facecolor(color)

            return [heatmap] + text_annotations + circles

        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False, repeat=False)
        if save:
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def animate_user_histogram(self, steps=240, interval=200, save_path="user_histogram.gif", show=True, save=True):
        fig, ax = plt.subplots(figsize=(10, 5))

        user_history = []
        for _ in range(steps):
            self.move_users_markov_chain()
            user_history.append(self.user_grid.flatten().copy())

        user_data = np.array(user_history)
        num_cells = self.rows * self.cols

        bars = ax.bar(np.arange(num_cells), user_data[0], color='skyblue')
        title = ax.set_title("User Distribution Histogram - Step 0")
        ax.set_ylim(0, np.max(user_data) + 50)
        ax.set_xlabel(f"Grid Cell Index (0 to {num_cells - 1})")
        ax.set_ylabel("Users per Cell")

        def update(frame):
            for bar, height in zip(bars, user_data[frame]):
                bar.set_height(height)
            title.set_text(f"User Distribution Histogram - Step {frame}")
            return bars

        anim = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False, repeat=False)
        if save:
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def animate_coverage_quality(self, steps=20, interval=500, save_path="coverage_quality.gif", show=True, save=True, antenna_grid=None):
        """Animates per-cell coverage and antenna placements with user coverage percentage."""
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
            new_cmap = LinearSegmentedColormap.from_list(
                f"{cmap.name}_trunc", cmap(np.linspace(minval, maxval, n))
            )
            return new_cmap
    
        if antenna_grid is not None:
            self.antenna_grid = np.array(antenna_grid)

        self.place_users()

        fig, ax = plt.subplots(figsize=(6, 6))
        base_cmap = colormaps["RdYlGn"]
        cmap = truncate_colormap(base_cmap, minval=0.0, maxval=0.9)  # drop darkest green

        coverage_grid, failures, _ = self.check_coverage()
        heatmap = ax.imshow(coverage_grid, cmap=cmap, vmin=0, vmax=100)
        title = ax.set_title("Coverage (Step 0)")
        fig.colorbar(heatmap, ax=ax, label="Coverage %")

        antenna_coords = np.argwhere(self.antenna_grid > 0)
        ant_x = antenna_coords[:, 1]
        ant_y = antenna_coords[:, 0]
        ant_plot = ax.scatter(ant_x, ant_y, c='blue', marker='^', s=80, label="Antennas")
    
        def update(frame):
            self.move_users_markov_chain()
            coverage_grid, failures, _ = self.check_coverage()
            heatmap.set_data(coverage_grid)

            covered_users = self.total_users - failures
            coverage_pct = 100 * covered_users / self.total_users

            title.set_text(f"Coverage ({self.get_sim_time()}) – {coverage_pct:.1f}% users covered")
            self.sim_time_hours += 1

            return [heatmap, title, ant_plot]

        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False, repeat=False)
        if save:
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        if show:
            plt.show()
        else:
            plt.close(fig)