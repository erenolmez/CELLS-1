# %%
import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
# %%
class CellularNetworkEnv(gym.Env):
    """ Gym environment for optimizing antenna placement and handling failures. """
    
    def __init__(self):
        super(CellularNetworkEnv, self).__init__()
        
        # Grid settings
        self.rows = 6
        self.cols = 6
        # self.max_users = 3  # Max users per cell
        self.coverage_radius = 1  # Range of coverage for each antenna

        # State representation
        self.car_grid = np.zeros((self.rows, self.cols), dtype=int)  # Users per cell
        self.antenna_grid = np.zeros((self.rows, self.cols), dtype=int)  # 0 (No Antenna), 1 (Antenna)
        
        
        # Place antennas 
        self.total_users = 5000
        self.num_cells = self.rows * self.cols
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.rows),
            spaces.Discrete(self.cols),
            spaces.Discrete(2)  # 0 = add, 1 = remove
        ))
        self.num_antennas = self.total_users // self.num_cells
        # self.num_antennas = random.randint(1, self.rows * self.cols // 4)
        self.max_antennas = self.total_users // self.num_cells  # e.g., 14
        self.antenna_capacity = 30  # Max users an antenna can handle
        self.place_antennas()
        
        # Place users randomly
        self.place_users()
        
        # Define action space (Move antennas or optimize coverage)
        # self.action_space = spaces.Discrete(10)  # Placeholder for actions
        # self.action_space = spaces.Discrete(self.rows * self.cols)

        # Define observation space (Flattened grid states)
        # self.observation_space = spaces.Box(low=0, high=self.max_users, shape=(self.rows * self.cols * 2,), dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=5000, shape=(self.rows * self.cols * 2,), dtype=np.int32)

        self._seed_value = None
        self._compute_neighbor_map()
        # Reset environment
        # self.reset()
    
    def seed(self, seed=None):
        self._seed_value = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

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
        """Randomly distribute antennas across the grid (multiple per cell allowed)."""
        self.antenna_grid = np.zeros((self.rows, self.cols), dtype=int)

        for _ in range(self.num_antennas):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            self.antenna_grid[r, c] += 1

    def place_users(self):
        """ Randomly distribute 5000 users across the grid. """
        # total_users = 5000
        # flat_grid = np.random.multinomial(total_users, [1/(self.rows*self.cols)]*(self.rows*self.cols))
        flat_grid = np.random.multinomial(self.total_users, [1 / self.num_cells] * self.num_cells)
        self.car_grid = np.array(flat_grid).reshape((self.rows, self.cols))
        print(self.car_grid)
        
    def check_coverage(self):
        covered_grid = np.zeros((self.rows, self.cols), dtype=int)
        failures = 0
        redirects = 0
        antenna_load = np.zeros((self.rows, self.cols), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                users_in_cell = self.car_grid[r, c]
                remaining_users = users_in_cell

                for dr in range(-self.coverage_radius, self.coverage_radius + 1):
                    for dc in range(-self.coverage_radius, self.coverage_radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            max_capacity = self.antenna_grid[nr, nc] * self.antenna_capacity
                            free_capacity = max(0, max_capacity - antenna_load[nr, nc])

                            if free_capacity > 0 and remaining_users > 0:
                                served = min(free_capacity, remaining_users)
                                antenna_load[nr, nc] += served
                                remaining_users -= served

                                covered_grid[r, c] = 1
                                if (nr != r or nc != c):
                                    redirects += served

                        if remaining_users == 0:
                            break

                failures += remaining_users

        return covered_grid, failures, redirects

    def move_users_markov_chain(self, p_move=0.4):
        new_car_grid = np.zeros((self.rows, self.cols), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                users = self.car_grid[r, c]
                if users == 0:
                    continue

                neighbors = self.neighbor_map[(r, c)]
                n_neighbors = len(neighbors)
                n_move = int(p_move * users)
                n_stay = users - n_move

                if n_neighbors > 0:
                    move_distribution = np.random.multinomial(n_move, [1 / n_neighbors] * n_neighbors)
                    for i, count in enumerate(move_distribution):
                        nr, nc = neighbors[i]
                        new_car_grid[nr, nc] += count

                new_car_grid[r, c] += n_stay

        self.car_grid = new_car_grid

    def update_after_user_movement(self):
            """ Update coverage and failures after user movement. """
            self.check_coverage()  # Recalculate coverage status
            self.render()  # Display updated grids
    
    def reset(self):
        """ Reset environment with deterministic random state if seeded. """
        if self._seed_value is not None:
            np.random.seed(self._seed_value)
            random.seed(self._seed_value)
        
        self.place_antennas()
        self.place_users()
        return np.concatenate((self.car_grid.flatten(), self.antenna_grid.flatten()))

    def calculate_reward(self, failures, redirects):
        total_users = self.total_users
        total_antennas = np.sum(self.antenna_grid)

        failure_penalty = failures / total_users  # Normalize [0, 1]
        redirect_penalty = redirects / total_users
        antenna_cost = total_antennas / self.max_antennas  # Normalize [0, 1]

        reward = 1.0 - (2.0 * failure_penalty + 0.5 * redirect_penalty + 0.3 * antenna_cost)
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
        done = False

        obs = np.concatenate((self.car_grid.flatten(), self.antenna_grid.flatten()))
        return obs, reward, done, {}

    def render(self):
        """ Display the grid state with separate visuals for cars, antennas, and coverage. """
        self.covered_grid, failures, redirects = self.check_coverage()
        print("\nCar Grid (Users per cell):")
        print(self.car_grid)
        print("\nAntenna Grid (0: No Antenna, 1: Antenna):")
        print(self.antenna_grid)
        print("\nCoverage Grid (1: Covered, 0: Not Covered):")
        print(self.covered_grid)
        print(f"\nFailures: {failures}, Redirects: {redirects}")

    def render_heatmaps(self):
        """Visualize car, antenna, and coverage grids as separate annotated heatmaps."""
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

         #plot_single_heatmap(self.car_grid, title="Car Grid (Users per Cell)", cmap="YlGnBu")
         #plot_single_heatmap(self.car_grid, title="Car Grid (Users per Cell)", cmap="Reds")
         #plot_single_heatmap(self.car_grid, title="Car Grid (Users per Cell)", cmap="OrRd")
         #plot_single_heatmap(self.car_grid, title="Car Grid (Users per Cell)", cmap="PuBu")
        plot_single_heatmap(self.car_grid, title="Car Grid (Users per Cell)", cmap="YlOrRd")
        # plot_single_heatmap(self.car_grid, title="Car Grid (Users per Cell)", cmap="YlOrBr")
    
        # plot_single_heatmap(self.antenna_grid, title="Antenna Grid (0 or 1)", cmap="viridis")
        # plot_single_heatmap(self.covered_grid, title="Coverage Grid (0 = Uncovered, 1 = Covered)", cmap="Greens")

# %%
# # Test the environment
# env = CellularNetworkEnv()
# env.render()
# env.render_heatmaps()
# print("Total users before:", np.sum(env.car_grid))


# # %%

# # Move users using Markov Chain
# env.move_users_markov_chain()
# print("\nAfter User Movement (Markov Chain):")
#  #env.render()
# env.render_heatmaps()
# print("Total users after: ", np.sum(env.car_grid))
# # # %%

# %%
