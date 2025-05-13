#%%
import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
#%%
class IndividualUserEnv(gym.Env):
    def __init__(self, grid_size=6, num_users=5000, seed=None):
        super(IndividualUserEnv, self).__init__()

        self.rows = grid_size
        self.cols = grid_size
        self.num_users = num_users
        self.seed_value = seed

        self.action_space = spaces.Discrete(1)  # No real actions, just stepping movement
        self.observation_space = spaces.Box(
            low=0, high=max(self.rows, self.cols),
            shape=(self.num_users, 2), dtype=np.int32
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self):
        """Randomly place individual users in grid cells."""
        self.users = [(random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)) for _ in range(self.num_users)]
        return np.array(self.users, dtype=np.int32)

    def move_users(self, p_move=0.4):
        """Move individual users based on Markov-style neighbor movement."""
        new_positions = []
        for r, c in self.users:
            if np.random.rand() < p_move:
                neighbors = [(r + dr, c + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]
                valid = [(nr, nc) for nr, nc in neighbors if 0 <= nr < self.rows and 0 <= nc < self.cols]
                if valid:
                    new_r, new_c = random.choice(valid)
                    new_positions.append((new_r, new_c))
                else:
                    new_positions.append((r, c))
            else:
                new_positions.append((r, c))
        self.users = new_positions

    def step(self, action=None):
        """Simulate one time step: just user movement."""
        self.move_users()
        obs = np.array(self.users, dtype=np.int32)
        reward = 0  # No optimization or coverage goal
        done = False
        return obs, reward, done, {}

    def render(self):
        """Render heatmap of user distribution on the grid."""
        grid = np.zeros((self.rows, self.cols), dtype=int)
        for r, c in self.users:
            grid[r, c] += 1

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=max(1, grid.max()))
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j, i, grid[i, j], ha="center", va="center",
                        color="white" if grid[i, j] > grid.max() / 2 else "black")
        ax.set_title("User Density Heatmap")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

#%%
env = IndividualUserEnv(seed=42)
print("Initial user positions:")
print(env.users)  # Prints full list of (row, col) coordinates

env.render()

for step in range(3):
    obs, reward, done, _ = env.step()
    print(f"\nStep {step + 1} user positions:")
    print(obs)  # Same as env.users after the step
    env.render()

# %%
