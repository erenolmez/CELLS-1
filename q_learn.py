# %%
import numpy as np
import random
from collections import defaultdict
from grid_env import CellularNetworkEnv


# === State Encoding ===
def downsample(grid):
    return grid.reshape(3, 2, 3, 2).sum(axis=(1, 3))

def get_state(env):
    """
    Downsamples car_grid and antenna_grid into (3x3) blocks,
    flattens and returns them as a tuple for Q-table indexing.
    """
    car_ds = downsample(env.car_grid)
    antenna_ds = downsample(env.antenna_grid)
    return tuple(car_ds.flatten()) + tuple(antenna_ds.flatten())


# === Initialize Environment and Q-Table ===
env = CellularNetworkEnv()
q_table = defaultdict(float)  # Q-values keyed by (state, action)
possible_actions = [(r, c, op) for r in range(env.rows) for c in range(env.cols) for op in range(2)]

# === Hyperparameters ===
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 500
steps_per_episode = 20

# === Logging for analysis ===
failure_log = []
redirect_log = []
reward_log = []

# === Set Seed for Reproducibility ===
SEED = 42
env.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# === Training Loop ===
env.reset()
for ep in range(episodes):
    state = get_state(env)
    total_reward = 0
    failures, redirects = 0, 0

    for _ in range(steps_per_episode):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = random.choice(possible_actions)
        else:
            action = max(possible_actions, key=lambda a: q_table[(state, a)])

        obs, reward, done, _ = env.step(action)
        next_state = get_state(env)

        # Q-Learning update
        best_next = max(q_table[(next_state, a)] for a in possible_actions)
        q_table[(state, action)] += alpha * (
            reward + gamma * best_next - q_table[(state, action)]
        )

        state = next_state
        total_reward += reward

    # Track performance
    _, failures, redirects = env.check_coverage()
    failure_log.append(failures)
    redirect_log.append(redirects)
    reward_log.append(total_reward)

    if ep % 25 == 0:
        print(f"Episode {ep}: Reward = {total_reward:.3f}, Failures = {failures}, Redirects = {redirects}")

print("\nTraining Complete")

# === Testing Phase ===
print("\n=== Testing Learned Policy ===")
env.reset()
state = get_state(env)
for step in range(10):
    action = max(possible_actions, key=lambda a: q_table[(state, a)])
    obs, reward, done, _ = env.step(action)
    print(f"\nStep {step}: Action = {action}, Reward = {reward:.3f}")
    env.render()
    state = get_state(env)

# %%
