# %%
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from grid_env import CellularNetworkEnv
import pandas as pd

# === Quadrant-based State Encoding ===
def get_state(env):
    q1 = np.sum(env.car_grid[:3, :3])
    q2 = np.sum(env.car_grid[:3, 3:])
    q3 = np.sum(env.car_grid[3:, :3])
    q4 = np.sum(env.car_grid[3:, 3:])

    a1 = np.sum(env.antenna_grid[:3, :3])
    a2 = np.sum(env.antenna_grid[:3, 3:])
    a3 = np.sum(env.antenna_grid[3:, :3])
    a4 = np.sum(env.antenna_grid[3:, 3:])

    return (
        int(q1 // 500), int(q2 // 500), int(q3 // 500), int(q4 // 500),
        int(a1 // 5), int(a2 // 5), int(a3 // 5), int(a4 // 5)
    )

# === Smart Action Sampling: Bias toward user density ===
def sample_biased_action(env, op_set):
    weights = env.car_grid.flatten()
    weights = weights / weights.sum()
    idx = np.random.choice(np.arange(env.rows * env.cols), p=weights)
    r, c = divmod(idx, env.cols)
    op = random.choice(op_set)
    return (r, c, op)

# === Initialize Environment ===
env = CellularNetworkEnv()
q_table = defaultdict(float)
action_counter = Counter()

# === Hyperparameters ===
alpha = 0.1
gamma = 0.95
episodes = 750
steps_per_episode = 30

max_epsilon = 1.0
min_epsilon = 0.01
# decay_rate = 0.995
decay_rate = 0.001

epsilon = max_epsilon

# === Logs ===
log = []  # place this before training starts

SEED = 42
env.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# === Training Loop ===
for ep in range(episodes):
    env.reset()
    state = get_state(env)
    total_reward = 0

    # Use only add-actions early, introduce remove later
    op_set = [0] if ep < 200 else [0, 1]
    possible_actions = [(r, c, op) for r in range(env.rows) for c in range(env.cols) for op in op_set]

    for _ in range(steps_per_episode):
        if np.random.rand() < epsilon:
            action = sample_biased_action(env, op_set)
        else:
            action = max(possible_actions, key=lambda a: q_table[(state, a)])

        obs, reward, done, _ = env.step(action)
        next_state = get_state(env)

        # Q-learning update
        best_next = max(q_table[(next_state, a)] for a in possible_actions)
        q_table[(state, action)] += alpha * (
            reward + gamma * best_next - q_table[(state, action)]
        )

        state = next_state
        total_reward += reward
        action_counter[action] += 1

    # After the episode ends: exploration rate decay which is exponential decay
    # epsilon = max(min_epsilon, epsilon * decay_rate) # multiplicative decay
    epsilon = min_epsilon + \
    (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)  # smooth exponential decay

    _, failures, redirects = env.check_coverage()
    log.append({
        "episode": ep,
        "sim_time": env.sim_time_hours,
        "time_label": env.get_sim_time(),
        "failures": failures,
        "redirects": redirects,
        "reward": total_reward,
        "epsilon": epsilon
    })

    if ep % 25 == 0:
        print(f"Episode {ep}: Reward = {total_reward:.3f}, Failures = {failures}, Redirects = {redirects}")

print("\n✅ Training Complete")

# === Optional Visualization ===
print("\n🎥 Animating user movement after training...")
env.animate_car_grid(steps=30, interval=300)

# === Action Frequency Check ===
print("\nTop Actions:")
for action, count in action_counter.most_common(5):
    print(f"{action}: {count} times")

# === Plot Results ===
# Convert log to DataFrame
df = pd.DataFrame(log)

plt.figure(figsize=(12, 5))
plt.plot(df["episode"], df["reward"], label='Reward', linewidth=1.5)
plt.plot(df["episode"], df["failures"], label='Failures', linewidth=1.2)
plt.plot(df["episode"], df["redirects"], label='Redirects', linewidth=1.2)

plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Q-Learning Performance (Time-Aware Log)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# === Test the Learned Policy ===
print("\n=== Testing Learned Policy ===")
env.reset()
state = get_state(env)
possible_actions = [(r, c, op) for r in range(env.rows) for c in range(env.cols) for op in [0, 1]]

for step in range(10):
    action = max(possible_actions, key=lambda a: q_table[(state, a)])
    obs, reward, done, _ = env.step(action)
    print(f"\nStep {step}: Action = {action}, Reward = {reward:.3f}")
    env.render()
    state = get_state(env)

baseline_failures = []
for _ in range(20):
    env.reset()
    _, f, _ = env.check_coverage()
    baseline_failures.append(f)
print("Average failure on random reset:", np.mean(baseline_failures))
