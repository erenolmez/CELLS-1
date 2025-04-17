"""
DQN Training Script for CellularNetworkEnv
=========================================
This script implements a Deep Q‑Network (DQN) agent for the 6×6 cellular‑antenna
optimization environment you already created (`grid_env.py`).

Key Features
------------
* **Raw State Input** – concatenates the car grid and antenna grid (72‑dim vector)
* **Discrete Action Space** – 72 actions (6 rows × 6 cols × 2 ops)
* **Experience Replay** – fixed‑size buffer for stability
* **Target Network** – updated at intervals to reduce oscillations
* **ε‑Greedy Policy** – with exponential decay
* **Logging & Visualization** – reward, failures, redirects plotted after training
* **Seed Control** – for reproducibility

Usage
-----
Run directly:
```
python dqn_train.py
```
It will train for the default 750 episodes and then animate user movement using
the learned policy, similar to `q_learn.py`.
"""

from __future__ import annotations
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from grid_env import CellularNetworkEnv

# ========================
#      Hyper‑parameters
# ========================
EPISODES = 750
STEPS_PER_EPISODE = 30
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
TARGET_UPDATE_FREQ = 1_000          # in environment steps
MAX_EPS = 1.0
MIN_EPS = 0.05
EPS_DECAY = 0.0008                 # exp decay rate
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
#   Utility: Action <‑> Id
# ========================
N_ROWS = 6
N_COLS = 6
N_OPS = 2  # 0 = add, 1 = remove
N_ACTIONS = N_ROWS * N_COLS * N_OPS  # 72


def encode_action(r: int, c: int, op: int) -> int:
    """Map (row, col, op) -> int in [0, 71]."""
    return r * (N_COLS * N_OPS) + c * N_OPS + op


def decode_action(action_id: int) -> Tuple[int, int, int]:
    """Inverse of `encode_action`."""
    r = action_id // (N_COLS * N_OPS)
    c = (action_id % (N_COLS * N_OPS)) // N_OPS
    op = action_id % N_OPS
    return r, c, op

# ========================
#     Replay Buffer
# ========================
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

# ========================
#        Q‑Network
# ========================
class DQN(nn.Module):
    def __init__(self, input_dim: int = 72, output_dim: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ========================
#      Training Helpers
# ========================

def select_action(state: np.ndarray, policy_net: DQN, steps_done: int) -> int:
    """ε‑greedy action selection."""
    eps_threshold = MIN_EPS + (MAX_EPS - MIN_EPS) * np.exp(-EPS_DECAY * steps_done)
    if random.random() < eps_threshold:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().to(DEVICE).unsqueeze(0)
        q_values = policy_net(state_t)
        return int(q_values.argmax(dim=1).item())


def compute_td_loss(
    batch: Transition, policy_net: DQN, target_net: DQN, optimizer: optim.Optimizer
):
    state_batch = torch.from_numpy(np.vstack(batch.state)).float().to(DEVICE)
    action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_state_batch = torch.from_numpy(np.vstack(batch.next_state)).float().to(DEVICE)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    q_values = policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1, keepdim=True)[0]
        target = reward_batch + (1 - done_batch) * GAMMA * next_q_values

    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ========================
#             Main
# ========================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = CellularNetworkEnv()
    env.seed(SEED)

    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(BUFFER_CAPACITY)

    episode_metrics: List[dict] = []
    steps_done = 0  # global env steps for epsilon & target update

    for ep in range(EPISODES):
        obs = env.reset()
        state = np.concatenate([env.car_grid.flatten(), env.antenna_grid.flatten()])
        total_reward = 0.0

        for _ in range(STEPS_PER_EPISODE):
            action_id = select_action(state, policy_net, steps_done)
            action_tuple = decode_action(action_id)
            obs, reward, done, _ = env.step(action_tuple)
            next_state = np.concatenate([
                env.car_grid.flatten(), env.antenna_grid.flatten()
            ])

            memory.push(state, action_id, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            # Learn
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                compute_td_loss(batch, policy_net, target_net, optimizer)

            # Target network update
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Episode summary
        _, failures, redirects = env.check_coverage()
        episode_metrics.append({
            "episode": ep,
            "reward": total_reward,
            "failures": failures,
            "redirects": redirects,
        })

        if ep % 25 == 0:
            print(
                f"Ep {ep:03d} | Reward: {total_reward:6.2f} | "
                f"Failures: {failures:5d} | Redirects: {redirects:5d} | "
                f"Buffer: {len(memory):5d}"
            )

    # =======================
    #   Training Completed
    # =======================
    print("\n✅ DQN training complete!")

    # Save model
    model_path = Path("dqn_policy_net.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path.absolute()}")

    # Plot results
    df = pd.DataFrame(episode_metrics)
    plt.figure(figsize=(12, 5))
    plt.plot(df["episode"], df["reward"], label="Reward", linewidth=1.5)
    plt.plot(df["episode"], df["failures"], label="Failures", linewidth=1.2)
    plt.plot(df["episode"], df["redirects"], label="Redirects", linewidth=1.2)
    plt.xlabel("Episode")
    plt.legend()
    plt.title("DQN Performance over Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =============================
    #  Test the Learned Policy
    # =============================
    print("\n=== Testing Learned Policy ===")
    env.reset()
    state = np.concatenate([env.car_grid.flatten(), env.antenna_grid.flatten()])

    for step in range(10):
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().to(DEVICE).unsqueeze(0)
            action_id = int(policy_net(state_t).argmax(dim=1).item())
        action_tuple = decode_action(action_id)
        obs, reward, done, _ = env.step(action_tuple)
        print(f"Step {step}: Action={action_tuple}, Reward={reward:.3f}")
        env.render()
        state = np.concatenate([env.car_grid.flatten(), env.antenna_grid.flatten()])

    # Optional animation like in q_learn
    print("\n🎥 Animating user movement after DQN training…")
    env.animate_car_grid(steps=30, interval=300)


if __name__ == "__main__":
    main()