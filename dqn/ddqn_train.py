"""
Double DQN Training Script for CellularNetworkEnv
================================================
This file upgrades vanilla DQN to **Double DQN** **and** keeps the same
visualisation tools you liked from `q_learn.py` (training curves **and** live
user‑movement animation).
Run:
```bash
python dqn_train.py
```
"""

import random
from collections import deque
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib

from grid_env import CellularNetworkEnv

unique_states = set()

def hash_state(state_vector):
    # Round to reduce sensitivity, then hash
    rounded = tuple(np.round(state_vector, decimals=3))
    return hashlib.md5(str(rounded).encode()).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Hyper‑parameters
# ──────────────────────────────────────────────────────────────────────────────
GAMMA = 0.99            # discount factor
LR = 1e-3               # Adam learning rate
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
TARGET_UPDATE_FREQ = 500  # sync target net every N env steps

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.0002   # keeps ε > 0.2 until ~1500 episodes

EPISODES = 3000
STEPS_PER_EPISODE = 60
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Action encoding helpers  (72 discrete actions)
# ──────────────────────────────────────────────────────────────────────────────

def encode_action(r: int, c: int, op: int) -> int:
    """(row, col, op) ➜ flat index 0‑71."""
    return r * 12 + c * 2 + op

def decode_action(idx: int):
    if idx == 72:
        return None  # no-op
    r = idx // 12
    c = (idx % 12) // 2
    op = idx % 2
    return r, c, op

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, k):
        batch = random.sample(self.buffer, k)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Neural Net
# ──────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 72, output_dim: int = 72):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Env & seeding
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = CellularNetworkEnv()
env.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dqn = QNetwork().to(device)          # policy net
target_dqn = QNetwork().to(device)   # target net
target_dqn.load_state_dict(dqn.state_dict())

a_optimizer = optim.Adam(dqn.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Training Loop – Double DQN
# ──────────────────────────────────────────────────────────────────────────────

epsilon = EPSILON_START
step_counter = 0
log = []

best_failures = float('inf')   # lowest failure count seen so far
patience      = 300            # how many episodes with no new best before stopping
wait          = 0              # episodes since last improvement

for ep in range(EPISODES):
    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device)
    ep_reward = 0.0

    for _ in range(STEPS_PER_EPISODE):
        step_counter += 1
        # ε‑greedy action selection (policy net)
        if random.random() < epsilon:
            a_idx = random.randrange(72)
        else:
            with torch.no_grad():
                a_idx = dqn(state).argmax().item()

        a = decode_action(a_idx)
        obs2, r, done, _ = env.step(a)
        next_state = torch.tensor(obs2, dtype=torch.float32, device=device)
        buffer.push(state.cpu().numpy(), a_idx, r, next_state.cpu().numpy(), done)

        # ▶︎ Track every visited state  (ADD THIS LINE)
        unique_states.add(hash_state(next_state.cpu().numpy()))

         # store transition
        buffer.push(state.cpu().numpy(), a_idx, r,
        next_state.cpu().numpy(), done)
    
        state = next_state
        ep_reward += r

        # Learn
        if len(buffer) >= BATCH_SIZE:
            s, a_b, r_b, s2, d_b = buffer.sample(BATCH_SIZE)
            s = torch.tensor(s, dtype=torch.float32, device=device)
            a_b = torch.tensor(a_b, dtype=torch.int64, device=device).unsqueeze(1)
            r_b = torch.tensor(r_b, dtype=torch.float32, device=device).unsqueeze(1)
            s2 = torch.tensor(s2, dtype=torch.float32, device=device)
            d_b = torch.tensor(d_b, dtype=torch.float32, device=device).unsqueeze(1)

            # Double DQN targets
            a_sel = dqn(s2).argmax(1, keepdim=True)
            with torch.no_grad():
                q_next = target_dqn(s2).gather(1, a_sel)
                td_target = r_b + (1 - d_b) * GAMMA * q_next

            q_curr = dqn(s).gather(1, a_b)
            loss = nn.MSELoss()(q_curr, td_target)

            a_optimizer.zero_grad()
            loss.backward()
            a_optimizer.step()

        # Sync target net
        if step_counter % TARGET_UPDATE_FREQ == 0:
            target_dqn.load_state_dict(dqn.state_dict())

    # ε decay per episode
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY * ep)

    # Log metrics
    _, fails, reds = env.check_coverage()
    log.append({"episode": ep, "reward": ep_reward, "fails": fails, "reds": reds, "eps": epsilon})

    # --- Early‑stopping check ---------------------------------
    if fails < best_failures:
        best_failures = fails
        wait = 0                     # reset patience
    else:
        wait += 1

    if wait >= patience:
        print(f"↪️  Early stop at episode {ep} – failures plateaued "
              f"(best = {best_failures}).")
        break
    # ----------------------------------------------------------
    # if ep % 25 == 0:
        # print(f"Ep {ep:3d} | R {ep_reward:7.3f} | F {fails:4d} | ε {epsilon:.3f}")
    if ep == EPISODES - 1:
        print(f"✅ Training complete — final reward: {ep_reward:.3f}, failures: {fails}, epsilon: {epsilon:.3f}")

print("\n✅ Double DQN training complete.")
print(f"\n📊 Unique states visited during training: {len(unique_states)}")
print(f"📦 Replay buffer size    : {len(buffer)}")
print(f"💻 Training was done on: {device}")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Plot training curves
# ──────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(log)
plt.figure(figsize=(12, 5))
plt.plot(df['episode'], df['reward'], label='Reward')
plt.plot(df['episode'], df['fails'], label='Failures')
plt.plot(df['episode'], df['reds'], label='Redirects')
plt.xlabel('Episode')
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 8.  Quick deterministic rollout + animation
# ──────────────────────────────────────────────────────────────────────────────
print("\n🎥 Animating user movement after training...")
# Show 24 simulated hours (one per env.step)
obs = env.reset()
state = torch.tensor(obs, dtype=torch.float32, device=device)
for t in range(24):
    with torch.no_grad():
        a_idx = dqn(state).argmax().item()
    a = decode_action(a_idx)
    obs2, _, _, _ = env.step(a)
    state = torch.tensor(obs2, dtype=torch.float32, device=device)
# Use built‑in heat‑map animation
env.animate_car_grid(steps=24, interval=400)
