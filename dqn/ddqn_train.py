"""
Double DQN “exploratory‑design” Trainer
=======================================
Context (🆕 per Prof. mail)
--------------------------
* We **do not** intend to keep moving antennas during deployment.
* RL is exploited *offline* to **search** a good *static* layout that
  copes well with dynamic user flows.
* After training we freeze the best layout and evaluate it for a long
  horizon (e.g. 10 days) exactly as a “static” method would.

Key Script Features
-------------------
• 73 discrete actions – *add / remove* on (6 × 6) grid + NO‑OP  
• **ACTION_PERIOD** (🆕)   → agent may touch the layout only every *K*
  simulated hours (default 24 h); all other steps are forced NO‑OP  
• Unique‑state hashing to gauge exploration  
• Early stopping once failure count stops improving for ≥ patience  
• CUDA‑ready, falls back to CPU automatically  

Run:
    python ddqn_train.py
"""
from __future__ import annotations
import random, hashlib
from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from grid_env import CellularNetworkEnv


# ─────────────────────────────────────────────
# 1.  Hyper‑parameters
# ─────────────────────────────────────────────
GAMMA              = 0.99
LR                 = 1e-3
BATCH_SIZE         = 64
BUFFER_SIZE        = 50_000
TARGET_UPDATE_FREQ = 500       # env steps

EPISODES          = 3_000
STEPS_PER_EPISODE = 240         # 60 h  ≈ 2.5 days per episode
ACTION_PERIOD     = 24         # 🆕 agent allowed to act once per 24 h

EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.0002         # keeps ε≈0.2 until ~1500 eps

SEED = 42


# ─────────────────────────────────────────────
# 2.  Action helpers   (6×6×2  +  NO‑OP = 73)
# ─────────────────────────────────────────────
N_ROWS, N_COLS, N_OPS = 6, 6, 2
N_ACTIONS = N_ROWS * N_COLS * N_OPS + 1   # 72 + 1
NOOP_ID   = 72                            # final index

def encode_action(r:int, c:int, op:int) -> int:
    return r * (N_COLS*N_OPS) + c*N_OPS + op

def decode_action(idx:int) -> Tuple[int,int,int] | None:
    if idx == NOOP_ID:          # forced NO‑OP
        return None
    r = idx // (N_COLS*N_OPS)
    c = (idx % (N_COLS*N_OPS)) // N_OPS
    op = idx % N_OPS
    return r, c, op


# ─────────────────────────────────────────────
# 3.  Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity:int):
        self.buf = deque(maxlen=capacity)
    def push(self,*tr): self.buf.append(tuple(tr))
    def sample(self, k):
        batch = random.sample(self.buf, k)
        s,a,r,s2,d = zip(*batch)
        return (np.array(s),
                np.array(a),
                np.array(r,  np.float32),
                np.array(s2),
                np.array(d,  np.float32))
    def __len__(self): return len(self.buf)


# ─────────────────────────────────────────────
# 4.  Q‑Network
# ─────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, inp:int = 72, out:int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp,128), nn.ReLU(),
            nn.Linear(128,64),  nn.ReLU(),
            nn.Linear(64,out)
        )
    def forward(self,x): return self.net(x)


# ─────────────────────────────────────────────
# 5.  Utilities
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eps_threshold(step:int):
    return EPSILON_END + (EPSILON_START-EPSILON_END)*np.exp(-EPSILON_DECAY*step)

def hash_state(vec:np.ndarray) -> str:
    return hashlib.md5(np.round(vec,3).tobytes()).hexdigest()


# ─────────────────────────────────────────────
# 6.  Environment & seeding
# ─────────────────────────────────────────────
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
env = CellularNetworkEnv()
env.seed(SEED)

policy_net = QNet().to(device)
target_net = QNet().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimiser = optim.Adam(policy_net.parameters(), lr=LR)
replay    = ReplayBuffer(BUFFER_SIZE)


# ─────────────────────────────────────────────
# 7.  Training loop (Double DQN)
# ─────────────────────────────────────────────
unique_states=set()
log=[]

best_fail=float('inf'); patience=300; wait=0
step_ctr=0

for ep in range(EPISODES):
    obs   = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device)
    ep_reward = 0.0

    for t in range(STEPS_PER_EPISODE):
        step_ctr += 1

        # ――― choose action ―――
        if t % ACTION_PERIOD == 0:                     # 🆕 agent may act
            if random.random() < eps_threshold(step_ctr):
                a_id = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    a_id = int(policy_net(state).argmax().item())
        else:
            a_id = NOOP_ID                             # forced NO‑OP

        # ――― env step ―――
        next_obs, reward, done, _ = env.step(decode_action(a_id))
        next_state = torch.tensor(next_obs, dtype=torch.float32, device=device)

        unique_states.add(hash_state(next_state.cpu().numpy()))
        replay.push(state.cpu().numpy(), a_id, reward,
                    next_state.cpu().numpy(), done)

        state = next_state
        ep_reward += reward

        # ――― learn ―――
        if len(replay) >= BATCH_SIZE:
            s,a,r,s2,d = replay.sample(BATCH_SIZE)
            s  = torch.tensor(s, dtype=torch.float32, device=device)
            a  = torch.tensor(a, dtype=torch.int64,  device=device).unsqueeze(1)
            r  = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
            s2 = torch.tensor(s2,dtype=torch.float32,device=device)
            d  = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

            q    = policy_net(s).gather(1,a)
            with torch.no_grad():
                a_sel  = policy_net(s2).argmax(1,keepdim=True)
                q_next = target_net(s2).gather(1,a_sel)
                tgt = r + (1-d)*GAMMA*q_next

            loss = nn.functional.mse_loss(q, tgt)
            optimiser.zero_grad(); loss.backward(); optimiser.step()

        if step_ctr % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # ――― episode summary ―――
    _,fails,reds = env.check_coverage()
    log.append(dict(ep=ep,rew=ep_reward,fail=fails,red=reds))

    if fails < best_fail: best_fail,wait=fails,0
    else: wait+=1
    if wait >= patience:
        print(f"↪️  Early‑stop @ ep {ep}, best failures {best_fail}")
        break


# ─────────────────────────────────────────────
# 8.  Results
# ─────────────────────────────────────────────
print("\n✅ Double DQN training complete")
print(f"🔸 unique states : {len(unique_states)}")
print(f"🔸 replay size   : {len(replay)}")
print(f"🔸 device        : {device}")

df = pd.DataFrame(log)
plt.figure(figsize=(12,5))
plt.plot(df.ep, df.rew,  label="Reward")
plt.plot(df.ep, df.fail, label="Failures")
plt.plot(df.ep, df.red,  label="Redirects")
plt.legend(); plt.grid(True); plt.xlabel("Episode"); plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 9.  Freeze layout & animate 24 h
# ─────────────────────────────────────────────
print("\n🎥 Deterministic rollout with learned policy…")
env.reset()
state = torch.tensor(np.concatenate([env.car_grid.ravel(),
                                     env.antenna_grid.ravel()]),
                     dtype=torch.float32, device=device)
for _ in range(24):
    with torch.no_grad():
        a_id = int(policy_net(state).argmax().item())
    nxt,_,_,_ = env.step(decode_action(a_id))
    state = torch.tensor(nxt, dtype=torch.float32, device=device)

env.animate_car_grid(steps=24, interval=400)
