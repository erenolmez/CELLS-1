import os
import re
import matplotlib.pyplot as plt

# === Path ===
log_path = os.path.join(os.path.dirname(__file__), "ga_terminal_log2.txt")

# === Read ===
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

gens, rewards, antennas = [], [], []
current_gen = None

# === Structured parse ===
for line in lines:
    gen_match = re.search(r"Generation\s+(\d+)", line)
    if gen_match:
        current_gen = int(gen_match.group(1))

    reward_match = re.search(r"Best reward this gen:\s*(-?\d+\.\d+)", line)
    ant_match = re.search(r"Antennas used:\s*(\d+)", line)

    if current_gen is not None and reward_match and ant_match:
        gens.append(current_gen)
        rewards.append(float(reward_match.group(1)))
        antennas.append(int(ant_match.group(1)))
        current_gen = None

# === Validate ===
assert gens and rewards and antennas, "No valid data parsed from log."

# === Plot ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# --- Reward curve (left y-axis)
ax1.plot(gens, rewards, color="tab:blue", linewidth=2.0, label="Best Reward")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Reward", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.3)

# --- Antenna count (right y-axis)
ax2 = ax1.twinx()
ax2.plot(gens, antennas, color="tab:green", linewidth=1.8, linestyle="--", label="Antenna Count")
ax2.set_ylabel("Antennas", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")

# --- Shaded phases
ax1.axvspan(0, 50, color='gray', alpha=0.07, label="Phase 1")
ax1.axvspan(50, 200, color='gold', alpha=0.10, label="Phase 2")
ax1.axvspan(200, max(gens), color='red', alpha=0.06, label="Phase 3")

# --- Bold vertical markers
ax1.axvline(50, color="gray", linestyle="--", linewidth=2.0)
ax1.axvline(200, color="red", linestyle="--", linewidth=2.0)

# --- Bolder annotations
if len(rewards) > 199:
    ax1.annotate("↳ Reinforcement", xy=(50, rewards[49]), xytext=(60, rewards[49] - 0.05),
                 arrowprops=dict(arrowstyle="->", lw=2.0, color="gray"),
                 fontsize=10, color="gray", fontweight="bold")

    ax1.annotate("↳ 1-Year Eval", xy=(200, rewards[199]), xytext=(210, rewards[199] - 0.05),
                 arrowprops=dict(arrowstyle="->", lw=2.0, color="red"),
                 fontsize=10, color="red", fontweight="bold")

# --- Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Genetic Algorithm Progress — Reward & Antenna Count", fontweight="bold")
plt.tight_layout()
plt.show()