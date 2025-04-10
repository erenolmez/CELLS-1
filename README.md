# 📡 CELLS: Cellular Environment for Load-aware and Location-aware Simulation

**CELLS** is a simulation framework for studying cellular coverage, congestion, and infrastructure strategies in a dynamic, grid-based city environment. It models human mobility over time and evaluates the performance of antenna placement and resource allocation under capacity and cost constraints.

---

## 🌍 Overview

CELLS simulates:
- 🧍‍♂️ Thousands of mobile users navigating a city using a **Markovian movement model**
- 📡 Antennas with limited **range and capacity**, placed on a discrete 6×6 grid
- 🧠 A flexible simulation loop that supports:
  - Rule-based strategies
  - Optimization methods (e.g., linear programming, heuristics)
  - Machine learning or reinforcement learning agents (optional)
- ⏱️ **Time-aware tracking** with `1 simulation step = 1 real-world hour`

---

## 🧰 Key Features

- ✅ **Grid-based city simulation**
  - User distribution evolves over time
  - Coverage, redirection, and failure events tracked dynamically

- 📶 **Antenna control interface**
  - Add or remove antennas at any cell
  - Analyze the impact of spatial decisions over time

- 📊 **Detailed logging**
  - Sim time (hour, day)
  - Coverage quality: number of failed or redirected users
  - Infrastructure cost

- 🔁 **Modular architecture**
  - Drop-in support for external controllers, decision logic, or AI agents

---

## ⌛ Time Modeling

Each simulation step corresponds to **1 hour** of real time.

- 24 steps = 1 day
- 720 steps = ~1 month
- Users move with a tunable probability `p_move` calibrated to match ~4–6 location changes/day
- Antennas can be reconfigured over time to match shifting load