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

CELLS includes time-aware simulation logic with configurable temporal resolution:

- Each **simulation step** corresponds to a defined amount of **real-world time**, set via the `time_step` parameter (in minutes).
  - Default: `time_step = 60` → 1 sim step = 1 real hour
  - Examples:
    - `time_step = 15` → 4 steps = 1 hour
    - `time_step = 120` → 1 step = 2 hours

- User movement is governed by a **Markov Chain model**, with the probability of user relocation (`p_move`) automatically scaled based on the `time_step`.
  - At `time_step = 60`, the default `p_move = 0.4`
  - Shorter steps → lower movement probability
  - Longer steps → higher movement probability

- This scaling aims to maintain consistency in movement behavior across different simulation resolutions.

- Time is internally tracked and used for step-based animation, performance evaluation, and metric logging.
  - 24 steps (at 60 mins/step) = 1 simulated day  
  - 720 steps ≈ 1 simulated month