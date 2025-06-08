import numpy as np
import random
import time


class CellularNetworkEnv:
    def __init__(self, rows=20, cols=20,
                 total_users=50000, antenna_capacity=300,
                 time_step=60):
        self.rows, self.cols = rows, cols
        self.total_users = total_users
        self.antenna_capacity = antenna_capacity
        self.time_step = time_step
        self.sim_time_hours = 0

        self.coverage_radius = 1
        self.num_cells = rows * cols
        self.max_antennas = self.num_cells
        self.num_antennas = 0

        self.car_grid = np.zeros((rows, cols), dtype=int)
        self.antenna_grid = np.zeros((rows, cols), dtype=int)

        self._compute_neighbor_map()
        self.place_users()
        self.place_antennas()

    def _compute_neighbor_map(self):
        self.neighbor_map = {
            (r, c): [(r + dr, c + dc)
                     for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                     if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]
            for r in range(self.rows) for c in range(self.cols)
        }

    def compute_dynamic_p_move(self):
        return 0.4 * (self.time_step / 60)

    def get_sim_time(self):
        d, h = divmod(self.sim_time_hours, 24)
        return f"Day {d}, {h:02d}:00"

    def place_users(self):
        flat = np.random.multinomial(self.total_users,
                                     [1 / self.num_cells] * self.num_cells)
        self.car_grid = flat.reshape(self.rows, self.cols)

    def move_users_markov_chain(self):
        p_move = self.compute_dynamic_p_move()
        new_grid = np.zeros_like(self.car_grid)
        for r in range(self.rows):
            for c in range(self.cols):
                users = self.car_grid[r, c]
                if users == 0:
                    continue
                neighbors = self.neighbor_map[(r, c)]
                k = len(neighbors)
                missing = 8 - k
                stay_prob = 1.0 - p_move + missing * (p_move / 8)
                probs = [stay_prob] + [p_move / 8] * k
                moves = np.random.multinomial(users, probs)
                new_grid[r, c] += moves[0]
                for idx, cnt in enumerate(moves[1:]):
                    nr, nc = neighbors[idx]
                    new_grid[nr, nc] += cnt
        self.car_grid = new_grid

    def greedy_place_antennas(self, num_antennas, density):
        """
        Greedy placement by picking highest-density cells.
        """
        top_idx = np.argsort(density.ravel())[::-1][:num_antennas]
        self.antenna_grid.fill(0)
        for idx in top_idx:
            r, c = divmod(idx, self.cols)
            self.antenna_grid[r, c] = 1

    def calculate_reward(self, failures, redirects):
        total_users = self.total_users
        total_antennas = np.sum(self.antenna_grid)

        failure_penalty = failures / total_users  # Normalize [0, 1]
        redirect_penalty = redirects / total_users
        antenna_cost = total_antennas / self.max_antennas  # Normalize [0, 1]

        reward = 1.0 - (2.0 * failure_penalty + 0.1 * redirect_penalty + 0.5 * antenna_cost)
        return reward

    def check_coverage(self):
        """
        Returns covered_grid (0-100%), failures, redirects.
        Redirects count when skipping saturated antennas.
        """
        covered_grid = np.zeros((self.rows, self.cols), dtype=int)
        failures = 0
        redirects = 0
        antenna_load = np.zeros((self.rows, self.cols), dtype=int)

        offsets = [(0,0), (-1,0), (1,0), (0,-1), (0,1),
                   (-1,-1), (-1,1), (1,-1), (1,1)]

        for r in range(self.rows):
            for c in range(self.cols):
                users_in_cell = self.car_grid[r, c]
                remaining = users_in_cell
                served = 0
                skipped_saturated = False

                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                        continue
                    cap = self.antenna_grid[nr, nc] * self.antenna_capacity
                    free = max(0, cap - antenna_load[nr, nc])
                    if free == 0 and cap > 0:
                        skipped_saturated = True
                        continue
                    if remaining == 0:
                        break
                    alloc = min(free, remaining)
                    antenna_load[nr, nc] += alloc
                    remaining -= alloc
                    served += alloc
                    if skipped_saturated:
                        redirects += alloc
                failures += remaining
                if users_in_cell > 0:
                    covered_grid[r, c] = int(100 * served / users_in_cell)
        return covered_grid, failures, redirects

    def place_antennas(self, warmup_hours=6, search_range=None, coverage_tolerance=0.97):
        """
        Greedy density placement, then choose the minimal antenna count
        achieving coverage >= coverage_tolerance * max_coverage.
        """
        print("\n⏳ Optimizing antenna count for minimal satellites…")
        self.place_users()
        density = np.zeros_like(self.car_grid, dtype=float)
        for _ in range(warmup_hours):
            self.move_users_markov_chain()
            density += self.car_grid
        density /= warmup_hours

        total_range = range(1, self.num_cells + 1)
        coverage_list = []
        grid_list = []

        # Evaluate coverage for all antenna counts
        for n in total_range:
            self.greedy_place_antennas(n, density)
            covered, failures, redirects = self.check_coverage()
            cov_pct = np.mean(covered)
            reward = self.calculate_reward(failures, redirects)
            coverage_list.append(cov_pct)
            grid_list.append(self.antenna_grid.copy())
            print(f"  • {n:3d} antennas → Coverage: {cov_pct:.2f}% | Redirects: {redirects} | Reward: {reward:.4f}")

        # Determine threshold
        max_cov = max(coverage_list)
        target = max_cov * coverage_tolerance
        # Find minimal n meeting target
        for idx, cov in enumerate(coverage_list):
            if cov >= target:
                best_n = idx + 1
                best_cov = cov
                # compute coverage, failures, redirects, reward
                covered, failures, best_redirects = self.check_coverage()
                best_reward = self.calculate_reward(failures, best_redirects)
                break
        best_grid = grid_list[best_n - 1]

        self.antenna_grid = best_grid
        self.num_antennas = best_n
        print(f"\n✅ Selected {best_n} antennas → Coverage: {best_cov:.2f}% (≥ {coverage_tolerance*100:.0f}% of {max_cov:.2f}%) | Redirects: {best_redirects} | Reward: {best_reward:.4f}")

    def simulate_without_animation(self, hours=8766):
        print(f"\n🚀 Running headless simulation for {hours} hours…")
        self.total_cov = 0
        self.total_cost = 0
        self.total_reward = 0
        self.total_redirects = 0
        for _ in range(hours):
            self.move_users_markov_chain()
            self.sim_time_hours += 1
            covered, failures, redirects = self.check_coverage()
            cov_pct = np.mean(covered)
            cost = failures + self.num_antennas * 5
            reward = self.calculate_reward(failures, redirects)
            self.total_cov += cov_pct
            self.total_cost += cost
            self.total_reward += reward
            self.total_redirects += redirects
            print(f"H{self.sim_time_hours:4d} | Cov {cov_pct:5.2f}% | Fail {failures:5d} | Redir {redirects:5d} | Ant {self.num_antennas:3d} | Cost {cost:6d} | Reward {reward:6.4f}")
        avg_cov = self.total_cov / hours
        avg_cost = self.total_cost / hours
        avg_reward = self.total_reward / hours
        avg_redir = self.total_redirects / hours
        print("\n📊 HEADLESS SIMULATION RESULTS")
        print(f"• Avg Coverage: {avg_cov:.2f}%")
        print(f"• Avg Cost    : {avg_cost:.2f}")
        print(f"• Avg Reward  : {avg_reward:.4f}")
        print(f"• Avg Redirects: {avg_redir:.2f}")
        print(f"• Cumulative Cost: {self.total_cost}")

# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":
    env = CellularNetworkEnv()
    start = time.time()
    env.simulate_without_animation(hours=8766)
    print("\n📍 Antenna placement matrix:")
    print(env.antenna_grid)
    elapsed = time.time() - start
    print(f"\n⏱ Total simulation time: {elapsed:.2f}s ({elapsed/60:.2f}m)")