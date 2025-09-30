# make_yearly_reports.py
# Generate all requested 1-year figures for both mobility modes.
#
# Outputs:
#   ./report_uniform/*.png
#   ./report_attraction/*.png
#   ./report_compare/capacity_util_boxplot.png
#
# Notes:
# - The *only* reward-based figure now reads the real GA logs (no env-made values).
# - For uniform mode, the reward plot is skipped until ga_terminal_log3.txt is provided.
# - Coverage visuals with antenna overlays use a **fixed 0–100% colorbar**.
# - Summary charts (monthly/per-cell/overall lines) may auto-zoom via cov_limits.

from __future__ import annotations
import json
from pathlib import Path
import re
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from updated_grid_env import CellularNetworkEnv

# ========================= Config =========================
ROWS, COLS        = 20, 20
TOTAL_USERS       = 50_000
ANT_CAP           = 300
TIME_STEP_MIN     = 60           # 1 hour per step
STEPS_PER_YEAR    = 8_760
SEED              = 42
_RNG = np.random.default_rng(SEED)

GRIDS = {
    "uniform":    Path("best_uniform.npy"),
    "attraction": Path("best_attraction.npy"),
}

# Map movement mode -> GA log to use for *reward* plots only
GA_LOGS = {
    "uniform":    Path("ga_terminal_log3.txt"),
    "attraction": Path("ga_terminal_log2.txt"),
}

# GA code path (to parse constants like FINAL_WINDOW / reinforcement start)
GA_CODE_PATH = (Path(__file__).resolve().parent / "ga_fast_parallel_cpu.py")

# ====================== Utilities =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(outdir: Path, name: str):
    outpath = outdir / f"{name}.png"
    fig = plt.gcf()
    try:
        has_prop = hasattr(fig, "get_constrained_layout")
        if has_prop and not fig.get_constrained_layout():
            fig.tight_layout()
        elif not has_prop:
            fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=220)
    print(f"✔ saved {outpath}")
    plt.close(fig)

def build_env(best_grid: np.ndarray, mode: str) -> CellularNetworkEnv:
    env = CellularNetworkEnv(
        rows=best_grid.shape[0],
        cols=best_grid.shape[1],
        total_users=TOTAL_USERS,
        antenna_capacity=ANT_CAP,
        time_step=TIME_STEP_MIN,
        movement_mode=mode,
    )
    env.antenna_grid = best_grid.copy()
    env.seed(SEED)
    env.place_users()
    return env

def random_baseline_grid_like(best_grid: np.ndarray) -> np.ndarray:
    rows, cols = best_grid.shape
    total = int(best_grid.sum())
    baseline = np.zeros_like(best_grid, dtype=int)
    choices = _RNG.integers(0, rows * cols, size=total)
    for idx in choices:
        r, c = divmod(int(idx), cols)
        baseline[r, c] += 1
    return baseline

def cov_limits(arr: np.ndarray) -> tuple[float, float]:
    med = float(np.nanmedian(arr))
    if med >= 90:
        return 85.0, 100.0
    return 0.0, 100.0

def add_cell_grid(ax: plt.Axes, rows: int, cols: int, color="k", lw=0.4, alpha=0.25):
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=color, linestyle="-", linewidth=lw, alpha=alpha)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

# =================== Year-long simulation =================
def simulate_year(env: CellularNetworkEnv):
    steps = STEPS_PER_YEAR
    rows, cols = env.rows, env.cols

    fails_t  = np.zeros(steps, dtype=np.int64)
    reds_t   = np.zeros(steps, dtype=np.int64)
    reward_t = np.zeros(steps, dtype=np.float64)
    cov_userwise_t = np.zeros(steps, dtype=np.float64)

    user_density_sum = np.zeros((rows, cols), dtype=np.float64)
    cell_cov_sum     = np.zeros((rows, cols), dtype=np.float64)
    served_sum       = np.zeros((rows, cols), dtype=np.float64)

    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    hours_per_month = [d*24 for d in days_per_month]
    month_edges = np.cumsum([0] + hours_per_month)

    monthly_cov_sum = np.zeros(12, dtype=np.float64)
    monthly_counts  = np.zeros(12, dtype=np.int64)

    total_ant = max(int(env.antenna_grid.sum()), 0)
    antenna_cost = (total_ant / env.max_antennas) if env.max_antennas > 0 else 0.0

    snapshot_cov = None

    for t in range(steps):
        env.move_users_markov_chain()
        cov_grid, fails, reds = env.check_coverage()
        reward = env.calculate_reward(fails, reds)

        fails_t[t] = fails
        reds_t[t]  = reds
        reward_t[t] = reward
        cov_userwise_t[t] = 100.0 * (env.total_users - fails) / env.total_users

        user_density_sum += env.user_grid
        cell_cov_sum     += cov_grid
        served_sum       += (cov_grid / 100.0) * env.user_grid

        m = np.searchsorted(month_edges, t, side="right") - 1
        m = min(max(m, 0), 11)
        monthly_cov_sum[m] += cov_userwise_t[t]
        monthly_counts[m]  += 1

        snapshot_cov = cov_grid
        env.sim_time_hours += 1

    user_density_avg = user_density_sum / steps
    cell_cov_avg     = cell_cov_sum / steps
    denom_capacity   = steps * env.antenna_capacity * np.maximum(env.antenna_grid, 1)
    capacity_util    = (served_sum / denom_capacity).clip(0, 1)
    capacity_util    = capacity_util[env.antenna_grid > 0]

    monthly_cov_avg  = np.divide(monthly_cov_sum, np.maximum(monthly_counts, 1))
    episode_cost_t   = 1.0 - reward_t

    return dict(
        fails=fails_t,
        reds=reds_t,
        reward=reward_t,
        episode_cost=episode_cost_t,
        overall_cov=cov_userwise_t,
        monthly_cov=monthly_cov_avg,
        cell_cov_avg=cell_cov_avg,
        user_density_avg=user_density_avg,
        capacity_util=capacity_util,
        snapshot_cov=snapshot_cov,
        antenna_cost_const=antenna_cost,
    )

# ========================= GA log parsing =========================
def _parse_final_eval_window_from_ga(ga_src: str) -> int:
    m = re.search(r"gen\s*>=\s*\(\s*generations\s*-\s*(\d+)\s*\)", ga_src)
    if m:
        return int(m.group(1))
    m2 = re.search(r"generations\s*-\s*(\d+)", ga_src)
    if m2:
        return int(m2.group(1))
    return 50

def _parse_density_boost_start_from_ga(ga_src: str) -> int | None:
    m = re.search(r"DENSITY_BOOST_START_GEN\s*=\s*(\d+)", ga_src)
    if m:
        return int(m.group(1))
    m2 = re.search(r"density_boost_start_gen\s*:\s*int\s*=\s*(\d+)", ga_src)
    return int(m2.group(1)) if m2 else None

def _parse_ga_log(log_path: Path):
    if not log_path.exists():
        return [], [], [], None

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    gens: list[int] = []
    rewards: list[float] = []
    antennas: list[int] = []
    current_gen: int | None = None
    total_gens: int | None = None

    gen_hdr = re.compile(r"Generation\s+(\d+)\s*/\s*(\d+)", re.I)
    gen_loose = re.compile(r"Generation\s+(\d+)", re.I)
    reward_ant_combo = re.compile(
        r"(?:Best reward(?: this gen)?|Best reward)\s*:\s*([-\d\.eE]+)"
        r".*?(?:Antennas(?: used)?|📡\s*Antennas)\s*:\s*(\d+)",
        re.I,
    )

    for line in io.StringIO(text):
        if (m := gen_hdr.search(line)):
            current_gen = int(m.group(1)); total_gens = int(m.group(2))
        else:
            m2 = gen_loose.search(line)
            if m2:
                current_gen = int(m2.group(1))

        m3 = reward_ant_combo.search(line)
        if m3:
            r = float(m3.group(1)); a = int(m3.group(2))
            g = current_gen if current_gen is not None else (gens[-1] + 1 if gens else 1)
            gens.append(int(g)); rewards.append(r); antennas.append(a)
            current_gen = None

    return gens, rewards, antennas, total_gens

def _nearest_xy(gens: list[int], ys: list[float], x: int) -> tuple[int, float | None]:
    if not gens:
        return x, None
    idx = min(range(len(gens)), key=lambda i: abs(gens[i] - x))
    return gens[idx], ys[idx]

def _resolve_log_path(log_path: Path) -> Path | None:
    if log_path.exists():
        return log_path

    stem = log_path.stem
    candidates = [
        f"{stem}.txt",
        f"{stem}.log",
        f"{stem} - Copy.txt",
        f"{stem}_copy.txt",
        f"{stem}*.txt",
        f"{stem}*.log",
        "*log3*.txt" if "log3" in stem.lower() else "",
        "*ga_terminal_log3*.txt" if "log3" in stem.lower() else "",
    ]
    here = Path.cwd()
    for pat in filter(None, candidates):
        for hit in here.glob(pat):
            if hit.is_file():
                return hit

    script_dir = Path(__file__).resolve().parent
    for pat in filter(None, candidates):
        for hit in script_dir.glob(pat):
            if hit.is_file():
                return hit

    return None

# ========================= Plots ==========================
def plot_antenna_layout(outdir: Path, antenna_grid: np.ndarray, mode: str):
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    im = ax.imshow(antenna_grid, cmap="Blues")
    add_cell_grid(ax, *antenna_grid.shape)
    plt.title(f"Antenna Layout — {mode}")
    plt.xticks([]); plt.yticks([])
    cbar = plt.colorbar(im); cbar.set_label("Antennas per cell")
    savefig(outdir, "antenna_layout")

def plot_coverage_snapshot(outdir: Path, snapshot_cov: np.ndarray, antenna_grid: np.ndarray, mode: str):
    if snapshot_cov is None:
        print(f"! Skipping coverage snapshot for {mode}: snapshot_cov=None")
        return
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    im = ax.imshow(snapshot_cov, vmin=0.0, vmax=100.0, cmap="RdYlGn")
    add_cell_grid(ax, *snapshot_cov.shape)
    plt.title(f"Coverage Snapshot — {mode}")
    plt.xticks([]); plt.yticks([])

    ant_rc = np.argwhere(antenna_grid > 0)
    if len(ant_rc):
        ant_x = ant_rc[:, 1]
        ant_y = ant_rc[:, 0]
        counts = antenna_grid[ant_y, ant_x].astype(int)

        s_base = 45
        s_step = 35
        s_max  = 350
        sizes = (s_base + s_step * (counts - 1)).clip(s_base, s_max)

        plt.scatter(
            ant_x, ant_y,
            s=sizes,
            marker='^',
            facecolors='tab:blue',
            edgecolors='white',
            linewidths=1.2,
            alpha=0.95,
            zorder=4,
            label="Antennas (size ∝ count)"
        )

        example_counts = [1, 2, 3]
        example_sizes = (s_base + s_step * (np.array(example_counts) - 1)).clip(s_base, s_max)
        legend_handles = [
            plt.scatter([], [], s=example_sizes[i], marker='^',
                        facecolors='tab:blue', edgecolors='white', linewidths=1.2)
            for i in range(len(example_counts))
        ]
        plt.legend(
            legend_handles,
            [f"{c} antenna" if c == 1 else (f"{c} antennas" if c < 3 else "3+ antennas") for c in example_counts],
            loc="lower right",
            framealpha=0.65
        )

    cbar = plt.colorbar(im); cbar.set_label("Coverage % (per cell)")
    savefig(outdir, "coverage_snapshot")

def plot_failures_redirects(outdir: Path, fails: np.ndarray, reds: np.ndarray, mode: str):
    t = np.arange(len(fails))
    plt.figure(figsize=(9,4))
    plt.plot(t, fails, lw=1.2, label="Failures/hour")
    plt.plot(t, reds,  lw=1.0, label="Redirects/hour")
    plt.xlabel("Hour"); plt.ylabel("Count")
    plt.title(f"Failures & Redirects — 1 Year ({mode})")
    plt.grid(True, ls="--", alpha=0.3); plt.legend()
    savefig(outdir, "time_failures_redirects")

def plot_capacity_util(outdir: Path, capacity_util: np.ndarray, mode: str):
    plt.figure(figsize=(6,3.8))
    plt.hist(capacity_util, bins=25)
    plt.xlabel("Load / Capacity"); plt.ylabel("Cell count")
    plt.title(f"Capacity Utilization — 1 Year ({mode})\n(Only cells with antennas)")
    savefig(outdir, "capacity_util_hist")

def plot_user_density(outdir: Path, user_density_avg: np.ndarray, mode: str):
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    im = ax.imshow(user_density_avg, cmap="YlOrRd")
    add_cell_grid(ax, *user_density_avg.shape)
    plt.title(f"Average User Density — 1 Year ({mode})")
    plt.xticks([]); plt.yticks([])
    cbar = plt.colorbar(im); cbar.set_label("Avg users per cell")
    savefig(outdir, "avg_user_density")

def plot_capacity_util_boxplot(outdir: Path, by_mode: dict[str, np.ndarray]):
    ensure_dir(outdir)
    labels = []
    data = []
    for key in ["uniform", "attraction"]:
        if key in by_mode and len(by_mode[key]) > 0:
            labels.append(key)
            data.append(by_mode[key])

    if not data:
        print("! Skipping capacity_util_boxplot: no data.")
        return

    plt.figure(figsize=(7.5, 4.6))
    bp = plt.boxplot(
        data, labels=labels, showmeans=True, showfliers=False,
        widths=0.5, patch_artist=True
    )
    colors = ["#7db1ff", "#ffa760"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel("Load / Capacity")
    plt.title("Capacity Utilization — Boxplot (only cells with antennas)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    savefig(outdir, "capacity_util_boxplot")

# ==================== Before/After ====================
def plot_before_after_yearavg(outdir: Path, env_best: CellularNetworkEnv, best_cell_cov_avg: np.ndarray, mode: str):
    baseline_grid = random_baseline_grid_like(env_best.antenna_grid)
    env_base = CellularNetworkEnv(
        rows=env_best.rows, cols=env_best.cols, total_users=env_best.total_users,
        antenna_capacity=env_best.antenna_capacity, time_step=env_best.time_step,
        movement_mode=env_best.movement_mode
    )
    env_base.antenna_grid = baseline_grid
    env_base.seed(SEED); env_base.place_users()
    base_stats = simulate_year(env_base)
    cov_base_avg = base_stats["cell_cov_avg"]
    cov_best_avg = best_cell_cov_avg

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.04], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im0 = ax0.imshow(cov_base_avg, vmin=0.0, vmax=100.0, cmap="RdYlGn")
    add_cell_grid(ax0, *cov_base_avg.shape)
    ax0.set_title("Random baseline — YEAR AVG")
    ax0.set_xticks([]); ax0.set_yticks([])

    im1 = ax1.imshow(cov_best_avg, vmin=0.0, vmax=100.0, cmap="RdYlGn")
    add_cell_grid(ax1, *cov_best_avg.shape)
    ax1.set_title("Best grid — YEAR AVG")
    ax1.set_xticks([]); ax1.set_yticks([])

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label("Coverage % (per cell)")
    savefig(outdir, "before_after_yearavg")

def _one_hour_snapshot_cov(grid: np.ndarray, movement_mode: str) -> np.ndarray:
    env = CellularNetworkEnv(rows=grid.shape[0], cols=grid.shape[1],
                             total_users=TOTAL_USERS, antenna_capacity=ANT_CAP,
                             time_step=TIME_STEP_MIN, movement_mode=movement_mode)
    env.antenna_grid = grid.copy()
    env.seed(SEED); env.place_users()
    env.move_users_markov_chain()
    cov, _, _ = env.check_coverage()
    return cov

def plot_before_after_snapshot(outdir: Path, env_best: CellularNetworkEnv, mode: str):
    baseline_grid = random_baseline_grid_like(env_best.antenna_grid)
    cov_base = _one_hour_snapshot_cov(baseline_grid, env_best.movement_mode)
    cov_best = _one_hour_snapshot_cov(env_best.antenna_grid, env_best.movement_mode)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.04], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); cax = fig.add_subplot(gs[0, 2])

    im0 = ax0.imshow(cov_base, vmin=0.0, vmax=100.0, cmap="RdYlGn")
    add_cell_grid(ax0, *cov_base.shape)
    ax0.set_title("Random baseline — SNAPSHOT")
    ax0.set_xticks([]); ax0.set_yticks([])

    im1 = ax1.imshow(cov_best, vmin=0.0, vmax=100.0, cmap="RdYlGn")
    add_cell_grid(ax1, *cov_best.shape)
    ax1.set_title("Best grid — SNAPSHOT")
    ax1.set_xticks([]); ax1.set_yticks([])

    cbar = fig.colorbar(im1, cax=cax); cbar.set_label("Coverage % (per cell)")
    savefig(outdir, "before_after_snapshot")

def plot_ga_progress_from_log(outdir: Path, log_path: Path, mode: str):
    resolved = _resolve_log_path(log_path)
    if resolved is None:
        print(f"! Skipping reward plot for {mode}: GA log missing -> {log_path}")
        return
    log_path = resolved

    ga_src = ""
    try:
        if GA_CODE_PATH.exists():
            ga_src = GA_CODE_PATH.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    final_window = _parse_final_eval_window_from_ga(ga_src) or 50
    reinf_start  = _parse_density_boost_start_from_ga(ga_src) or 50

    gens, rewards, antennas, total_gens = _parse_ga_log(log_path)
    if not gens:
        print(f"! No GA data parsed from {log_path} → skipping.")
        return
    x_min, x_max = min(gens), max(gens)
    total_for_window = total_gens or x_max
    one_year_eval_start = max(1, total_for_window - final_window)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(gens, rewards, linewidth=2.0, label="Best Reward")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(gens, antennas, linewidth=1.8, linestyle="--", label="Antenna Count")
    ax2.set_ylabel("Antennas")

    if reinf_start > x_min:
        ax1.axvspan(x_min, reinf_start, color='gray', alpha=0.08,
                    label=f"Phase 1 — Early GA search (0–{reinf_start}, 3 mo, 2190h)")
        ax1.axvline(reinf_start, color="gray", linestyle="--", linewidth=2.0)

    left = max(x_min, reinf_start)
    mid  = max(left, one_year_eval_start)
    if mid > left:
        ax1.axvspan(left, mid, color='gold', alpha=0.10,
                    label=f"Phase 2 — GA + reinforcement ({left}–{mid}, 3 mo, 2190h)")

    if one_year_eval_start < x_max:
        ax1.axvspan(one_year_eval_start, x_max, color='red', alpha=0.06,
                    label=f"Phase 3 — Final 1-year eval ({one_year_eval_start}–{x_max}, 1 yr, 8760h)")
        ax1.axvline(one_year_eval_start, color="red", linestyle="--", linewidth=2.0)

    gx, gy = _nearest_xy(gens, rewards, reinf_start)
    if gy is not None:
        ax1.annotate("↳ Reinforcement starts",
                     xy=(gx, gy),
                     xytext=(gx + max(2, (x_max - x_min) * 0.03), gy - 0.05),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color="gray"),
                     fontsize=10, color="gray", fontweight="bold")

    gx2, gy2 = _nearest_xy(gens, rewards, one_year_eval_start)
    if gy2 is not None:
        ax1.annotate("↳ 1-Year eval window",
                     xy=(gx2, gy2),
                     xytext=(gx2 + max(2, (x_max - x_min) * 0.03), gy2 - 0.05),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color="red"),
                     fontsize=10, color="red", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    seen = set(); lines = []; labels = []
    for L, lab in list(zip(lines1 + lines2, labels1 + labels2)):
        if lab and lab not in seen:
            lines.append(L); labels.append(lab); seen.add(lab)
    fig.legend(lines, labels, loc="upper right")

    plt.title("Genetic Algorithm Progress — Reward & Antenna Count", fontweight="bold")
    plt.tight_layout()
    savefig(outdir, "reward_decomposition")

# ==================== Summary coverage charts ====================
def plot_monthly_coverage(outdir: Path, monthly_cov: np.ndarray, mode: str):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    plt.figure(figsize=(7.8,3.6))
    plt.bar(months, monthly_cov)
    lo, hi = cov_limits(monthly_cov)
    plt.ylim(lo, hi); plt.ylabel("Avg userwise coverage (%)")
    plt.title(f"Monthly Userwise Coverage — 1 Year ({mode})")
    savefig(outdir, "monthly_coverage_bars")

def plot_per_cell_coverage(outdir: Path, cell_cov_avg: np.ndarray, mode: str):
    vals = cell_cov_avg.flatten()
    x = np.arange(len(vals))

    plt.figure(figsize=(10, 3.8))
    plt.bar(x, vals, width=1.0, color="tab:blue", edgecolor="none")
    plt.ylim(0.0, 100.0)
    plt.xlabel("Cell index")
    plt.ylabel("Avg coverage per cell (%)")
    plt.title(f"Per-Cell Average Coverage — 1 Year ({mode})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    savefig(outdir, "per_cell_coverage_bars")

def plot_overall_avg_coverage(outdir: Path, overall_cov: np.ndarray, fails: np.ndarray, reds: np.ndarray, mode: str):
    t = np.arange(len(overall_cov))
    fig, ax1 = plt.subplots(figsize=(9,4))

    ax1.plot(t, overall_cov, color="tab:green", lw=1.5, label="Userwise coverage (%)")
    lo, hi = cov_limits(overall_cov)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Hour"); ax1.set_ylabel("Userwise coverage (%)", color="tab:green")
    ax1.tick_params(axis='y', labelcolor="tab:green")
    ax1.grid(True, ls="--", alpha=0.3)

    ax2 = ax1.twinx()
    fail_rate = 100.0 * (fails.astype(float) / max(TOTAL_USERS, 1))
    red_rate  = 100.0 * (reds.astype(float)  / max(TOTAL_USERS, 1))
    l1, = ax2.plot(t, fail_rate, color="tab:red",  lw=1.2, ls="--", label="Failure rate (%)")
    l2, = ax2.plot(t, red_rate,  color="tab:orange", lw=1.2, ls=":",  label="Redirect rate (%)")
    ax2.set_ylabel("Failure / Redirect rate (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = [l1, l2], ["Failure rate (%)", "Redirect rate (%)"]
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"Overall Coverage + Failure/Redirect Rates — 1 Year ({mode})")
    plt.tight_layout()
    savefig(outdir, "overall_avg_coverage_line")

# ============================ Main ============================
if __name__ == "__main__":
    capacity_util_by_mode: dict[str, np.ndarray] = {}
    for mode, grid_path in GRIDS.items():
        if not grid_path.exists():
            print(f"! Skipping {mode}: grid file missing -> {grid_path}")
            continue

        outdir = Path(f"report_{mode}")
        ensure_dir(outdir)

        best_grid = np.load(grid_path)
        env = build_env(best_grid, mode)

        stats = simulate_year(env)

        capacity_util_by_mode[mode] = stats["capacity_util"]

        plot_antenna_layout(outdir, best_grid, mode)
        plot_coverage_snapshot(outdir, stats["snapshot_cov"], best_grid, mode)
        plot_failures_redirects(outdir, stats["fails"], stats["reds"], mode)
        plot_capacity_util(outdir, stats["capacity_util"], mode)
        plot_user_density(outdir, stats["user_density_avg"], mode)

        plot_before_after_yearavg(outdir, env, stats["cell_cov_avg"], mode)
        plot_before_after_snapshot(outdir, env, mode)

        plot_monthly_coverage(outdir, stats["monthly_cov"], mode)
        plot_per_cell_coverage(outdir, stats["cell_cov_avg"], mode)
        plot_overall_avg_coverage(outdir, stats["overall_cov"], stats["fails"], stats["reds"], mode)

        log_path = GA_LOGS.get(mode)
        if log_path is not None:
            plot_ga_progress_from_log(outdir, log_path, mode)

    compare_dir = Path("report_compare")
    plot_capacity_util_boxplot(compare_dir, capacity_util_by_mode)

    print("\nDone. Check ./report_uniform, ./report_attraction, and ./report_compare")
