# cells_gui.py — lean Tkinter GUI for CELLS project
# --------------------------------------------------
# • auto‑sized window (no blank strip on the right)
# • Simulation / Genetic Alg. / Plot tabs
# • Init Env + Reset Env side‑by‑side
# • GA settings include reinforcement‑start generation + optional per‑generation log
# • Build‑in buttons to *load* any antenna‑grid .npy and any GA progress log (.json or .txt)
#   and visualise coverage or training curves on demand.
# • Satellite‑dish emoji PNG (antenna.png) used as the logo.

from __future__ import annotations

import json
import os
import sys
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk  # pip install pillow

# ────────────────────────── local imports ────────────────────────────
try:
    from updated_grid_env import CellularNetworkEnv  # type: ignore
except ImportError:
    # add project root (one level up) to PYTHONPATH and retry
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from updated_grid_env import CellularNetworkEnv  # type: ignore

try:
    from ga_fast_parallel_cpu import run_ga  # type: ignore
except Exception:
    run_ga = None  # GA tab will show a warning instead

# --- constants -------------------------------------------------------
GUI_DIR = Path(__file__).resolve().parent
LOGO_FILE = GUI_DIR / "cells.png"
LOGO_SIZE = 256  # square logo size (px)

###############################################################################
class CellsGUI(tk.Tk):
    """Main window."""

    # ───────────────────────────── init ──────────────────────────────
    def __init__(self) -> None:
        super().__init__()
        self.title("CELLS — Antenna Optimiser")
        self.resizable(False, False)

        # logo on the left
        self._build_logo()

        # notebook on the right
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=0, column=1, sticky="nsew", padx=(6, 6), pady=(6, 6))

        self.env: CellularNetworkEnv | None = None
        self.loaded_grid_path: Path | None = None
        self.loaded_log_path: Path | None = None

        self.seed_var = tk.IntVar(value=42)

        # tabs
        self.sim_frame = ttk.Frame(self.nb)
        self.ga_frame = ttk.Frame(self.nb)
        self.plot_frame = ttk.Frame(self.nb)

        self._build_sim_tab()
        self._build_ga_tab()
        self._build_plot_tab()

        self.nb.add(self.sim_frame, text="Simulation")
        if run_ga is not None:
            self.nb.add(self.ga_frame, text="Genetic Alg.")
        self.nb.add(self.plot_frame, text="Plot")

        # final window size = logo width + notebook reqwidth
        # ── auto-scale GUI window to fit tallest tab ──
        self.update_idletasks()
        width = LOGO_SIZE + self.nb.winfo_reqwidth() + 24
        height = max(LOGO_SIZE + 20, self.nb.winfo_reqheight() + 16)
        self.geometry(f"{width}x{height}")

    # ──────────────────────────── logo ───────────────────────────────
    def _build_logo(self) -> None:
        if not LOGO_FILE.exists():
            tk.Label(self, text="CELLS", font=("Helvetica", 28, "bold")).grid(row=0, column=0, padx=6, pady=6)
            return

        # ── big sidebar logo ─────────────────────────────────────
        img = Image.open(LOGO_FILE).convert("RGBA").resize((LOGO_SIZE, LOGO_SIZE), Image.LANCZOS)
        self._logo_photo = ImageTk.PhotoImage(img)
        tk.Label(self, image=self._logo_photo).grid(row=0, column=0, padx=6, pady=6)

        # ── NEW: tiny window-icon for title-bar / task-bar ───────
        small_path = GUI_DIR / "antenna.png"      # put a 32×32 PNG here
        if small_path.exists():
            small_img = Image.open(small_path).convert("RGBA").resize((32, 32), Image.LANCZOS)
            self._icon_photo = ImageTk.PhotoImage(small_img)  # keep reference!
            self.iconphoto(True, self._icon_photo)            # register icon

    # ───────────────────── simulation tab ────────────────────────────
    def _build_sim_tab(self) -> None:
        f = self.sim_frame
        pad = {"padx": 4, "pady": 2}

        self.rows_var = tk.IntVar(value=20)
        self.cols_var = tk.IntVar(value=20)
        self.users_var = tk.IntVar(value=50_000)
        self.cap_var = tk.IntVar(value=300)
        self.step_size_var = tk.IntVar(value=60)
        self.seed_var = tk.IntVar(value=42)
        self.move_mode_var = tk.StringVar(value="uniform")
        self.student_ratio_var = tk.DoubleVar(value=0.5)

        ttk.Label(f, text="Rows").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=4, to=100, width=6, textvariable=self.rows_var).grid(row=0, column=1, **pad)

        ttk.Label(f, text="Cols").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=4, to=100, width=6, textvariable=self.cols_var).grid(row=1, column=1, **pad)

        ttk.Label(f, text="Total users").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=1000, to=1_000_000, increment=1000, width=8, textvariable=self.users_var).grid(row=2, column=1, **pad)

        ttk.Label(f, text="Antenna capacity").grid(row=3, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=10, to=1000, width=6, textvariable=self.cap_var).grid(row=3, column=1, **pad)

        ttk.Label(f, text="Step size (min)").grid(row=4, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=5, to=240, increment=5, width=7, textvariable=self.step_size_var).grid(row=4, column=1, **pad)

        ttk.Label(f, text="Global seed").grid(row=5, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=0, to=99999, width=7, textvariable=self.seed_var).grid(row=5, column=1, **pad)

        ttk.Label(f, text="Movement mode").grid(row=6, column=0, sticky="w", **pad)
        move_combo = ttk.Combobox(f, values=["uniform", "attraction"], state="readonly", width=10, textvariable=self.move_mode_var)
        move_combo.grid(row=6, column=1, **pad)
        move_combo.bind("<<ComboboxSelected>>", lambda _: self._toggle_student_bar())

        # Student ratio + zone widgets (initialised but shown only in attraction mode)
        self.student_ratio_label = ttk.Label(f, text="Student ratio")
        self.student_ratio_canvas = tk.Canvas(f, width=100, height=10)
        self.student_ratio_bar = self.student_ratio_canvas.create_rectangle(0, 0, 50, 10, fill="darkorange")
        self.worker_ratio_bar = self.student_ratio_canvas.create_rectangle(50, 0, 100, 10, fill="navy")

        self.student_ratio_slider = ttk.Scale(
            f, from_=0.0, to=1.0,
            variable=self.student_ratio_var,
            command=self._update_student_bar
        )
        self.student_ratio_spin = ttk.Spinbox(
            f, from_=0.0, to=1.0, increment=0.05, format="%.2f",
            width=6, textvariable=self.student_ratio_var,
            command=lambda: self._update_student_bar()
        )
        self.student_ratio_var.trace_add("write", lambda *_, **__: self._update_student_bar())

        self.zone_input_label = ttk.Label(f, text="Attraction zones (label: (r,c),...)")
        self.zone_input_text = tk.Text(f, width=20, height=4)

        # Buttons
        ttk.Button(f, text="Init Env", command=self.init_env).grid(row=11, column=0, sticky="ew", **pad)
        ttk.Button(f, text="Reset Env", command=self.reset_env).grid(row=11, column=1, sticky="ew", **pad)

        self._toggle_student_bar()


    def _toggle_student_bar(self):
        show = self.move_mode_var.get() == "attraction"
        widgets = [
            (self.student_ratio_label, 7, 0),
            (self.student_ratio_canvas, 7, 1),
            (self.student_ratio_slider, 8, 0),
            (self.student_ratio_spin, 8, 1),
            (self.zone_input_label, 9, 0),
            (self.zone_input_text, 9, 1),
        ]
        for w, r, c in widgets:
            if show:
                w.grid(row=r, column=c, padx=(4, 12), pady=(4, 6), sticky="ew")
            else:
                w.grid_remove()

        self._update_student_bar()

        # Resize GUI to fit input field
        self.update_idletasks()
        base_width = LOGO_SIZE + self.nb.winfo_reqwidth() + 24
        self.geometry(f"{base_width}x{LOGO_SIZE + (130 if show else 20)}")


    def _update_student_bar(self, *args):
        s_ratio = self.student_ratio_var.get()
        total_width = 100
        self.student_ratio_canvas.coords(self.student_ratio_bar, 0, 0, total_width * s_ratio, 10)
        self.student_ratio_canvas.coords(self.worker_ratio_bar, total_width * s_ratio, 0, total_width, 10)


    def _parse_zone_input(self) -> dict[str, list[tuple[int, int]]]:
        text = self.zone_input_text.get("1.0", tk.END).strip()
        zones: dict[str, list[tuple[int, int]]] = {}
        try:
            for line in text.splitlines():
                if ":" not in line:
                    continue
                label, coords = line.split(":", 1)
                points = []
                for pair in coords.split(","):
                    pair = pair.strip()
                    if pair.startswith("(") and pair.endswith(")"):
                        nums = pair[1:-1].split(",")
                        if len(nums) == 2:
                            points.append((int(nums[0]), int(nums[1])))
                if points:
                    zones[label.strip()] = points
        except Exception as e:
            messagebox.showerror("Zone Input Error", f"Failed to parse zones:\n{e}")
            zones.clear()
        return zones



    # ───────────────────── GA tab ────────────────────────────────────
    def _build_ga_tab(self) -> None:
        if run_ga is None:
            ttk.Label(self.ga_frame, text="GA script not importable – install requirements.").pack(padx=6, pady=6)
            return
        f = self.ga_frame
        pad = {"padx": 4, "pady": 2}

        self.pop_var = tk.IntVar(value=os.cpu_count() or 12)
        self.gen_var = tk.IntVar(value=250)
        self.mut_var = tk.DoubleVar(value=0.1)
        self.reinf_var = tk.IntVar(value=50)
        self.log_enable_var = tk.BooleanVar(value=True)

        ttk.Label(f, text="Population").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=4, to=128, width=7, textvariable=self.pop_var).grid(row=0, column=1, **pad)

        ttk.Label(f, text="Generations").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=10, to=500, width=7, textvariable=self.gen_var).grid(row=1, column=1, **pad)

        ttk.Label(f, text="Mutation rate").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=0.01, to=0.5, increment=0.01, format="%.2f", width=7, textvariable=self.mut_var).grid(row=2, column=1, **pad)

        ttk.Label(f, text="Reinf. start gen").grid(row=3, column=0, sticky="w", **pad)
        ttk.Spinbox(f, from_=0, to=400, width=7, textvariable=self.reinf_var).grid(row=3, column=1, **pad)

        ttk.Checkbutton(f, text="Save per‑generation log", variable=self.log_enable_var).grid(row=4, column=0, columnspan=2, sticky="w", **pad)

        ttk.Button(f, text="Run GA", command=self.run_ga_thread).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 4), padx=4)

    # ───────────────────── plot tab ──────────────────────────────────
    def _build_plot_tab(self) -> None:
        f = self.plot_frame
        pad_bt = {"padx": 6, "pady": 3, "sticky": "ew"}

        ttk.Button(f, text="User movement (current env)", command=self.plot_users).grid(**pad_bt)
        ttk.Button(f, text="Coverage (current env)", command=self.plot_cov_current).grid(**pad_bt)

        ttk.Separator(f, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)

        ttk.Button(f, text="Load antenna grid (.npy)", command=self.choose_grid_file).grid(**pad_bt)
        ttk.Button(f, text="Coverage (selected grid)", command=self.plot_cov_loaded).grid(**pad_bt)

        ttk.Separator(f, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)

        ttk.Button(f, text="Load GA log (.json/.txt)", command=self.choose_log_file).grid(**pad_bt)
        ttk.Button(f, text="Plot GA progress", command=self.plot_ga_progress).grid(**pad_bt)

    # ───────────────────── env helpers ───────────────────────────────
    def init_env(self) -> None:
        seed = self.seed_var.get()
        np.random.seed(seed)
        random.seed(seed)

        zones = self._parse_zone_input() if self.move_mode_var.get() == "attraction" else None

        self.env = CellularNetworkEnv(
            rows=self.rows_var.get(),
            cols=self.cols_var.get(),
            total_users=self.users_var.get(),
            antenna_capacity=self.cap_var.get(),
            time_step=self.step_size_var.get(),
            movement_mode=self.move_mode_var.get(),
            student_ratio=self.student_ratio_var.get(),
            zones=zones,
        )           

        messagebox.showinfo("Env", f"Environment initialised with seed {seed}.")

    def reset_env(self) -> None:
        if self.env is None:
            messagebox.showwarning("Env", "Init the environment first!")
            return
        self.env.reset()
        messagebox.showinfo("Env", "Environment reset.")

    # ───────────────────── plot helpers ──────────────────────────────
    def plot_users(self) -> None:
        if self.env is None:
            self.init_env()
        threading.Thread(target=self.env.animate_user_grid, kwargs={"steps": 120, "interval": 100}).start()

    def plot_cov_current(self) -> None:
        if self.env is None:
            self.init_env()
        threading.Thread(target=self.env.animate_coverage_quality, kwargs={"steps": 240, "interval": 100}).start()

    # ---- grid file selection & coverage ----
    def choose_grid_file(self) -> None:
        path = filedialog.askopenfilename(title="Select antenna grid .npy", filetypes=[("NumPy array", "*.npy")])
        if path:
            self.loaded_grid_path = Path(path)
            messagebox.showinfo("Grid", f"Loaded grid file:\n{path}")

    def plot_cov_loaded(self) -> None:
        if self.loaded_grid_path is None or not self.loaded_grid_path.exists():
            messagebox.showwarning("Grid", "No grid file loaded. Use 'Load antenna grid' first.")
            return
        best_grid = np.load(self.loaded_grid_path)
        env = CellularNetworkEnv(
            rows=best_grid.shape[0],
            cols=best_grid.shape[1],
            total_users=self.users_var.get(),
            antenna_capacity=self.cap_var.get(),
            movement_mode=self.move_mode_var.get(),
        )
        env.antenna_grid = best_grid
        env.place_users()
        threading.Thread(target=env.animate_coverage_quality, kwargs={"steps": 240, "interval": 100}).start()

    # ---- GA log selection & plotting ----
    def choose_log_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select GA progress log",
            filetypes=[("JSON", "*.json"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.loaded_log_path = Path(path)
            messagebox.showinfo("Log", f"Loaded log file:\n{path}")

    def plot_ga_progress(self) -> None:
        if self.loaded_log_path is None or not self.loaded_log_path.exists():
            messagebox.showwarning("Log", "No log file loaded. Use 'Load GA log' first.")
            return

        import matplotlib.pyplot as plt  # local import to keep GUI snappy

        gens: list[int] = []
        rewards: list[float] = []
        antennas: list[int] = []

        if self.loaded_log_path.suffix.lower() == ".json":
            try:
                with open(self.loaded_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rec in data:
                    gens.append(rec.get("generation") or rec.get("gen") or len(gens) + 1)
                    rewards.append(rec.get("reward") or rec.get("fitness"))
                    antennas.append(rec.get("antennas", 0))
            except Exception as e:
                messagebox.showerror("Log", f"Failed to parse JSON log:\n{e}")
                return
        else:  # assume plain text like ga_terminal_log.txt
            import re

            gen_pat = re.compile(r"Generation\s+(\d+)")
            reward_pat = re.compile(r"reward this gen:\s*([-+]?[0-9]*\.?[0-9]+)")
            ant_pat = re.compile(r"Antennas used:\s*(\d+)")

            current_gen = None
            with open(self.loaded_log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if (m := gen_pat.search(line)):
                        current_gen = int(m.group(1))
                    if current_gen is not None and (m2 := reward_pat.search(line)) and (m3 := ant_pat.search(line)):
                        gens.append(current_gen)
                        rewards.append(float(m2.group(1)))
                        antennas.append(int(m3.group(1)))
                        current_gen = None

        if not gens:
            messagebox.showerror("Log", "No data parsed from selected log file.")
            return

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(gens, rewards, label="Best reward", linewidth=2)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Reward")
        ax1.grid(True, linestyle="--", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(gens, antennas, "--", label="Antenna count", color="tab:orange")
        ax2.set_ylabel("Antennas")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        plt.title("GA progress — reward & antenna count")
        plt.tight_layout()
        plt.show()

    # ───────────────────── GA runner ────────────────────────────────
    def run_ga_thread(self) -> None:
        if run_ga is None:
            return

        def _run():
            try:
                run_ga(
                    pop_size=self.pop_var.get(),
                    generations=self.gen_var.get(),
                    base_mutation_rate=self.mut_var.get(),
                    density_boost_start_gen=self.reinf_var.get(),
                    logging=self.log_enable_var.get(),
                )
            except TypeError:
                # backward‑compatibility: fall back to original signature
                run_ga()
            except Exception as e:
                messagebox.showerror("GA", f"Error while running GA:\n{e}")
                return
            messagebox.showinfo("GA", "GA run complete — check output files.")

        threading.Thread(target=_run, daemon=True).start()

###############################################################################
if __name__ == "__main__":
    CellsGUI().mainloop()
