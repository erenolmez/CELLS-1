# cells_gui.py — lean Tkinter GUI for CELLS project
# --------------------------------------------------
# • auto-sized window (no blank strip on the right)
# • Simulation / Genetic Alg. / Plot tabs
# • Init Env + Reset Env side-by-side
# • GA settings include reinforcement-start generation + optional per-generation log
# • Build-in buttons to *load* any antenna-grid .npy and any GA progress log (.json or .txt)
#   and visualise coverage or training curves on demand.
# • Satellite-dish emoji PNG (antenna.png) used as the logo.

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
import csv  # CSV support for GA logs

# ────────────────────────── local imports ────────────────────────────
try:
    from updated_grid_env import CellularNetworkEnv  # type: ignore
except ImportError:
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

    def __init__(self) -> None:
        super().__init__()
        self.title("CELLS — Antenna Optimiser")
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

        # flag for default zone pre-load
        self.zones_loaded_default = False  # used in _toggle_student_bar

        # tabs
        self.sim_frame = ttk.Frame(self.nb)
        self.ga_frame = ttk.Frame(self.nb)
        self.plot_frame = ttk.Frame(self.nb)

        self._build_sim_tab()
        self._build_ga_tab()
        self._build_plot_tab()

        # Call toggle before sizing to ensure proper initial size based on default movement mode
        self._toggle_student_bar()

        self.nb.add(self.sim_frame, text="Simulation")
        if run_ga is not None:
            self.nb.add(self.ga_frame, text="Genetic Alg.")
        self.nb.add(self.plot_frame, text="Plot")

        # final window size = logo width + notebook reqwidth
        self.update_idletasks()
        width = LOGO_SIZE + 24 + self.nb.winfo_reqwidth()
        height = LOGO_SIZE + 40  # your exact original init height
        self.geometry(f"{width}x{height}")
    # ──────────────────────────── logo ───────────────────────────────
    def _build_logo(self) -> None:
        if not LOGO_FILE.exists():
            tk.Label(self, text="CELLS", font=("Helvetica", 28, "bold")).grid(row=0, column=0, padx=6, pady=6)
            return

        img = Image.open(LOGO_FILE).convert("RGBA").resize((LOGO_SIZE, LOGO_SIZE), Image.LANCZOS)
        self._logo_photo = ImageTk.PhotoImage(img)

        # Create a fixed-size frame to hold the logo, with height = LOGO_SIZE + 40 for vertical centering
        frame_height = LOGO_SIZE + 40
        self.logo_frame = ttk.Frame(self, width=LOGO_SIZE + 12, height=frame_height)
        self.logo_frame.grid(row=0, column=0, sticky="ns")  # fill vertically, no padding here

        # Calculate vertical offset to center logo inside frame
        y_offset = (frame_height - LOGO_SIZE) // 2

        # Place the logo label inside the frame using place for pixel control and vertical centering
        self.logo_label = tk.Label(self.logo_frame, image=self._logo_photo)
        self.logo_label.place(x=6, y=y_offset)

        small_path = GUI_DIR / "antenna.png"
        if small_path.exists():
            small_img = Image.open(small_path).convert("RGBA").resize((32, 32), Image.LANCZOS)
            self._icon_photo = ImageTk.PhotoImage(small_img)
            self.iconphoto(True, self._icon_photo)

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

        # --- Interactive zone UI ---
        self.zone_label_var = tk.StringVar(value="home")
        zone_opts = ["home", "school", "work", "city center"]
        self.zone_cells = {zone: set() for zone in zone_opts}
        self.zone_colors = {
            "home": "tomato", "school": "dodgerblue",
            "work": "gold", "city center": "mediumseagreen"
        }
        self.zone_canvas_cells = {}

        self.zone_label_dropdown = ttk.Combobox(f, values=zone_opts, textvariable=self.zone_label_var, state="readonly", width=12)
        self.zone_canvas = tk.Canvas(f, width=200, height=200, bg="white", highlightthickness=1, highlightbackground="gray")
        self.zone_canvas.bind("<Button-1>", self._on_zone_click)
        self.zone_canvas.bind("<Motion>", self._on_hover_zone)

        self.hover_label = ttk.Label(f, text="")
        self.student_ratio_label = ttk.Label(f, text="Student/Worker ratio")
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

        # Instantiate buttons but do NOT grid them here
        self.clear_zones_btn = ttk.Button(f, text="Clear All Zones", command=self._clear_all_zones)
        self.reset_zones_btn = ttk.Button(f, text="Reset Zones to Default", command=self._reset_zones_default)

        # Grid main control buttons
        ttk.Button(f, text="Init Env", command=self.init_env).grid(row=14, column=0, sticky="ew", **pad)
        ttk.Button(f, text="Reset Env", command=self.reset_env).grid(row=14, column=1, sticky="ew", **pad)

        self._toggle_student_bar()  # draw conditional UI

    def _toggle_student_bar(self):
        show = self.move_mode_var.get() == "attraction"

        widgets = [
            self.student_ratio_label, self.student_ratio_canvas,
            self.student_ratio_slider, self.student_ratio_spin,
            self.zone_label_dropdown, self.zone_canvas,
            self.hover_label,
        ]

        zone_buttons = [self.clear_zones_btn, self.reset_zones_btn]

        for w in widgets + zone_buttons:
            w.grid_remove()

        if show:
            self.student_ratio_label.grid(row=7, column=0, padx=(4, 12), pady=(4, 6), sticky="w")
            self.student_ratio_canvas.grid(row=7, column=1, padx=(4, 12), pady=(4, 6), sticky="ew")
            self.student_ratio_slider.grid(row=8, column=0, padx=4, pady=2, sticky="ew")
            self.student_ratio_spin.grid(row=8, column=1, padx=4, pady=2, sticky="ew")

            self.zone_label_dropdown.grid(row=9, column=0, padx=4, pady=(6, 2), sticky="w")
            self.zone_canvas.grid(row=10, column=0, columnspan=2, padx=6, pady=2)
            self.hover_label.grid(row=11, column=0, columnspan=2, sticky="ew")

            self.clear_zones_btn.grid(row=12, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 2))
            self.reset_zones_btn.grid(row=13, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 4))

            if not self.zones_loaded_default:
                self._load_default_attraction_zones()
                self.zones_loaded_default = True

            self.zone_canvas.after(100, self._draw_zone_grid)

        self._update_student_bar()
        self.update_idletasks()

        base_width = LOGO_SIZE + self.nb.winfo_reqwidth() + 24

        # Adjust logo frame height and logo vertical position based on mode
        if show:  # attraction mode
            frame_height = LOGO_SIZE + 360
            self.logo_frame.configure(height=frame_height)
            y_offset = (frame_height - LOGO_SIZE) // 2
            self.logo_label.place(x=6, y=y_offset)
        else:
            frame_height = LOGO_SIZE + 40  # match init height
            self.logo_frame.configure(height=frame_height)
            y_offset = (frame_height - LOGO_SIZE) // 2
            self.logo_label.place(x=6, y=y_offset)

        self.geometry(f"{base_width}x{frame_height}")

    def _update_student_bar(self, *args):
        s_ratio = self.student_ratio_var.get()
        total_width = 100
        self.student_ratio_canvas.coords(self.student_ratio_bar, 0, 0, total_width * s_ratio, 10)
        self.student_ratio_canvas.coords(self.worker_ratio_bar, total_width * s_ratio, 0, total_width, 10)

    def _draw_zone_grid(self):
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        width = self.zone_canvas.winfo_width()
        height = self.zone_canvas.winfo_height()
        cell_w = width // cols
        cell_h = height // rows
        self.zone_canvas.delete("all")
        self.zone_canvas_cells.clear()
        for r in range(rows):
            for c in range(cols):
                x0 = c * cell_w
                y0 = r * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                rect = self.zone_canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="gray")
                self.zone_canvas_cells[(r, c)] = rect
        for zone, cells in self.zone_cells.items():
            for (r, c) in cells:
                if (r, c) in self.zone_canvas_cells:
                    self.zone_canvas.itemconfig(self.zone_canvas_cells[(r, c)], fill=self.zone_colors[zone])

    def _on_zone_click(self, event):
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        width = self.zone_canvas.winfo_width()
        height = self.zone_canvas.winfo_height()
        c = int(event.x * cols / width)
        r = int(event.y * rows / height)
        if not (0 <= r < rows and 0 <= c < cols):
            return
        label = self.zone_label_var.get()
        rect = self.zone_canvas_cells.get((r, c))
        if rect is None:
            return
        if (r, c) in self.zone_cells[label]:
            self.zone_cells[label].remove((r, c))
            self.zone_canvas.itemconfig(rect, fill="white")
        else:
            self.zone_cells[label].add((r, c))
            self.zone_canvas.itemconfig(rect, fill=self.zone_colors[label])

    def _on_hover_zone(self, event):
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        width = self.zone_canvas.winfo_width()
        height = self.zone_canvas.winfo_height()
        c = int(event.x * cols / width)
        r = int(event.y * rows / height)
        if 0 <= r < rows and 0 <= c < cols:
            self.hover_label.config(text=f"Hovering: ({r}, {c})")
        else:
            self.hover_label.config(text="")

    def _load_default_attraction_zones(self):
        defaults = {
            "home":        [(17, 5), (18, 3), (16, 7)],
            "school":      [(2, 16), (3, 14)],
            "work":        [(2, 3), (3, 5), (1, 4)],
            "city center": [(9, 9), (10, 10), (11, 11)],
        }
        for label, coords in defaults.items():
            if label in self.zone_cells:
                self.zone_cells[label].update(coords)

    def _clear_all_zones(self):
        for label in self.zone_cells:
            self.zone_cells[label].clear()
        self._draw_zone_grid()

    def _reset_zones_default(self):
        self._clear_all_zones()
        self._load_default_attraction_zones()
        self._draw_zone_grid()

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

        self.steps_var = tk.IntVar(value=240)
        self.interval_var = tk.IntVar(value=100)
        self.save_gif_var = tk.BooleanVar(value=False)
        self.overlay_users_var = tk.BooleanVar(value=True)
        self.overlay_antennas_var = tk.BooleanVar(value=True)

        ttk.Label(f, text="Steps").grid(row=0, column=0, sticky="w", padx=6, pady=(4, 2))
        ttk.Spinbox(f, from_=10, to=1000, increment=10, width=8, textvariable=self.steps_var).grid(row=0, column=1, padx=6, pady=(4, 2))

        ttk.Label(f, text="Interval (ms)").grid(row=1, column=0, sticky="w", padx=6, pady=(2, 6))
        ttk.Spinbox(f, from_=50, to=2000, increment=50, width=8, textvariable=self.interval_var).grid(row=1, column=1, padx=6, pady=(2, 6))

        ttk.Button(f, text="User movement (current env)", command=self.plot_users).grid(row=2, column=0, columnspan=2, **pad_bt)
        ttk.Button(f, text="Coverage (current env)", command=self.plot_cov_current).grid(row=3, column=0, columnspan=2, **pad_bt)

        ttk.Separator(f, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Button(f, text="Load antenna grid (.npy)", command=self.choose_grid_file).grid(row=5, column=0, columnspan=2, **pad_bt)
        ttk.Button(f, text="Coverage (selected grid)", command=self.plot_cov_loaded).grid(row=6, column=0, columnspan=2, **pad_bt)

        ttk.Separator(f, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Button(f, text="Load GA log (.json/.txt/.csv)", command=self.choose_log_file).grid(row=8, column=0, columnspan=2, **pad_bt)
        ttk.Button(f, text="Plot GA progress", command=self.plot_ga_progress).grid(row=9, column=0, columnspan=2, **pad_bt)

    # ───────────────────── env helpers ───────────────────────────────
    def init_env(self) -> None:
        seed = self.seed_var.get()
        np.random.seed(seed)
        random.seed(seed)

        zones = {label: list(cells) for label, cells in self.zone_cells.items() if cells}

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
        threading.Thread(target=self.env.animate_user_grid, kwargs={
            "steps": self.steps_var.get(),
            "interval": self.interval_var.get(),
        }).start()

    def plot_cov_current(self) -> None:
        if self.env is None:
            self.init_env()
        threading.Thread(target=self.env.animate_coverage_quality, kwargs={
            "steps": self.steps_var.get(),
            "interval": self.interval_var.get(),
        }).start()

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

        threading.Thread(target=env.animate_coverage_quality, kwargs={
            "steps": self.steps_var.get(),
            "interval": self.interval_var.get(),
            # optionally: "save": True, "save_path": "coverage_selected.gif", "show": True
        }).start()

    # ---- GA log selection & plotting ----
    def choose_log_file(self) -> None:
        ### CHANGED ###
        path = filedialog.askopenfilename(
            title="Select GA progress log",
            filetypes=[
                ("Supported logs", "*.json *.txt *.csv"),
                ("JSON", "*.json"),
                ("Text", "*.txt"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
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

        suffix = self.loaded_log_path.suffix.lower()

        if suffix == ".json":
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

        ### CHANGED ###
        elif suffix == ".csv":
            try:
                with open(self.loaded_log_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        g = row.get("generation") or row.get("gen")
                        r = row.get("reward") or row.get("fitness")
                        a = row.get("antennas") or row.get("antenna_count") or row.get("num_antennas")
                        if g is None or r is None:
                            # skip incomplete lines
                            continue
                        gens.append(int(float(g)))
                        rewards.append(float(r))
                        antennas.append(int(float(a)) if a is not None and a != "" else 0)
            except Exception as e:
                messagebox.showerror("Log", f"Failed to parse CSV log:\n{e}")
                return

        else:  # assume plain text like ga_terminal_log.txt
            import re
            gen_pat = re.compile(r"Generation\s+(\d+)")
            reward_pat = re.compile(r"reward this gen:\s*([-+]?[0-9]*\.?[0-9]+)")
            ant_pat = re.compile(r"Antennas used:\s*(\d+)")

            current_gen = None
            try:
                with open(self.loaded_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if (m := gen_pat.search(line)):
                            current_gen = int(m.group(1))
                        if current_gen is not None and (m2 := reward_pat.search(line)) and (m3 := ant_pat.search(line)):
                            gens.append(current_gen)
                            rewards.append(float(m2.group(1)))
                            antennas.append(int(m3.group(1)))
                            current_gen = None
            except Exception as e:
                messagebox.showerror("Log", f"Failed to parse text log:\n{e}")
                return

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
                # backward-compatibility: fall back to original signature
                run_ga()
            except Exception as e:
                messagebox.showerror("GA", f"Error while running GA:\n{e}")
                return
            messagebox.showinfo("GA", "GA run complete — check output files.")

        threading.Thread(target=_run, daemon=True).start()

###############################################################################
if __name__ == "__main__":
    CellsGUI().mainloop()
