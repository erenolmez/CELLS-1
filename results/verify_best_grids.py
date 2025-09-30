#!/usr/bin/env python3
# verify_best_grids.py
# Compare best_uniform.npy / best_attraction.npy to known-good arrays (exact match).
# Exit code: 0 if all provided files match; 1 otherwise.

from __future__ import annotations
import argparse, sys, hashlib
from pathlib import Path
import numpy as np

# ---------- Expected arrays (from your message) ----------
EXPECTED_UNIFORM = np.array([
    [0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 3, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    [2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1],
    [0, 1, 2, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 1, 0],
    [0, 2, 0, 0, 2, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2],
    [0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 2],
    [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 0, 0, 1, 0, 0, 1]
], dtype=int)

EXPECTED_ATTRACTION = np.array([
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 3, 2, 1, 2, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 0, 1, 0],
    [0, 1, 0, 0, 0, 3, 3, 0, 0, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 1],
    [0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0],
    [2, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [0, 1, 2, 0, 0, 1, 2, 0, 2, 4, 1, 0, 0, 1, 0, 1, 0, 0, 1, 2],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [0, 2, 0, 2, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [1, 2, 1, 1, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 1, 0, 1, 1, 0, 2],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 3, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 2, 1, 2, 3, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 1, 0, 0],
    [1, 1, 0, 2, 0, 2, 0, 1, 0, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1]
], dtype=int)

# ---------- Utils ----------
def sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def compare_grid(npy_path: Path, expected: np.ndarray, label: str) -> bool:
    if not npy_path.exists():
        print(f"[{label}] ✖ file not found: {npy_path}")
        return False

    got = np.load(npy_path)
    got = np.asarray(got, dtype=int)

    ok = True
    if got.shape != expected.shape:
        print(f"[{label}] ✖ shape mismatch: got {got.shape}, expected {expected.shape}")
        ok = False

    diff = got - expected
    mismatches = np.argwhere(diff != 0)
    if mismatches.size > 0:
        ok = False
        print(f"[{label}] ✖ values differ at {mismatches.shape[0]} cells "
              f"(sum(got)={got.sum()}, sum(exp)={expected.sum()})")
        # show first few locations
        for (r, c) in mismatches[:20]:
            print(f"    at ({r},{c}): got {got[r,c]} vs exp {expected[r,c]}")
        if mismatches.shape[0] > 20:
            print(f"    ... and {mismatches.shape[0]-20} more")
    else:
        print(f"[{label}] ✓ exact match "
              f"(shape {got.shape}, antennas={got.sum()}, sha256={sha256(got)[:12]}…)")

    return ok

def main():
    ap = argparse.ArgumentParser(description="Verify best grid NPYs against known-good arrays.")
    ap.add_argument("--uniform", default="best_uniform.npy", type=Path, help="Path to best_uniform.npy")
    ap.add_argument("--attraction", default="best_attraction.npy", type=Path, help="Path to best_attraction.npy")
    args = ap.parse_args()

    any_fail = False
    # Uniform
    if args.uniform:
        any_fail |= (not compare_grid(args.uniform, EXPECTED_UNIFORM, "UNIFORM"))
    # Attraction
    if args.attraction:
        any_fail |= (not compare_grid(args.attraction, EXPECTED_ATTRACTION, "ATTRACTION"))

    sys.exit(1 if any_fail else 0)

if __name__ == "__main__":
    main()
