"""
ga_optimize_jax.py
==================
Genetic Algorithm for antenna placement accelerated with JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time, random
from typing import Tuple

# ─────────── constants ───────────
ROWS, COLS = 6, 6
GRID = ROWS * COLS
TOTAL_USERS = 5_000
ANT_CAP = 300
MAX_ANT = TOTAL_USERS // ANT_CAP
P_MOVE = 0.4
EP_STEPS = 24 * 10
W_FAIL, W_RED, W_ANT = 3.0, 0.1, 0.5
POP_SIZE = 64
GENERATIONS = 120
TOUR_K = 3
CX_PROB = 0.7
MUT_PROB = 0.05

KEY = jax.random.PRNGKey(0)
print("Running on:", jax.devices()[0].platform)

# ───────── neighbour LUT (36×8) ─────────
nbr = []
for r in range(ROWS):
    for c in range(COLS):
        lst = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    lst.append(nr * COLS + nc)
        while len(lst) < 8:
            lst.append(-1)
        nbr.append(lst)
NBR_IDX = jnp.array(nbr, jnp.int32)
VALID_MSK = (NBR_IDX >= 0).astype(jnp.int32)

# ───────── sim helpers ─────────
def init_state(key):
    cars = jax.random.multinomial(key, TOTAL_USERS, jnp.ones(GRID) / GRID).astype(jnp.int32)
    return cars, key

def step_state(cars, key):
    moving = jax.random.binomial(key, cars, P_MOVE)
    staying = cars - moving

    def body(i, acc):
        nbrs, mask = NBR_IDX[i], VALID_MSK[i]
        k = mask.sum()
        p = mask / jnp.maximum(k, 1)
        ki = jax.random.fold_in(key, i)
        dest = jax.random.multinomial(ki, moving[i], p)

        def inner(j, a):
            return jax.lax.cond(nbrs[j] >= 0,
                                 lambda v: v.at[nbrs[j]].add(dest[j]),
                                 lambda v: v, a)

        acc = jax.lax.fori_loop(0, 8, inner, acc)
        return acc

    add = jnp.zeros_like(cars)
    add = jax.lax.fori_loop(0, GRID, body, add)
    return staying + add, jax.random.split(key)[0]

def cov_stats(cars: jnp.ndarray, antenna: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    capacity = antenna * ANT_CAP

    def serve_cell(i, carry):
        users = cars[i]
        rem = users
        served = 0
        redirected = 0

        def try_serve(j, val):
            s, r, rem = val
            idx = NBR_IDX[i, j]
            cond = idx >= 0

            def update_if_valid(_):
                cap = capacity[idx]
                take = jnp.minimum(cap, rem)
                r_new = jnp.where(idx != i, r + take, r)
                return s + take, r_new, rem - take

            return jax.lax.cond(cond, update_if_valid, lambda _: (s, r, rem), operand=None)

        served, redirected, rem = jax.lax.fori_loop(0, 8, try_serve, (served, redirected, rem))
        return carry[0] + rem, carry[1] + redirected

    failures, redirects = jax.lax.fori_loop(0, GRID, serve_cell, (0, 0))
    return failures, redirects

def episode(bits, key):
    cars, key = init_state(key)
    fails = reds = 0
    for _ in range(EP_STEPS):
        cars, key = step_state(cars, key)
        df, dr = cov_stats(cars, bits)
        fails += df; reds += dr
    return jnp.array([fails, reds, bits.sum()], jnp.float32)

vm_ep = jax.vmap(episode, in_axes=(0, 0))

def fitness(pbits, keys):
    vals = vm_ep(pbits, keys)
    fail, red, ant = vals[:, 0], vals[:, 1], vals[:, 2]
    return W_FAIL * fail + W_RED * red + W_ANT * ant, vals

# ───────── GA utils ─────────
def rand_layout():
    bits = [0] * GRID
    for i in random.sample(range(GRID), MAX_ANT):
        bits[i] = 1
    return bits

def repair(b):
    ones = [i for i, x in enumerate(b) if x]
    zeros = [i for i, x in enumerate(b) if not x]
    if len(ones) > MAX_ANT:
        for i in random.sample(ones, len(ones) - MAX_ANT):
            b[i] = 0
    if len(ones) < MAX_ANT:
        for i in random.sample(zeros, MAX_ANT - len(ones)):
            b[i] = 1

def cx(p1, p2):
    if random.random() > CX_PROB:
        return p1[:], p2[:]
    pt = random.randint(1, GRID - 2)
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    repair(c1)
    repair(c2)
    return c1, c2

def mut(b):
    for i in range(GRID):
        if random.random() < MUT_PROB:
            b[i] ^= 1
    repair(b)

# ───────── GA loop ─────────
pop = [rand_layout() for _ in range(POP_SIZE)]
best, bscore, btrip = None, 1e30, None
t0 = time.time()

for g in range(GENERATIONS):
    pb = jnp.stack([jnp.array(x, jnp.int8) for x in pop])
    k = jax.random.split(KEY, g + POP_SIZE)[-POP_SIZE:]
    scr, vals = fitness(pb, k)
    scr_np = np.array(scr)
    ib = int(scr_np.argmin())
    if scr_np[ib] < bscore:
        bscore = float(scr_np[ib])
        best = pop[ib][:]
        btrip = tuple(map(int, vals[ib]))
    if g % 10 == 0 or g == GENERATIONS - 1:
        f, r, a = btrip
        eta = (time.time() - t0) / (g + 1) * (GENERATIONS - g - 1) / 60
        print(f"Gen {g:03d} | best={bscore:6.2f} (F={f} R={r} A={a}) | ETA {eta:4.1f}m")

    new = []
    while len(new) < POP_SIZE:
        p1 = min(random.sample(list(zip(pop, scr_np)), TOUR_K), key=lambda t: t[1])[0]
        p2 = min(random.sample(list(zip(pop, scr_np)), TOUR_K), key=lambda t: t[1])[0]
        c1, c2 = cx(p1, p2)
        mut(c1)
        mut(c2)
        new += [c1, c2]
    pop = new[:POP_SIZE]

print("\n=== DONE ===")
print("best fitness:", bscore)
print("fails, reds, ant:", btrip)
print(np.array(best).reshape(ROWS, COLS))
