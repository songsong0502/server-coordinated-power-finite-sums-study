"""E2: synchronous step-size rules on the 52-instance test set.

Methods: GD with tau = 1/Lnom, GD with the best tau of a 13-point grid,
Nesterov's constant-momentum method (AGD), adaptive GD (AdGD), and the
universal primal / fast gradient methods (UGM, UFGM).  Budget 20000 rounds.
Output: results/e2.json (per instance, per method) and curves for the
figure instances in results/e2_curves.npz.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from methods import gd, agd_sc, adgd, ugm, ufgm, rounds_to, floor_level
from testset import instances, build

BUDGET = 20000
TOLS = (1e-3, 1e-6)
GRID = [2.0 ** k for k in range(-8, 5)]          # multiples of 1/Lnom
RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_KEYS = {"P1-n50-a0.4-T-s0", "P1-n50-a0.4-G-s0", "P3s-d50-p1.4-exact",
            "P3s-d50-p1.4-noisy"}


def summarise(r, t):
    e = r["err"]
    return dict(r3=rounds_to(e, TOLS[0]), r6=rounds_to(e, TOLS[1]),
                floor=floor_level(e), final=float(e[-1]), best=float(e.min()),
                rounds=r["rounds"], floats=r["floats"], grads=r["grads"],
                trials=r.get("trials"), secs=t)


def run(item):
    key, spec = item
    P = build(spec)
    out, curves = {}, {}
    eps = 0.5 * P.mu * (TOLS[1] * np.linalg.norm(P.x0 - P.xstar)) ** 2

    def go(name, fn):
        t = time.time()
        r = fn()
        out[name] = summarise(r, time.time() - t)
        if key in FIG_KEYS:
            curves[name] = r["err"]
        return r

    go("GD-1/L", lambda: gd(P, 1.0 / P.Lnom, BUDGET))
    # tuned GD: every grid point is a full run; its cost is reported too
    grid = []
    for g in GRID:
        t = time.time()
        r = gd(P, g / P.Lnom, BUDGET)
        e = r["err"]
        ok = np.all(np.isfinite(e)) and e[-1] < 1e3
        grid.append((g, summarise(r, time.time() - t) if ok else None, e))
    def score(s):
        if s is None:
            return (2, np.inf)
        if np.isfinite(s["r6"]):
            return (0, s["r6"])
        return (1, s["best"])
    gbest = min(grid, key=lambda z: score(z[1]))
    s = dict(gbest[1])
    s["tau_mult"] = gbest[0]
    s["tuning_rounds"] = BUDGET * len(GRID)
    out["GD-tuned"] = s
    if key in FIG_KEYS:
        curves["GD-tuned"] = gbest[2]
    go("AGD", lambda: agd_sc(P, P.Lnom, P.mu, BUDGET))
    go("AdGD", lambda: adgd(P, BUDGET))
    go("UGM", lambda: ugm(P, eps, BUDGET))
    go("UFGM", lambda: ufgm(P, eps, BUDGET))
    info = dict(n=P.n, m=P.m, Lnom=P.Lnom, mu=P.mu, alpha=float(P.alpha),
                eps=eps)
    if hasattr(P, "min_abs_residual"):
        info["min_abs_residual"] = P.min_abs_residual
        info["newton_gnorm"] = P.newton_gnorm
    print(key, {k: (v["r6"], f"{v['best']:.1e}") for k, v in out.items()},
          flush=True)
    return key, spec, info, out, curves


def main():
    items = instances()
    with Pool(2) as pool:
        res = pool.map(run, items, chunksize=1)
    js, cv = {}, {}
    for key, spec, info, out, curves in res:
        js[key] = dict(spec=spec, info=info, methods=out)
        for name, e in curves.items():
            cv[f"{key}|{name}"] = e
    with open(os.path.join(RES, "e2.json"), "w") as fh:
        json.dump(js, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(RES, "e2_curves.npz"), **cv)


if __name__ == "__main__":
    main()
