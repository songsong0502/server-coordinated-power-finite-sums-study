"""E2b: how many independent kink constraints are active at the solution.

P3 (exact variant): exactly k samples are fitted at x*, k/d swept from 0.1
to 12 (d = 50), three seeds.  P1: the number of coordinates of x* exactly on
the kink (no tiny coordinates) swept from 0 to n (n = 50), three seeds.
Methods: GD-1/L, AdGD, UGM, UFGM; budget 20000 rounds.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from problems import RandomPower, synthetic_lp
from methods import gd, adgd, ugm, ufgm, rounds_to, floor_level

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUDGET = 20000


def run(job):
    fam, k, seed = job
    if fam == "P3":
        P = synthetic_lp(N=2000, d=50, m=8, p=1.4, variant="exact", n_fit=k,
                         zeta=1.0, seed=seed)
        kk = int(np.sum(np.abs(P.Ad @ P.xstar - P.bd) == 0.0))
        rank = int(np.linalg.matrix_rank(P.Ad[np.abs(P.Ad @ P.xstar - P.bd) == 0.0])) if kk else 0
    else:
        P = RandomPower(n=50, alpha=0.4, n_zero=k, n_tiny=0, m=8, seed=seed)
        kk = k
        rank = k
    eps = 0.5 * P.mu * (1e-6 * np.linalg.norm(P.x0 - P.xstar)) ** 2
    res = {}
    for name, fn in (("GD-1/L", lambda: gd(P, 1.0 / P.Lnom, BUDGET)),
                     ("AdGD", lambda: adgd(P, BUDGET)),
                     ("UGM", lambda: ugm(P, eps, BUDGET)),
                     ("UFGM", lambda: ufgm(P, eps, BUDGET))):
        r = fn()
        res[name] = dict(floor=floor_level(r["err"]), best=float(r["err"].min()),
                         r3=rounds_to(r["err"], 1e-3), r6=rounds_to(r["err"], 1e-6))
    print(fam, k, seed, kk, rank, {a: f"{b['floor']:.1e}" for a, b in res.items()},
          flush=True)
    return dict(fam=fam, k=k, seed=seed, active=kk, rank=rank, res=res)


def main():
    jobs = []
    for k in (5, 10, 25, 45, 55, 100, 200, 600):
        for s in (0, 1, 2):
            jobs.append(("P3", k, s))
    for k in (0, 5, 10, 20, 30, 40, 50):
        for s in (0, 1, 2):
            jobs.append(("P1", k, s))
    with Pool(2) as pool:
        out = pool.map(run, jobs, chunksize=1)
    with open(os.path.join(RES, "e2b.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
