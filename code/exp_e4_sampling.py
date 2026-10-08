"""E4: client sampling, s of m = 8 workers per round, five seeds.

Equal work: every configuration gets 2000 * m / s rounds, i.e. the same
number of worker gradient evaluations (times K for local-step methods).
  data partition:  FedAvg K = 1 and SCAFFOLD K = 8 on P1z1 and P3x, P3n
  coord partition: block sampling on P1c and P2 (alpha = 0.2, 0.5)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from problems import RandomPower, Semilinear
from methods import fedavg, scaffold, block_sampled, rounds_to, floor_level
from exp_e3_local_steps import build, l_loc

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BASE = 2000
M = 8


def job(arg):
    tag, meth, s, seed = arg
    R = BASE * M // s
    if meth in ("FedAvg", "SCAFFOLD"):
        P = build(tag)
        eta = 1.0 / l_loc(P)
        if meth == "FedAvg":
            r = fedavg(P, 1, eta, R, s=s, seed=seed)
        else:
            r = scaffold(P, 8, eta, R, s=s, seed=seed)
    else:
        if tag == "P1c":
            P = RandomPower(n=50, alpha=0.4, n_zero=5, m=M, seed=0, partition="coord")
            tau = 1.0 / P.Lnom
        else:
            P = Semilinear(alpha=0.2 if tag == "P2a2" else 0.5, m=M)
            tau = 0.1 / P.N ** 2
        r = block_sampled(P, s, tau, R, seed=seed)
    e = r["err"]
    epochs = np.arange(len(e)) * s / M
    out = dict(tag=tag, meth=meth, s=s, seed=seed, rounds=R,
               floor=floor_level(e), best=float(e.min()),
               r3=rounds_to(e, 1e-3), r6=rounds_to(e, 1e-6),
               ep3=float(rounds_to(e, 1e-3) * s / M), ep6=float(rounds_to(e, 1e-6) * s / M))
    print(tag, meth, s, seed, f"{out['floor']:.2e}", out["ep3"], out["ep6"], flush=True)
    # thin the curve to 2001 points on the epoch axis for the figures
    idx = np.unique(np.linspace(0, len(e) - 1, 2001).astype(int))
    return out, (epochs[idx], e[idx]) if seed == 0 else None


def main():
    jobs = []
    for tag in ("P1z1", "P3x", "P3n"):
        for meth in ("FedAvg", "SCAFFOLD"):
            for s in (8, 4, 2, 1):
                for seed in range(5):
                    if s == 8 and seed > 0:
                        continue
                    jobs.append((tag, meth, s, seed))
    for tag in ("P1c", "P2a2", "P2a5"):
        for s in (8, 4, 2, 1):
            for seed in range(5):
                jobs.append((tag, "Block", s, seed))
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    curves = {}
    for o, c in res:
        if c is not None:
            curves[f"{o['tag']}|{o['meth']}|{o['s']}|ep"] = c[0]
            curves[f"{o['tag']}|{o['meth']}|{o['s']}|err"] = c[1]
    with open(os.path.join(RES, "e4.json"), "w") as fh:
        json.dump([r[0] for r in res], fh, indent=1, default=float)
    np.savez_compressed(os.path.join(RES, "e4_curves.npz"), **curves)


if __name__ == "__main__":
    main()
