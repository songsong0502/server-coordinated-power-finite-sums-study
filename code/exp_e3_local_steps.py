"""E3: local steps between synchronisations.

(a) data partition, m = 8: FedAvg and SCAFFOLD with K in {1,2,4,8,16,32}
    local steps of size eta = 1/L_loc (L_loc = largest nominal curvature of
    the local objectives F_i = m f_i), budget 5000 rounds.
      P1-data: n = 50, alpha = 0.4, 5 + 5 kink coordinates, zeta in {0, 0.3, 1, 3}
      P3-exact (k = 600 > d fitted samples) and P3-noisy, d = 50, p = 1.4
(b) coordinate partition, m = 8: block-Jacobi with K in {1,4,16,64}
      P1-coord (n = 50, alpha = 0.4, kink-touching), tau = 1/Lnom
      P2 with alpha in {0.2, 0.5}, tau = 0.1 h^2
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from problems import RandomPower, synthetic_lp, Semilinear
from methods import fedavg, scaffold, block_local, rounds_to, floor_level

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUDGET = 5000
KS = (1, 2, 4, 8, 16, 32)


def l_loc(P):
    if P.name == "P1":
        return max(P.m * np.linalg.eigvalsh(Ai)[-1] for Ai in P.Ai)
    return max(P.m * np.linalg.eigvalsh(P.Ad[g].T @ P.Ad[g] / P.Nn)[-1] + P.mu
               for g in P.groups)


def build(tag):
    if tag.startswith("P1z"):
        z = float(tag[3:])
        return RandomPower(n=50, alpha=0.4, n_zero=5, m=8, seed=0, zeta=z)
    if tag == "P3x":
        return synthetic_lp(N=2000, d=50, m=8, p=1.4, variant="exact",
                            n_fit=600, zeta=1.0, seed=0)
    if tag == "P3n":
        return synthetic_lp(N=2000, d=50, m=8, p=1.4, variant="noisy",
                            zeta=1.0, seed=0)
    raise ValueError(tag)


def summ(r):
    e = r["err"]
    return dict(r3=rounds_to(e, 1e-3), r6=rounds_to(e, 1e-6), floor=floor_level(e),
                best=float(e.min()), floats=r["floats"], grads=r["grads"])


def job_data(arg):
    tag, meth, K = arg
    P = build(tag)
    eta = 1.0 / l_loc(P)
    fn = fedavg if meth == "FedAvg" else scaffold
    r = fn(P, K, eta, BUDGET)
    out = summ(r)
    out.update(tag=tag, meth=meth, K=K, eta=eta)
    keep = (K in (1, 8, 32))
    print(tag, meth, K, f"{out['floor']:.2e}", out["r3"], out["r6"], flush=True)
    return out, (r["err"] if keep else None)


def job_coord(arg):
    tag, K = arg
    if tag == "P1c":
        P = RandomPower(n=50, alpha=0.4, n_zero=5, m=8, seed=0, partition="coord")
        tau = 1.0 / P.Lnom
    else:
        a = 0.2 if tag == "P2a2" else 0.5
        P = Semilinear(alpha=a, m=8)
        tau = 0.1 / P.N ** 2
    r = block_local(P, K, tau, BUDGET)
    out = summ(r)
    out.update(tag=tag, K=K, tau=tau)
    print(tag, "BJ", K, f"{out['floor']:.2e}", out["r3"], out["r6"], flush=True)
    return out, r["err"]


def main():
    jobs = [(t, mth, K) for t in ("P1z0", "P1z0.3", "P1z1", "P1z3", "P3x", "P3n")
            for mth in ("FedAvg", "SCAFFOLD") for K in KS]
    cjobs = [(t, K) for t in ("P1c", "P2a2", "P2a5") for K in (1, 4, 16, 64)]
    with Pool(2) as pool:
        A = pool.map(job_data, jobs, chunksize=1)
        B = pool.map(job_coord, cjobs, chunksize=1)
    curves = {}
    for (o, e) in A:
        if e is not None:
            curves[f"{o['tag']}|{o['meth']}|{o['K']}"] = e
    for (o, e) in B:
        curves[f"{o['tag']}|BJ|{o['K']}"] = e
    with open(os.path.join(RES, "e3.json"), "w") as fh:
        json.dump(dict(data=[a[0] for a in A], coord=[b[0] for b in B]), fh,
                  indent=1, default=float)
    np.savez_compressed(os.path.join(RES, "e3_curves.npz"), **curves)


if __name__ == "__main__":
    main()
