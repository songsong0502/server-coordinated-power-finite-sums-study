"""Auxiliary measurements quoted in the text.

(1) where the single-process iterates with the assembled matrix depart from
    the block-ordered iterates of the process implementation (same arithmetic
    as exp_e1_dist.py, whose distributed iterates equal the block-ordered ones
    bitwise);
(2) UGM and UFGM: rounds per accepted step and final curvature estimate on the
    four figure instances of E2 and on P1-n200-a0.4-T-s0;
(3) spectral radius of the linear part of one block-local round on P1
    (coordinate partition, n = 50, m = 8, tau = 1/Lnom) and of block Jacobi.
Output: results/aux.json
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_e1_dist import make_block, single_blocks, single_assembled
from problems import RandomPower
from testset import instances, build

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def deviation():
    n, m, seed, alpha, nz, iters = 200, 4, 3, 0.4, 20, 2000
    R = np.vstack([make_block(n, m, i, seed) for i in range(m)])
    tau = 1.0 / np.linalg.eigvalsh(R.T @ R + np.eye(n))[-1]
    trb = np.array(single_blocks(n, m, seed, alpha, nz, tau, iters))
    tra, _, xs = single_assembled(n, m, seed, alpha, nz, tau, iters)
    tra = np.array(tra)
    D = np.abs(trb - tra)
    kink = np.arange(nz)
    k1 = int(np.argmax(D.max(1) > 1e-10))
    err = np.linalg.norm(trb - xs, axis=1) / np.linalg.norm(trb[0] - xs)
    signs = np.sign(trb[:, kink])
    flips = (signs[1:] * signs[:-1] < 0).sum(1)       # sign changes per step
    first_flip = int(np.argmax(flips > 0))
    return dict(max_dev_upto500=float(D[:501].max()), first_it_1e10=k1,
                err_at_first=float(err[k1]), err_final=float(err[-1]),
                err_min_index=int(np.argmin(err)),
                dev_kink=float(D[:, kink].max()),
                dev_other=float(D[:, nz:].max()),
                kink_max_abs_at_first=float(np.abs(trb[k1, kink]).max()),
                first_kink_sign_flip=first_flip,
                kink_flips_per_step_after=float(flips[k1:].mean()),
                rel_err_curve_diff=float(np.max(np.abs(err - np.linalg.norm(tra - xs, axis=1)
                                                       / np.linalg.norm(tra[0] - xs)) / err)))


def universal_counts(P, eps, budget=20000):
    """UGM and UFGM re-run with a count of accepted steps (same rules as methods.py)."""
    out = {}
    x = P.x0.copy(); fx, gx = P.f(x), P.grad(x); L = 1.0; rounds = 1; acc = 0
    while rounds < budget:
        M = L
        while True:
            y = x - gx / M; fy = P.f(y); rounds += 1; d = y - x
            if fy <= fx + gx @ d + 0.5 * M * (d @ d) + 0.5 * eps or rounds >= budget:
                ok = fy <= fx + gx @ d + 0.5 * M * (d @ d) + 0.5 * eps
                break
            M *= 2.0
        if ok:
            x, fx, gx = y, fy, P.grad(y); L = M / 2.0; acc += 1
    out["UGM"] = dict(rounds_per_step=rounds / acc, final_L=L, Lnom=P.Lnom)
    x0 = P.x0.copy(); y = x0.copy(); A = 0.0; L = 1.0; s = np.zeros(P.n)
    rounds = 0; acc = 0
    while rounds + 2 <= budget:
        v = x0 - s; M = L
        while rounds + 2 <= budget:
            a = (1 + np.sqrt(1 + 4 * M * A)) / (2 * M); An = A + a; t = a / An
            xk = t * v + (1 - t) * y; fx, gx = P.f(xk), P.grad(xk)
            yn = t * (v - a * gx) + (1 - t) * y; fy = P.f(yn); rounds += 2
            d = yn - xk
            ok = fy <= fx + gx @ d + 0.5 * M * (d @ d) + 0.5 * eps * t
            if ok:
                break
            M *= 2.0
        if not ok:
            break
        y, A, L = yn, An, M / 2.0; s = s + a * gx; acc += 1
    out["UFGM"] = dict(rounds_per_step=rounds / max(acc, 1), final_L=L)
    return out


def spectral():
    P = RandomPower(n=50, alpha=0.4, n_zero=5, m=8, seed=0, partition="coord")
    A, n, tau = P.A, P.n, 1.0 / P.Lnom
    D = np.zeros_like(A)
    for B in P.blocks:
        D[np.ix_(B, B)] = A[np.ix_(B, B)]
    out = {"jacobi": float(max(abs(np.linalg.eigvals(np.linalg.solve(D, D - A)))))}
    for K in (1, 2, 4, 16, 64):
        T = np.zeros((n, n))
        for B in P.blocks:
            nb = np.setdiff1d(np.arange(n), B)
            G = np.eye(len(B)) - tau * A[np.ix_(B, B)]
            S = sum(np.linalg.matrix_power(G, j) for j in range(K))
            T[np.ix_(B, B)] = np.linalg.matrix_power(G, K)
            T[np.ix_(B, nb)] = -tau * S @ A[np.ix_(B, nb)]
        out[f"K{K}"] = float(max(abs(np.linalg.eigvals(T))))
    return out


def main():
    res = {"deviation": deviation(), "spectral": spectral(), "universal": {}}
    spec = dict(instances())
    for key in ("P1-n50-a0.4-G-s0", "P1-n50-a0.4-T-s0", "P1-n200-a0.4-T-s0",
                "P3s-d50-p1.4-exact", "P3s-d50-p1.4-noisy"):
        P = build(spec[key])
        eps = 0.5 * P.mu * (1e-6 * np.linalg.norm(P.x0 - P.xstar)) ** 2
        res["universal"][key] = universal_counts(P, eps)
        print(key, res["universal"][key], flush=True)
    print(res["deviation"]); print(res["spectral"])
    with open(os.path.join(RES, "aux.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
