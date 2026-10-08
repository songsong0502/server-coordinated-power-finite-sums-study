"""E1(a,b): the fixed-step stagnation that the test problems are built to show.

(a) GD on P1 (n = 50, five coordinates of x* on the kink and five at
    scales 1e-10..1e-8), alpha = 0.5 with tau in {0.01, 0.005, 0.001,
    0.0005} and tau = 0.001 with alpha in {0.2, 0.4, 0.6, 0.8}, 10000
    iterations; these are the settings of Fig. 1 of Chen, Kelley and Wang
    (2025), except for the two-sided term and the planted kink coordinates.
(b) plateau level against tau for alpha in {0.3, 0.5, 0.7}, eight step
    sizes, 40000 iterations, median of the last 5000; slope fitted in
    log-log coordinates and compared with 1/(1-alpha).
(c) the same with the kink coordinates removed (n_zero = 0).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from problems import RandomPower
from methods import gd

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def abs_err_run(P, tau, iters):
    x = P.x0.copy()
    e = np.empty(iters + 1)
    e[0] = np.linalg.norm(x - P.xstar)
    for k in range(iters):
        x = x - tau * P.grad(x)
        e[k + 1] = np.linalg.norm(x - P.xstar)
    return e


def main():
    out = {"a_tau": [], "a_alpha": [], "b": [], "c": []}
    curves = {}
    P = RandomPower(n=50, alpha=0.5, n_zero=5, m=1, seed=0)
    out["Lnom"] = P.Lnom
    for tau in (0.01, 0.005, 0.001, 0.0005):
        e = abs_err_run(P, tau, 10000)
        curves[f"tau{tau}"] = e
        out["a_tau"].append(dict(tau=tau, final=float(e[-1]),
                                 plateau=float(np.median(e[-2000:])),
                                 finite=bool(np.isfinite(e[-1]))))
    for a in (0.2, 0.4, 0.6, 0.8):
        Q = RandomPower(n=50, alpha=a, n_zero=5, m=1, seed=0)
        e = abs_err_run(Q, 0.001, 10000)
        curves[f"alpha{a}"] = e
        out["a_alpha"].append(dict(alpha=a, final=float(e[-1]),
                                   plateau=float(np.median(e[-2000:])),
                                   predicted=0.001 ** (1.0 / (1.0 - a))))
    taus = np.logspace(-3.5, -2.1, 8)
    for key, nz in (("b", 5), ("c", 0)):
        for a in (0.3, 0.5, 0.7):
            Q = RandomPower(n=50, alpha=a, n_zero=nz, m=1, seed=0)
            lev = [float(np.median(abs_err_run(Q, t, 40000)[-5000:])) for t in taus]
            row = dict(alpha=a, taus=taus.tolist(), levels=lev,
                       predicted_slope=1.0 / (1.0 - a))
            pos = np.array(lev) > 1e-13
            if pos.sum() >= 3:
                X = np.log(taus[pos])
                Y = np.log(np.array(lev)[pos])
                coef, cov = np.polyfit(X, Y, 1, cov=True)
                yhat = np.polyval(coef, X)
                r2 = 1 - np.sum((Y - yhat) ** 2) / np.sum((Y - Y.mean()) ** 2)
                row.update(slope=float(coef[0]), se=float(np.sqrt(cov[0, 0])),
                           r2=float(r2), npts=int(pos.sum()))
            out[key].append(row)
            print(key, a, row.get("slope"), row.get("se"), [f"{v:.1e}" for v in lev],
                  flush=True)
    with open(os.path.join(RES, "e1_validation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(os.path.join(RES, "e1_curves.npz"), **curves)


if __name__ == "__main__":
    main()
