"""E1(c): a process-per-worker implementation of the star network.

Worker i generates and stores only its own row block R_i of the data
matrix of problem P1 (data partition) and serves
    grad_i(x) = R_i'(R_i x) + x/m + |x|^a sign(x)/m - c_i.
The server holds vectors only.  We check
  (1) bitwise identity of 2000 GD iterates against a single process that
      evaluates the same blocks in the same order;
  (2) the deviation from a single process using the assembled matrix
      A = R'R + I (different floating-point association);
  (3) per-worker peak resident memory for m in {1, 2, 4, 8} at n = 4000;
  (4) wall-clock time per round (the host has 2 cores; reported as is).
Memory is read from /proc/self/status (VmHWM).
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import multiprocessing as mp
import time

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def mem_mb():
    out = {}
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                out[k] = float(v.split()[0]) / 1024.0
    return out


def row_split(n, m):
    return np.array_split(np.arange(n), m)


def make_block(n, m, i, seed):
    """Rows of R owned by worker i, generated locally from (seed, i)."""
    rows = row_split(n, m)[i]
    rng = np.random.default_rng([seed, i, m])
    return rng.standard_normal((rows.size, n))


def make_vectors(n, seed, alpha, n_zero):
    rng = np.random.default_rng([seed, 10 ** 6])
    xs = rng.standard_normal(n)
    xs[:n_zero] = 0.0
    x0 = rng.standard_normal(n)
    return xs, x0


def dpsi(x, a):
    return np.abs(x) ** a * np.sign(x)


def worker(conn, n, m, i, seed, alpha, n_zero):
    Ri = make_block(n, m, i, seed)
    xs, _ = make_vectors(n, seed, alpha, n_zero)
    ci = Ri.T @ (Ri @ xs) + xs / m + dpsi(xs, alpha) / m
    conn.send(("ready", mem_mb()))
    while True:
        msg = conn.recv()
        if msg is None:
            conn.send(("bye", mem_mb()))
            break
        x = msg
        conn.send(Ri.T @ (Ri @ x) + x / m + dpsi(x, alpha) / m - ci)


def server(n, m, seed, alpha, n_zero, tau, iters, record=True):
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for i in range(m):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, n, m, i, seed, alpha, n_zero))
        p.start()
        pipes.append(a)
        procs.append(p)
    mem_ready = [c.recv()[1] for c in pipes]
    xs, x = make_vectors(n, seed, alpha, n_zero)
    traj = [x.copy()] if record else None
    t0 = time.perf_counter()
    for _ in range(iters):
        for c in pipes:
            c.send(x)
        g = np.zeros(n)
        for c in pipes:                      # fixed reduction order
            g = g + c.recv()
        x = x - tau * g
        if record:
            traj.append(x.copy())
    dt = (time.perf_counter() - t0) / iters
    for c in pipes:
        c.send(None)
    mem_end = [c.recv()[1] for c in pipes]
    for p in procs:
        p.join()
    return x, traj, dt, mem_ready, mem_end, mem_mb()


def single_blocks(n, m, seed, alpha, n_zero, tau, iters):
    """One process, same blocks, same order of summation."""
    Rs = [make_block(n, m, i, seed) for i in range(m)]
    xs, x = make_vectors(n, seed, alpha, n_zero)
    cs = [R.T @ (R @ xs) + xs / m + dpsi(xs, alpha) / m for R in Rs]
    traj = [x.copy()]
    for _ in range(iters):
        g = np.zeros(n)
        for R, c in zip(Rs, cs):
            g = g + (R.T @ (R @ x) + x / m + dpsi(x, alpha) / m - c)
        x = x - tau * g
        traj.append(x.copy())
    return traj


def single_assembled(n, m, seed, alpha, n_zero, tau, iters):
    R = np.vstack([make_block(n, m, i, seed) for i in range(m)])
    A = R.T @ R + np.eye(n)
    xs, x = make_vectors(n, seed, alpha, n_zero)
    c = A @ xs + dpsi(xs, alpha)
    traj = [x.copy()]
    t0 = time.perf_counter()
    for _ in range(iters):
        x = x - tau * (A @ x + dpsi(x, alpha) - c)
        traj.append(x.copy())
    dt = (time.perf_counter() - t0) / iters
    return traj, dt, xs


def main():
    out = {}
    # (1)-(2) exactness at n = 200, m = 4, alpha = 0.4, 20 kink coordinates
    n, m, seed, alpha, nz, iters = 200, 4, 3, 0.4, 20, 2000
    R = np.vstack([make_block(n, m, i, seed) for i in range(m)])
    L = np.linalg.eigvalsh(R.T @ R + np.eye(n))[-1]
    tau = 1.0 / L
    xd, trd, _, _, _, _ = server(n, m, seed, alpha, nz, tau, iters)
    trb = single_blocks(n, m, seed, alpha, nz, tau, iters)
    tra, _, xs = single_assembled(n, m, seed, alpha, nz, tau, iters)
    dev_b = max(float(np.max(np.abs(a - b))) for a, b in zip(trd, trb))
    dev_a = max(float(np.max(np.abs(a - b))) for a, b in zip(trd, tra))
    e0 = np.linalg.norm(trd[0] - xs)
    err_d = [float(np.linalg.norm(z - xs) / e0) for z in trd]
    err_a = [float(np.linalg.norm(z - xs) / e0) for z in tra]
    out["exactness"] = dict(n=n, m=m, alpha=alpha, n_zero=nz, iters=iters,
                            tau=tau, max_dev_same_order=dev_b,
                            max_dev_assembled=dev_a,
                            final_err_dist=err_d[-1], final_err_assembled=err_a[-1])
    np.savez_compressed(os.path.join(RES, "e1_dist_curves.npz"),
                        err_d=np.array(err_d), err_a=np.array(err_a))
    print(out["exactness"], flush=True)

    # (3)-(4) memory and time at n = 4000
    n, iters = 4000, 60
    rows = []
    for m in (1, 2, 4, 8):
        _, _, dt, mr, me, ms = server(n, m, 0, 0.4, 0, 1e-5, iters, record=False)
        rows.append(dict(m=m, block_mb=8.0 * n * n / m / 2 ** 20,
                         worker_hwm=[d["VmHWM"] for d in me],
                         worker_rss=[d["VmRSS"] for d in me],
                         server_rss=ms["VmRSS"], sec_per_round=dt))
        print(rows[-1], flush=True)
    out["memory"] = rows
    out["note"] = "host: %d cores" % os.cpu_count()
    with open(os.path.join(RES, "e1_dist.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
