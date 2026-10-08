"""Server-coordinated first-order methods with communication accounting.

A *round* is one server broadcast followed by one gather from the
participating workers.  Every method returns a dict with

    err      error after each round (np.array, err[0] at x0)
    rounds   number of rounds used
    floats   scalars moved (down + up), summed over workers
    grads    worker gradient evaluations, summed over workers
    x        final point

Synchronous methods evaluate sum_i grad_i(x) in one round, which is the
same arithmetic as the single-machine gradient; they are written with the
full gradient for speed (the process-based implementation in dist.py
checks the identity).
"""
import numpy as np


class Counter:
    def __init__(self, prob, budget):
        self.p, self.budget = prob, budget
        self.err = [prob.err(prob.x0)]
        self.rounds = 0
        self.floats = 0
        self.grads = 0

    def tick(self, x, down, up, grads):
        self.rounds += 1
        self.floats += down + up
        self.grads += grads
        self.err.append(self.p.err(x))
        return self.rounds >= self.budget

    def out(self, x, **extra):
        d = dict(err=np.array(self.err), rounds=self.rounds,
                 floats=self.floats, grads=self.grads, x=x)
        d.update(extra)
        return d


def _proj(prob):
    return getattr(prob, "project", lambda z: z)


# ---------------------------------------------------------------------------
# synchronous methods: one round = broadcast x (n per worker) + gather
# gradient (n per worker) [+ one scalar per worker if f is needed]
# ---------------------------------------------------------------------------
def gd(prob, tau, budget):
    P, n, m = _proj(prob), prob.n, prob.m
    C = Counter(prob, budget)
    x = prob.x0.copy()
    while True:
        x = P(x - tau * prob.grad(x))
        if C.tick(x, n * m, n * m, m):
            break
    return C.out(x)


def agd_sc(prob, L, mu, budget):
    """Nesterov's constant-momentum method for mu-strongly convex, L-smooth
    problems (scheme 2.2.22 in Nesterov's 2018 book), used here with the
    nominal L of the smooth part, which the non-Lipschitz term violates."""
    P, n, m = _proj(prob), prob.n, prob.m
    q = np.sqrt(mu / L)
    beta = (1.0 - q) / (1.0 + q)
    C = Counter(prob, budget)
    x = prob.x0.copy()
    y = x.copy()
    while True:
        xn = P(y - prob.grad(y) / L)
        y = xn + beta * (xn - x)
        x = xn
        if C.tick(x, n * m, n * m, m):
            break
    return C.out(x)


def adgd(prob, budget, lam0=None):
    """Adaptive gradient descent without descent (Malitsky and Mishchenko,
    ICML 2020, Algorithm 1).  One round per iteration."""
    n, m = prob.n, prob.m
    C = Counter(prob, budget)
    lam_prev = lam0 if lam0 is not None else 1e-6
    theta = np.inf
    x_prev = prob.x0.copy()
    g_prev = prob.grad(x_prev)
    x = x_prev - lam_prev * g_prev
    if C.tick(x, n * m, n * m, m):          # round that computed g_prev
        return C.out(x)
    while True:
        g = prob.grad(x)
        dg = np.linalg.norm(g - g_prev)
        dx = np.linalg.norm(x - x_prev)
        cand = dx / (2.0 * dg) if dg > 0 else np.inf
        lam = min(np.sqrt(1.0 + theta) * lam_prev, cand)
        if not np.isfinite(lam):
            lam = lam_prev
        x_prev, g_prev = x, g
        x = x - lam * g
        theta, lam_prev = lam / lam_prev, lam
        if C.tick(x, n * m, n * m, m):
            break
    return C.out(x)


def ugm(prob, eps, budget, L0=1.0):
    """Universal primal gradient method (Nesterov 2015, Math. Program. 152).
    Each trial point costs one round: broadcast the trial point, gather
    f_i and grad_i there (the gradient is reused if the trial is accepted)."""
    P, n, m = _proj(prob), prob.n, prob.m
    C = Counter(prob, budget)
    x = prob.x0.copy()
    fx, gx = prob.f(x), prob.grad(x)
    if C.tick(x, n * m, n * m + m, m):       # initial evaluation round
        return C.out(x, trials=0)
    L = L0
    trials = 0
    stop = False
    while not stop:
        M = L
        while True:
            y = P(x - gx / M)
            fy, gy = prob.f(y), prob.grad(y)
            trials += 1
            d = y - x
            ok = fy <= fx + gx @ d + 0.5 * M * (d @ d) + 0.5 * eps
            # the error curve records the current iterate after each round
            stop = C.tick(y if ok else x, n * m, n * m + m, m)
            if ok or stop:
                break
            M *= 2.0
        if ok:
            x, fx, gx = y, fy, gy
            L = M / 2.0
    return C.out(x, trials=trials)


def ufgm(prob, eps, budget, L0=1.0):
    """Universal fast gradient method (Nesterov 2015, Math. Program. 152),
    Euclidean prox-function, projection onto the feasible box if any.
    Each trial needs two rounds: (f, grad) at x_{k+1}, then f at y_{k+1}."""
    P, n, m = _proj(prob), prob.n, prob.m
    C = Counter(prob, budget)
    x0 = prob.x0.copy()
    y = x0.copy()
    A = 0.0
    L = L0
    s = np.zeros(n)                          # sum of a_j * grad f(x_j)
    trials = 0
    while True:
        v = P(x0 - s)
        M = L
        while True:
            a = (1.0 + np.sqrt(1.0 + 4.0 * M * A)) / (2.0 * M)
            An = A + a
            t = a / An
            xk = t * v + (1.0 - t) * y
            fx, gx = prob.f(xk), prob.grad(xk)
            if C.tick(y, n * m, n * m + m, m):
                return C.out(y, trials=trials)
            xh = P(v - a * gx)
            yn = t * xh + (1.0 - t) * y
            fy = prob.f(yn)
            trials += 1
            d = yn - xk
            ok = fy <= fx + gx @ d + 0.5 * M * (d @ d) + 0.5 * eps * t
            if C.tick(yn if ok else y, n * m, m, 0):
                return C.out(yn if ok else y, trials=trials)
            if ok:
                break
            M *= 2.0
        y, A, L = yn, An, M / 2.0
        s = s + a * gx


# ---------------------------------------------------------------------------
# local-step methods, data partition.  F_i = m f_i, f = (1/m) sum F_i.
# ---------------------------------------------------------------------------
def fedavg(prob, K, eta, budget, s=None, seed=0):
    n, m = prob.n, prob.m
    s = m if s is None else s
    rng = np.random.default_rng(seed)
    C = Counter(prob, budget)
    x = prob.x0.copy()
    while True:
        S = rng.choice(m, size=s, replace=False) if s < m else range(m)
        acc = np.zeros(n)
        for i in S:
            y = x.copy()
            for _ in range(K):
                y -= eta * m * prob.grad_i(i, y)
            acc += y
        x = acc / s
        if C.tick(x, n * s, n * s, K * s):
            break
    return C.out(x)


def scaffold(prob, K, eta, budget, s=None, seed=0, eta_g=1.0):
    """SCAFFOLD with option II control variates (Karimireddy et al., ICML
    2020).  Down: x and c; up: delta y and delta c (4n per participant)."""
    n, m = prob.n, prob.m
    s = m if s is None else s
    rng = np.random.default_rng(seed)
    C = Counter(prob, budget)
    x = prob.x0.copy()
    c = np.zeros(n)
    ci = [np.zeros(n) for _ in range(m)]
    while True:
        S = rng.choice(m, size=s, replace=False) if s < m else range(m)
        dy = np.zeros(n)
        dc = np.zeros(n)
        for i in S:
            y = x.copy()
            for _ in range(K):
                y -= eta * (m * prob.grad_i(i, y) - ci[i] + c)
            cnew = ci[i] - c + (x - y) / (K * eta)
            dy += y - x
            dc += cnew - ci[i]
            ci[i] = cnew
        x = x + eta_g * dy / s
        c = c + dc / m
        if C.tick(x, 2 * n * s, 2 * n * s, K * s):
            break
    return C.out(x)


# ---------------------------------------------------------------------------
# local-step methods, coordinate partition (worker i owns block B_i).
# ---------------------------------------------------------------------------
def block_local(prob, K, tau, budget):
    """Block-Jacobi with K local gradient steps on the own block; the other
    blocks stay frozen at the last synchronised point.  K = 1 is GD."""
    P = _proj(prob)
    n, m = prob.n, prob.m
    C = Counter(prob, budget)
    x = prob.x0.copy()
    while True:
        xn = x.copy()
        for B in prob.blocks:
            v = x.copy()
            for _ in range(K):
                v[B] = P(v - tau * prob.grad(v))[B]
            xn[B] = v[B]
        x = xn
        # down: full x to each worker; up: own block
        if C.tick(x, n * m, n, K * m):
            break
    return C.out(x)


def block_sampled(prob, s, tau, budget, seed=0):
    """Only s of the m block owners update per round."""
    P = _proj(prob)
    n, m = prob.n, prob.m
    rng = np.random.default_rng(seed)
    C = Counter(prob, budget)
    x = prob.x0.copy()
    while True:
        S = rng.choice(m, size=s, replace=False)
        g = prob.grad(x)
        z = P(x - tau * g)
        for i in S:
            B = prob.blocks[i]
            x[B] = z[B]
        nb = sum(len(prob.blocks[i]) for i in S)
        if C.tick(x, n * s, nb, s):
            break
    return C.out(x)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rounds_to(err, tol):
    """First round index at which err <= tol (np.inf if never)."""
    idx = np.nonzero(np.asarray(err) <= tol)[0]
    return float(idx[0]) if idx.size else np.inf


def floor_level(err, frac=0.1):
    """Median error over the last frac of the run."""
    e = np.asarray(err)
    k = max(1, int(len(e) * frac))
    return float(np.median(e[-k:]))
