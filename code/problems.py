"""Test problems for the star-network study.

Every problem is a strongly convex finite sum  f(x) = sum_i f_i(x)  over m
workers.  Each problem object exposes

    n, m            dimension and number of workers
    x0              starting point
    xstar           minimiser (exact, or a high-accuracy reference)
    mu              a lower bound on the strong convexity modulus
    Lnom            a "nominal" Lipschitz constant (smooth part only)
    f(x), grad(x)   full objective and gradient
    f_i(i,x), grad_i(i,x)   worker i's share
    err(x)          the reported error measure

Partition modes:
    'data'   worker i owns a share of the data and evaluates f_i on the full x
    'coord'  worker i owns a block of coordinates (rows of the operator)
"""
import numpy as np


def _psi(t, a):
    """|t|^{1+a}/(1+a)"""
    return np.abs(t) ** (1.0 + a) / (1.0 + a)


def _dpsi(t, a):
    """d/dt |t|^{1+a}/(1+a) = |t|^a sign(t); a-Holder, not Lipschitz at 0."""
    return np.abs(t) ** a * np.sign(t)


# ---------------------------------------------------------------------------
# P1: random power model, problem (14) of Chen-Kelley-Wang (2025) with the
# two-sided term |u|^{1+alpha}; n_zero coordinates of x* sit on the kink.
# ---------------------------------------------------------------------------
class RandomPower:
    name = "P1"

    def __init__(self, n=50, alpha=0.5, n_zero=5, m=4, seed=0,
                 zeta=0.0, partition="data", n_tiny=None):
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((n, n))
        self.n, self.m, self.alpha = n, m, alpha
        self.partition = partition
        xs = rng.standard_normal(n)
        if n_zero > 0:
            # coordinates exactly on the kink, plus a few at tiny scales that
            # break the period-2 symmetry of the limit cycle (as in the
            # earlier report); both are part of the instance definition
            rng2 = np.random.default_rng(seed + 77)
            nt = n_zero if n_tiny is None else n_tiny
            xs[:n_zero] = 0.0
            if nt > 0:
                xs[n_zero:n_zero + nt] = (np.logspace(-10, -8, nt)
                                          * rng2.choice([-1.0, 1.0], nt))
        self.xstar = xs
        self.x0 = rng.standard_normal(n)
        self.A = R.T @ R + np.eye(n)
        self.mu = 1.0
        self.Lnom = float(np.linalg.eigvalsh(self.A)[-1])
        self.n_zero = n_zero
        self.c = self.A @ xs + _dpsi(xs, alpha)
        if partition == "data":
            rows = np.array_split(np.arange(n), m)
            self.Ai = [R[r].T @ R[r] + np.eye(n) / m for r in rows]
            # heterogeneity: zero-sum perturbations of the linear terms
            rng3 = np.random.default_rng(seed + 991)
            D = rng3.standard_normal((m, n))
            D -= D.mean(axis=0, keepdims=True)
            scale = np.linalg.norm(self.c) / np.sqrt(m)
            # one global factor keeps sum_i D_i = 0; mean row norm = zeta*scale
            rn = np.mean(np.linalg.norm(D, axis=1))
            D *= zeta * scale / max(rn, 1e-300)
            if m == 1:
                D[:] = 0.0
            self.ci = [self.Ai[i] @ xs + _dpsi(xs, alpha) / m + D[i]
                       for i in range(m)]
        else:
            self.blocks = np.array_split(np.arange(n), m)
        self.zeta = zeta

    # full objective -------------------------------------------------------
    def f(self, x):
        return 0.5 * x @ (self.A @ x) + np.sum(_psi(x, self.alpha)) - self.c @ x

    def grad(self, x):
        return self.A @ x + _dpsi(x, self.alpha) - self.c

    # worker shares (data partition) ------------------------------------------
    def f_i(self, i, x):
        return (0.5 * x @ (self.Ai[i] @ x) + np.sum(_psi(x, self.alpha)) / self.m
                - self.ci[i] @ x)

    def grad_i(self, i, x):
        return self.Ai[i] @ x + _dpsi(x, self.alpha) / self.m - self.ci[i]

    def err(self, x):
        return np.linalg.norm(x - self.xstar) / np.linalg.norm(self.x0 - self.xstar)

    def fstar(self):
        return self.f(self.xstar)


# ---------------------------------------------------------------------------
# P2: constrained semilinear problem (5.8) of Chen-Kelley-Wang (2026),
#   min_{u in [-1,1]^n} 1/2 u'Au + delta/(1+a) e'|u|^{1+a} - 1/(1+p) e'|u|^{1+p} - b'u
# five-point Laplacian, h = 2^-4, boundary data 0.5 - sin(x) sin(y).
# No closed-form solution; the error is the fixed-point residual (5.7) with a
# fixed reference step, relative to its value at x0.
# ---------------------------------------------------------------------------
def five_point_matrix(N):
    n1 = N - 1
    n = n1 * n1
    h2 = float(N * N)
    A = np.zeros((n, n))
    for j in range(n1):
        for i in range(n1):
            k = j * n1 + i
            A[k, k] = 4.0 * h2
            if i > 0:
                A[k, k - 1] = -h2
            if i < n1 - 1:
                A[k, k + 1] = -h2
            if j > 0:
                A[k, k - n1] = -h2
            if j < n1 - 1:
                A[k, k + n1] = -h2
    return A


def boundary_vector(N, g):
    n1 = N - 1
    h = 1.0 / N
    h2 = float(N * N)
    b = np.zeros(n1 * n1)
    for j in range(n1):
        y = (j + 1) * h
        for i in range(n1):
            x = (i + 1) * h
            k = j * n1 + i
            if i == 0:
                b[k] += h2 * g(0.0, y)
            if i == n1 - 1:
                b[k] += h2 * g(1.0, y)
            if j == 0:
                b[k] += h2 * g(x, 0.0)
            if j == n1 - 1:
                b[k] += h2 * g(x, 1.0)
    return b


class Semilinear:
    name = "P2"

    def __init__(self, h_exp=4, alpha=0.5, p=1.5, delta=20.0, m=4):
        N = 2 ** h_exp
        self.N, self.alpha, self.p, self.delta, self.m = N, alpha, p, delta, m
        self.n = (N - 1) ** 2
        self.A = five_point_matrix(N)
        self.b = boundary_vector(N, lambda x, y: 0.5 - np.sin(x) * np.sin(y))
        self.x0 = np.clip(np.linalg.solve(self.A, self.b), -1.0, 1.0)
        ev = np.linalg.eigvalsh(self.A)
        self.mu = float(ev[0])
        self.Lnom = float(ev[-1]) + p
        self.blocks = np.array_split(np.arange(self.n), m)
        self.partition = "coord"
        self.tau_ref = 0.1 / N ** 2
        self.r0 = self.residual(self.x0)
        self.xstar = None

    def f(self, u):
        au = np.abs(u)
        return (0.5 * u @ (self.A @ u)
                + self.delta * np.sum(au ** (1.0 + self.alpha)) / (1.0 + self.alpha)
                - np.sum(au ** (1.0 + self.p)) / (1.0 + self.p) - self.b @ u)

    def grad(self, u):
        return (self.A @ u + self.delta * _dpsi(u, self.alpha)
                - np.abs(u) ** (self.p - 1.0) * u - self.b)

    def project(self, u):
        return np.clip(u, -1.0, 1.0)

    def residual(self, u):
        return np.linalg.norm(u - self.project(u - self.tau_ref * self.grad(u)))

    def err(self, u):
        return self.residual(u) / self.r0


# ---------------------------------------------------------------------------
# P3: l_p regression with ridge, f(x) = (1/N) sum_j |a_j'x - b_j|^p / p
#                                        + mu/2 ||x||^2 - c'x,  1 < p < 2.
# The gradient of |r|^p/p is (p-1)-Holder at r = 0.
#   variant 'exact'  manufactured: a fraction rho of the samples is fitted
#                    exactly at x* (residual 0, on the kink); c makes x*
#                    the minimiser.
#   variant 'noisy'  c = 0, every sample noisy; x* by damped Newton.
# Clients hold disjoint sample sets (data partition).
# ---------------------------------------------------------------------------
class LpRegression:
    name = "P3"

    def __init__(self, Adata, bdata, groups, p=1.5, mu=1e-2, variant="noisy",
                 xstar_planted=None, x0=None):
        self.Ad, self.bd = Adata, bdata
        self.Nn, self.n = Adata.shape
        self.p, self.mu, self.variant = p, mu, variant
        self.groups = groups
        self.m = len(groups)
        self.partition = "data"
        self.alpha = p - 1.0
        self.c = np.zeros(self.n)
        if variant == "exact":
            xs = xstar_planted
            r = Adata @ xs - bdata
            self.c = mu * xs + Adata.T @ _dpsi(r, p - 1.0) / self.Nn
            self.xstar = xs
        else:
            self.xstar = None
        self.Lnom = float(np.linalg.eigvalsh(Adata.T @ Adata / self.Nn)[-1]) + mu
        self.x0 = np.zeros(self.n) if x0 is None else x0
        if variant == "noisy":
            self.xstar = self._newton()

    def f(self, x):
        r = self.Ad @ x - self.bd
        return (np.sum(np.abs(r) ** self.p) / (self.p * self.Nn)
                + 0.5 * self.mu * x @ x - self.c @ x)

    def grad(self, x):
        r = self.Ad @ x - self.bd
        return self.Ad.T @ _dpsi(r, self.p - 1.0) / self.Nn + self.mu * x - self.c

    def f_i(self, i, x):
        g = self.groups[i]
        r = self.Ad[g] @ x - self.bd[g]
        return (np.sum(np.abs(r) ** self.p) / (self.p * self.Nn)
                + 0.5 * self.mu * x @ x / self.m - self.c @ x / self.m)

    def grad_i(self, i, x):
        g = self.groups[i]
        r = self.Ad[g] @ x - self.bd[g]
        return (self.Ad[g].T @ _dpsi(r, self.p - 1.0) / self.Nn
                + self.mu * x / self.m - self.c / self.m)

    def _newton(self, tol=1e-13, maxit=200):
        """Damped Newton; f is C^2 wherever no residual vanishes."""
        x = np.linalg.lstsq(self.Ad, self.bd, rcond=None)[0]
        for _ in range(maxit):
            g = self.grad(x)
            if np.linalg.norm(g) < tol:
                break
            r = self.Ad @ x - self.bd
            w = (self.p - 1.0) * np.maximum(np.abs(r), 1e-300) ** (self.p - 2.0)
            H = (self.Ad.T * w) @ self.Ad / self.Nn + self.mu * np.eye(self.n)
            d = np.linalg.solve(H, -g)
            t, f0 = 1.0, self.f(x)
            while self.f(x + t * d) > f0 + 1e-4 * t * g @ d and t > 1e-12:
                t *= 0.5
            x = x + t * d
        self.newton_gnorm = float(np.linalg.norm(self.grad(x)))
        self.min_abs_residual = float(np.min(np.abs(self.Ad @ x - self.bd)))
        return x

    def err(self, x):
        return np.linalg.norm(x - self.xstar) / np.linalg.norm(self.x0 - self.xstar)

    def fstar(self):
        return self.f(self.xstar)


def synthetic_lp(N=2000, d=50, m=8, p=1.5, variant="noisy", rho=0.3,
                 zeta=0.0, mu=1e-2, seed=0, n_fit=None):
    """Synthetic clients: client i draws covariates around its own mean
    (shift size zeta) with its own scale; labels from one shared model."""
    rng = np.random.default_rng(seed)
    xt = rng.standard_normal(d)
    sizes = np.full(m, N // m)
    sizes[: N - sizes.sum()] += 1
    A_parts, groups, start = [], [], 0
    for i in range(m):
        shift = zeta * rng.standard_normal(d)
        scale = np.exp(0.5 * zeta * rng.standard_normal(d))
        A_parts.append(shift + scale * rng.standard_normal((sizes[i], d)))
        groups.append(np.arange(start, start + sizes[i]))
        start += sizes[i]
    A = np.vstack(A_parts) / np.sqrt(d)
    noise = rng.standard_t(df=2, size=N) * 0.5
    if variant == "exact":
        fit = rng.random(N) < rho
        if n_fit is not None:            # exactly n_fit fitted samples
            fit = np.zeros(N, bool)
            fit[rng.choice(N, size=n_fit, replace=False)] = True
        noise[fit] = 0.0
        b = A @ xt + noise
        return LpRegression(A, b, groups, p=p, mu=mu, variant="exact",
                            xstar_planted=xt)
    b = A @ xt + noise
    return LpRegression(A, b, groups, p=p, mu=mu, variant="noisy")


def diabetes_lp(m=4, p=1.5, mu=1e-2):
    """sklearn's bundled diabetes data (442 x 10), standardised; clients are
    contiguous age groups, so their covariate distributions differ."""
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    y = (y - y.mean()) / y.std()
    order = np.argsort(X[:, 0], kind="stable")
    X, y = X[order], y[order]
    groups = np.array_split(np.arange(len(y)), m)
    A = np.hstack([X, np.ones((len(y), 1))]) / np.sqrt(X.shape[1] + 1)
    return LpRegression(A, y, groups, p=p, mu=mu, variant="noisy")
