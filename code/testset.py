"""The 52-instance test set used for the synchronous comparison (E2)."""
from problems import RandomPower, synthetic_lp, diabetes_lp


def instances():
    out = []
    for n in (50, 200):
        for alpha in (0.2, 0.4, 0.6, 0.8):
            for touch in (True, False):
                for seed in (0, 1):
                    nz = n // 10 if touch else 0
                    key = f"P1-n{n}-a{alpha}-{'T' if touch else 'G'}-s{seed}"
                    out.append((key, dict(kind="P1", n=n, alpha=alpha,
                                          n_zero=nz, seed=seed)))
    for d in (20, 50):
        for p in (1.2, 1.4, 1.6, 1.8):
            for var in ("exact", "noisy"):
                key = f"P3s-d{d}-p{p}-{var}"
                out.append((key, dict(kind="P3s", d=d, p=p, variant=var)))
    for p in (1.2, 1.4, 1.6, 1.8):
        out.append((f"P3d-p{p}", dict(kind="P3d", p=p)))
    return out


def build(spec):
    k = spec["kind"]
    if k == "P1":
        return RandomPower(n=spec["n"], alpha=spec["alpha"],
                           n_zero=spec["n_zero"], m=8, seed=spec["seed"])
    if k == "P3s":
        return synthetic_lp(N=2000, d=spec["d"], m=8, p=spec["p"],
                            variant=spec["variant"], zeta=1.0, seed=0)
    if k == "P3d":
        return diabetes_lp(m=4, p=spec["p"])
    raise ValueError(k)
