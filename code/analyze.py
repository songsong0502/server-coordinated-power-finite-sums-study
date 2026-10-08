"""Figures and tables for the paper, from results/*.json and *.npz."""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.2,
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.2,
                     "font.family": "serif", "mathtext.fontset": "cm",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
METH = ["GD-1/L", "GD-tuned", "AGD", "AdGD", "UGM", "UFGM"]
COL = {"GD-1/L": "#1f77b4", "GD-tuned": "#17becf", "AGD": "#9467bd",
       "AdGD": "#d62728", "UGM": "#2ca02c", "UFGM": "#ff7f0e"}
LS = {"GD-1/L": "-", "GD-tuned": "--", "AGD": "-.", "AdGD": "-", "UGM": "--",
      "UFGM": ":"}
TAB = {}
FLOOR = 1e-17


def clip(e):
    return np.maximum(np.asarray(e, float), FLOOR)


# ---------------------------------------------------------------- E1
def e1():
    v = json.load(open(os.path.join(RES, "e1_validation.json")))
    cv = np.load(os.path.join(RES, "e1_curves.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for t, c in zip((0.01, 0.005, 0.001, 0.0005), ("C0", "C1", "C2", "C3")):
        ax[0].semilogy(cv[f"tau{t}"], color=c, lw=0.9, label=rf"$\tau={t}$")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\|x_k-x^\star\|$")
    ax[0].set_title(r"(a) $\alpha=0.5$"); ax[0].legend(loc="upper right")
    for a, c in zip((0.2, 0.4, 0.6, 0.8), ("C0", "C1", "C2", "C3")):
        ax[1].semilogy(cv[f"alpha{a}"], color=c, lw=0.9, label=rf"$\alpha={a}$")
    ax[1].set_xlabel("iteration"); ax[1].set_title(r"(b) $\tau=10^{-3}$")
    ax[1].legend(loc="upper right")
    mk = {"0.3": "o", "0.5": "s", "0.7": "^"}
    for row, c in zip(v["b"], ("C0", "C1", "C2")):
        t = np.array(row["taus"]); L = np.array(row["levels"])
        ax[2].loglog(t, L, mk[str(row["alpha"])], color=c, ms=4,
                     label=rf"$\alpha={row['alpha']}$: slope {row['slope']:.2f}")
        mid = len(t) // 2
        ax[2].loglog(t, L[mid] * (t / t[mid]) ** row["predicted_slope"], "--",
                     color=c, lw=0.8)
    ax[2].set_xlabel(r"step size $\tau$"); ax[2].set_ylabel("plateau level")
    ax[2].set_title("(c) plateau against step"); ax[2].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_validation.pdf")); plt.close(fig)
    TAB["e1"] = v
    TAB["e1_dist"] = json.load(open(os.path.join(RES, "e1_dist.json")))
    TAB["aux"] = json.load(open(os.path.join(RES, "aux.json")))


# ---------------------------------------------------------------- E2
def family(key):
    if key.startswith("P1"):
        return "P1-T" if "-T-" in key else "P1-G"
    if key.startswith("P3s"):
        return "P3-exact" if key.endswith("exact") else "P3-noisy"
    return "P3-noisy"


def profiles(js, tol_key, ax, title):
    keys = list(js)
    T = np.array([[js[k]["methods"][m][tol_key] for m in METH] for k in keys], float)
    ok = np.isfinite(T).any(1)
    T = T[ok]
    best = np.nanmin(np.where(np.isfinite(T), T, np.nan), axis=1)
    ratio = T / best[:, None]
    taus = np.logspace(0, 2.5, 400)
    for j, m in enumerate(METH):
        rho = [(ratio[:, j] <= t).mean() for t in taus]
        ax.semilogx(taus, rho, color=COL[m], ls=LS[m], label=m)
    ax.set_xlabel(r"performance ratio $\theta$")
    ax.set_ylabel(r"fraction of problems $\rho_s(\theta)$")
    ax.set_ylim(0, 1.02); ax.set_title(title + f" ({ok.sum()} problems)")
    return dict(nprob=int(ok.sum()),
                wins={m: float((ratio[:, j] == 1).mean()) for j, m in enumerate(METH)},
                solved={m: float(np.isfinite(ratio[:, j]).mean()) for j, m in enumerate(METH)})


def e2():
    js = json.load(open(os.path.join(RES, "e2.json")))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    p3 = profiles(js, "r3", ax[0], r"(a) $10^{-3}$")
    p6 = profiles(js, "r6", ax[1], r"(b) $10^{-6}$")
    ax[1].legend(loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_profiles.pdf")); plt.close(fig)
    TAB["e2_profiles"] = dict(r3=p3, r6=p6)
    # per-family table
    fam = defaultdict(list)
    for k, v in js.items():
        fam[family(k)].append(v)
    rows = {}
    for f, lst in fam.items():
        r = {}
        for m in METH:
            r6 = np.array([x["methods"][m]["r6"] for x in lst])
            r3 = np.array([x["methods"][m]["r3"] for x in lst])
            best = np.array([x["methods"][m]["best"] for x in lst])
            r[m] = dict(n=len(lst), solved6=int(np.isfinite(r6).sum()),
                        solved3=int(np.isfinite(r3).sum()),
                        med_r6=float(np.median(r6[np.isfinite(r6)])) if np.isfinite(r6).any() else None,
                        med_r3=float(np.median(r3[np.isfinite(r3)])) if np.isfinite(r3).any() else None,
                        med_best=float(np.median(best)))
        rows[f] = r
    TAB["e2_family"] = rows
    TAB["e2_tau_mult"] = {k: v["methods"]["GD-tuned"]["tau_mult"] for k, v in js.items()}
    TAB["e2_trials"] = {k: {m: v["methods"][m]["trials"] for m in ("UGM", "UFGM")}
                        for k, v in js.items()}
    # curves for four instances
    cv = np.load(os.path.join(RES, "e2_curves.npz"))
    inst = [("P1-n50-a0.4-T-s0", r"(a) P1, $\alpha=0.4$, kink-touching"),
            ("P1-n50-a0.4-G-s0", r"(b) P1, $\alpha=0.4$, generic"),
            ("P3s-d50-p1.4-exact", r"(c) P3 exact, $p=1.4$"),
            ("P3s-d50-p1.4-noisy", r"(d) P3 noisy, $p=1.4$")]
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
    for a, (k, t) in zip(ax, inst):
        for m in METH:
            e = clip(cv[f"{k}|{m}"])
            a.semilogy(np.arange(len(e)), e, color=COL[m], ls=LS[m], lw=0.9, label=m)
        a.set_xscale("symlog", linthresh=10)
        a.set_xlim(0, 20000); a.set_title(t, fontsize=7.6); a.set_xlabel("rounds")
        a.set_ylim(1e-17, 3)
    ax[0].set_ylabel("relative error")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_curves.pdf")); plt.close(fig)


# ---------------------------------------------------------------- E2b
def e2b():
    d = json.load(open(os.path.join(RES, "e2b.json")))
    g = defaultdict(list)
    for r in d:
        g[(r["fam"], r["k"])].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    out = {}
    for j, (fam, scale, xl) in enumerate((("P3", 50.0, r"fitted samples $k/d$"),
                                          ("P1", 50.0, r"kink coordinates $k/n$"))):
        ks = sorted(k for f, k in g if f == fam)
        for m in ("GD-1/L", "AdGD", "UGM", "UFGM"):
            med = [np.median([x["res"][m]["floor"] for x in g[(fam, k)]]) for k in ks]
            lo = [min(x["res"][m]["floor"] for x in g[(fam, k)]) for k in ks]
            hi = [max(x["res"][m]["floor"] for x in g[(fam, k)]) for k in ks]
            xk = np.array(ks) / scale
            if fam == "P1":
                xk = np.where(xk == 0, 0.0, xk)
            ax[j].semilogy(xk, clip(med), marker="o", ms=3, color=COL[m], ls=LS[m], label=m)
            ax[j].fill_between(xk, clip(lo), clip(hi), color=COL[m], alpha=0.12, lw=0)
            out[f"{fam}|{m}"] = dict(k=ks, med=[float(v) for v in med])
        ax[j].axvline(1.0, color="k", lw=0.6, ls=":")
        ax[j].set_xlabel(xl); ax[j].set_ylim(1e-17, 1)
        if fam == "P3":
            ax[j].set_xscale("log")
    ax[0].set_ylabel("final error level"); ax[0].set_title("(a) P3 exact, $d=50$, $p=1.4$")
    ax[1].set_title(r"(b) P1, $n=50$, $\alpha=0.4$"); ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_kinkrank.pdf")); plt.close(fig)
    TAB["e2b"] = out
    TAB["e2b_r6"] = {f"{f}|{k}": [x["res"]["AdGD"]["r6"] for x in v] for (f, k), v in g.items()}


# ---------------------------------------------------------------- E3
def e3():
    js = json.load(open(os.path.join(RES, "e3.json")))
    cv = np.load(os.path.join(RES, "e3_curves.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4))
    Ks = [1, 2, 4, 8, 16, 32]
    sty = {"0": ("C0", "o"), "0.3": ("C1", "s"), "1": ("C2", "^"), "3": ("C3", "v")}
    for z, (c, mk) in sty.items():
        tag = f"P1z{z}"
        for meth, ls in (("FedAvg", "-"), ("SCAFFOLD", "--")):
            y = [next(r["floor"] for r in js["data"] if r["tag"] == tag and r["meth"] == meth and r["K"] == K) for K in Ks]
            ax[0].loglog(Ks, y, ls=ls, marker=mk, ms=3, color=c,
                         label=(rf"$\zeta={z}$" if meth == "FedAvg" else None))
    ax[0].set_xlabel("local steps $K$"); ax[0].set_ylabel("final error level")
    ax[0].set_title("(a) P1 data partition"); ax[0].legend(loc="center right", fontsize=6.5)
    ax[0].text(1.6, 3e-4, "dashed: SCAFFOLD", fontsize=6.5)
    for tag, c, nm in (("P3n", "C0", "noisy"), ("P3x", "C3", "exact")):
        for meth, ls, mk in (("FedAvg", "-", "o"), ("SCAFFOLD", "--", "s")):
            y = [next(r["floor"] for r in js["data"] if r["tag"] == tag and r["meth"] == meth and r["K"] == K) for K in Ks]
            ax[1].loglog(Ks, clip(y), ls=ls, marker=mk, ms=3, color=c, label=f"{nm}, {meth}")
    ax[1].set_xlabel("local steps $K$"); ax[1].set_title("(b) P3, $p=1.4$, $d=50$")
    ax[1].legend(fontsize=6.2, loc="center right"); ax[1].set_ylim(1e-16, 1)
    for tag, c, lab, Kset in (("P2a5", "C0", r"P2 $\alpha=0.5$", (1, 4, 16, 64)),
                              ("P1c", "C3", r"P1", (1, 4))):
        for K, ls in ((1, "-"), (4, "--"), (16, "-."), (64, ":")):
            if K not in Kset:
                continue
            e = np.asarray(cv[f"{tag}|BJ|{K}"], float)
            e = np.where(np.isfinite(e), e, np.nan)
            ax[2].semilogy(clip(e[:1500]), color=c, ls=ls, lw=0.9, label=f"{lab}, K={K}")
    ax[2].set_xlabel("rounds"); ax[2].set_title("(c) block-local steps")
    ax[2].set_ylim(1e-17, 1e4); ax[2].legend(fontsize=6.2, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_local.pdf")); plt.close(fig)
    TAB["e3"] = js


# ---------------------------------------------------------------- E4
def e4():
    js = json.load(open(os.path.join(RES, "e4.json")))
    cv = np.load(os.path.join(RES, "e4_curves.npz"))
    g = defaultdict(list)
    for r in js:
        g[(r["tag"], r["meth"], r["s"])].append(r)
    summ = {}
    for k, v in g.items():
        fl = np.array([x["floor"] for x in v])
        summ["|".join(map(str, k))] = dict(n=len(v), med=float(np.median(fl)),
                                           lo=float(fl.min()), hi=float(fl.max()),
                                           ep3=float(np.median([x["ep3"] for x in v])),
                                           ep6=float(np.median([x["ep6"] for x in v])))
    TAB["e4"] = summ
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.3))
    cols = {8: "C0", 4: "C1", 2: "C2", 1: "C3"}
    panels = (("P1z1", "SCAFFOLD", r"(a) P1 ($\zeta=1$), SCAFFOLD $K=8$"),
              ("P3n", "FedAvg", r"(b) P3 noisy, FedAvg $K=1$"),
              ("P2a2", "Block", r"(c) P2 $\alpha=0.2$, block sampling"))
    for a, (tag, meth, t) in zip(ax, panels):
        for s in (8, 4, 2, 1):
            ep = cv[f"{tag}|{meth}|{s}|ep"]; e = clip(cv[f"{tag}|{meth}|{s}|err"])
            a.semilogy(ep, e, color=cols[s], lw=0.9, label=f"$s={s}$")
        a.set_xlabel("epochs (equal work)"); a.set_title(t, fontsize=7.6)
    ax[0].set_ylabel("error"); ax[0].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_sampling.pdf")); plt.close(fig)


if __name__ == "__main__":
    e1(); e2(); e2b(); e3(); e4()
    with open(os.path.join(RES, "tables.json"), "w") as fh:
        json.dump(TAB, fh, indent=1, default=float)
    print(json.dumps({k: TAB[k] for k in ("e2_profiles", "e2_family")}, indent=1, default=float))
