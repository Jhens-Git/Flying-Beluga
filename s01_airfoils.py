#!/usr/bin/env python3
"""
s01_airfoils.py -- XFOIL automation for a low-Re tailless airfoil.

 * computes the Reynolds-number band from your chords / speed
 * pulls candidate airfoils (UIUC database if online) + a parametric reflexed family it generates
 * drives XFOIL through subprocess (Ncrit 5 and 9, Re 50k..200k)
 * extracts Cm0, alpha_L0, CLmax, cd, L/D, transition location, ranks candidates for a flying wing
 * writes results/airfoil_summary.json (consumed by script 03) and results/airfoil_report.png

Run:  python3 s01_airfoils.py          (add --no-download to skip the UIUC fetch)
"""
import os, sys, json, shutil, subprocess, tempfile, itertools, urllib.request
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import config as C

XFOIL = shutil.which("xfoil")
XVFB = shutil.which("xvfb-run")   # Ubuntu's xfoil crashes (SIGFPE) with graphics off and needs an X server:
                                   #   sudo apt install xfoil xvfb xfonts-base xfonts-75dpi


# ---------------------------------------------------------------- Reynolds band
def reynolds(c, V=C.V_CRUISE):
    return C.RHO * V * c / C.MU


def print_re_band():
    print("== Reynolds numbers (V = %.0f m/s) ==" % C.V_CRUISE)
    for name, c in [("root 0.26 m", .26), ("MAC ~0.17 m", .17), ("tip 0.09 m", .09), ("tip 0.075 m", .075)]:
        print(f"  {name:12s} Re = {reynolds(c):9.0f}   (at 8 m/s: {reynolds(c, 8):8.0f})")
    print("  -> sweep Re = 50k..200k covers slow flight at the tip to cruise at the root\n")


# ---------------------------------------------------------------- geometry helpers
def reflex_airfoil(t=0.09, A=0.08, B=1.8, n=80):
    """NACA-4 thickness on a reflexed camber line  yc = A x(1-x)(1-Bx)  (B>1 -> reflexed TE)."""
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = A * x * (1 - x) * (1 - B * x)
    dy = A * ((1 - 2 * x) * (1 - B * x) - B * x * (1 - x))
    th = np.arctan(dy)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    X = np.concatenate([xu[::-1], xl[1:]]); Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y


def write_dat(path, name, X, Y):
    with open(path, "w") as f:
        f.write(name + "\n")
        for a, b in zip(X, Y):
            f.write(f"{a:.6f} {b:.6f}\n")


def build_candidates(download=True):
    cands = {}
    # parametric reflex family: thickness x camber amplitude x reflex factor
    for t, A, B in itertools.product([0.08, 0.10], [0.03, 0.06, 0.10], [1.6, 2.2, 3.0]):
        nm = f"RFX_t{int(t*100):02d}_A{int(A*100):02d}_B{int(B*10):02d}"
        p = os.path.join(C.AF_DIR, nm + ".dat")
        write_dat(p, nm, *reflex_airfoil(t, A, B))
        cands[nm] = p
    # a symmetric baseline for comparison
    X, Y = reflex_airfoil(0.09, 0.0, 1.0)
    p = os.path.join(C.AF_DIR, "SYM09.dat"); write_dat(p, "SYM09", X, Y); cands["SYM09"] = p
    # UIUC database
    if download:
        for n in C.UIUC_NAMES:
            p = os.path.join(C.AF_DIR, n.upper() + ".dat")
            if not os.path.exists(p):
                try:
                    data = urllib.request.urlopen(C.UIUC_BASE + n + ".dat", timeout=8).read().decode()
                    if len(data.splitlines()) > 20:
                        open(p, "w").write(data)
                except Exception:
                    continue
            if os.path.exists(p):
                cands[n.upper()] = p
    # anything you drop in results/airfoils/*.dat is picked up automatically
    for f in os.listdir(C.AF_DIR):
        if f.endswith(".dat"):
            cands.setdefault(f[:-4], os.path.join(C.AF_DIR, f))
    return cands


# ---------------------------------------------------------------- XFOIL driver
def run_xfoil(datfile, Re, ncrit, a0=-6.0, a1=11.0, da=1.0, timeout=40):
    if XFOIL is None:
        raise RuntimeError("xfoil not found:  sudo apt install xfoil")
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(datfile, os.path.join(td, "a.dat"))     # XFOIL truncates long file names -> use short relative ones
        pol = os.path.join(td, "p.txt")
        cmds = ["LOAD a.dat", "PANE", "OPER", f"VISC {Re:.0f}", "VPAR", f"N {ncrit}", "",
                "ITER 300", "PACC", "p.txt", "",
                f"ASEQ 0 {a1} {da}",         # sweep up from 0 (better convergence)
                "INIT", f"ASEQ -0.5 {a0} {da}",   # then down
                "PACC", "", "QUIT"]
        try:
            subprocess.run(([XVFB, "-a", XFOIL] if XVFB else [XFOIL]), input="\n".join(cmds) + "\n", text=True, capture_output=True, timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            pass
        if not os.path.exists(pol):
            return None
        rows = []
        for ln in open(pol).read().splitlines()[12:]:
            p = ln.split()
            if len(p) >= 7:
                try:
                    rows.append([float(v) for v in p[:7]])
                except ValueError:
                    pass
        if len(rows) < 6:
            return None
        arr = np.array(rows)                       # alpha CL CD CDp CM Xtr_top Xtr_bot
        return arr[np.argsort(arr[:, 0])]


def metrics(pol, cl_cruise=C.CRUISE_CL_GUESS):
    a, cl, cd, cm, xt, xb = pol[:, 0], pol[:, 1], pol[:, 2], pol[:, 4], pol[:, 5], pol[:, 6]
    if cl.max() < cl_cruise + 0.1 or (cl < cl_cruise).sum() < 3:
        return None
    lo = cl < 0.6                                           # attached-flow region: linear fits, extrapolated to CL = 0
    if lo.sum() < 3:
        return None
    pa = np.polyfit(a[lo], cl[lo], 1)                       # CL = k*alpha + b
    a0L = float(-pa[1] / pa[0]); cla = float(np.degrees(pa[0]))
    pm = np.polyfit(cl[lo], cm[lo], 1)
    o = np.argsort(cl); cl_s = cl[o]
    cm0 = float(np.interp(0, cl_s, cm[o])) if cl.min() < 0 < cl.max() else float(pm[1])   # Cm(c/4) at CL = 0
    cdc = float(np.interp(cl_cruise, cl_s, cd[o]))
    ld = cl / cd
    k = int(np.argmax(ld))
    return dict(cm0=cm0, alpha0L=a0L, cla2d=cla, clmax=float(cl.max()), cd_cruise=cdc,
                ld_cruise=cl_cruise / cdc, ld_max=float(ld[k]), cl_at_ldmax=float(cl[k]),
                xtr_top=float(np.interp(cl_cruise, cl_s, xt[o])), xtr_bot=float(np.interp(cl_cruise, cl_s, xb[o])),
                cdmin=float(cd.min()))


def job(args):
    name, path, Re, nc = args
    try:
        pol = run_xfoil(path, Re, nc)
        m = metrics(pol) if pol is not None else None
    except Exception as e:
        print(f"[{name} Re={Re}] {type(e).__name__}: {e}", flush=True)
        m = None
    return name, Re, nc, m


# ---------------------------------------------------------------- scoring
def score(m100_5, m60_5, _unused=None):
    """Flying-wing figure of merit (higher = better). Penalise nose-down Cm0, low clmax, poor low-Re robustness."""
    if m100_5 is None or m60_5 is None:
        return -1e9
    s = m100_5["ld_cruise"]                                # cruise L/D at Re 100k, turbulent outdoor air
    s += 0.5 * m60_5["ld_cruise"]                          # tip-ish Reynolds number robustness
    s -= 900 * max(0.0, -m100_5["cm0"] - 0.005)            # want Cm0 >= -0.005 (reflex); every -0.01 costs 9 L/D
    s -= 60 * max(0.0, 0.9 - m100_5["clmax"])
    return s


def main():
    if XFOIL is None:
        sys.exit("xfoil not found. Run: sudo apt install xfoil xvfb xfonts-base xfonts-75dpi")
    if XVFB is None:
        print("WARNING: xvfb-run not found; xfoil will likely crash. sudo apt install xvfb xfonts-base xfonts-75dpi")

    print_re_band()
    cands = build_candidates(download="--no-download" not in sys.argv)
    nw = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(cands)} candidate airfoils.")
    # ---- stage 1: quick screen at Re = 75k and 100k, Ncrit = 5 (outdoor turbulence)
    print("Stage 1: screening at Re 75k/100k, Ncrit 5 ...", flush=True)
    res = {}
    tasks = [(n, p, Re, 5) for n, p in cands.items() for Re in (75_000, 100_000)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, Re, nc, m in ex.map(job, tasks):
            res.setdefault(name, {})[f"{Re}_{nc}"] = m
    table = sorted(((score(d.get("100000_5"), d.get("75000_5"), None), n) for n, d in res.items()), reverse=True)
    # ---- stage 2: full Re x Ncrit grid for the best few
    top = [n for sc, n in table[:3] if sc > -1e8]
    print(f"Stage 2: full Re/Ncrit grid for {top} ...", flush=True)
    tasks = [(n, cands[n], Re, nc) for n in top for Re in C.RE_LIST for nc in C.NCRIT_LIST if f"{Re}_{nc}" not in res[n]]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, Re, nc, m in ex.map(job, tasks):
            res[name][f"{Re}_{nc}"] = m
    print("\nrank  airfoil              score  | Re=100k Ncrit=5:  Cm0     a0L   clmax  cd@CL.35  L/D   xtr_top")
    for i, (sc, n) in enumerate(table[:12]):
        m = res[n].get("100000_5")
        if m:
            print(f"{i+1:3d}   {n:20s} {sc:6.1f} |   {m['cm0']:+.4f} {m['alpha0L']:+5.1f}  {m['clmax']:.2f}   {m['cd_cruise']:.4f}  {m['ld_cruise']:5.1f}   {m['xtr_top']:.2f}")
    best = table[0][1]
    mb = res[best]["100000_5"]
    summ = dict(best=best, file=cands[best], Re_ref=100000, ncrit=5, cm0=mb["cm0"], alpha0L_deg=mb["alpha0L"],
                clmax=mb["clmax"], cd_cruise=mb["cd_cruise"], cla2d=mb["cla2d"], ld_cruise=mb["ld_cruise"],
                ranking=[n for _, n in table[:10]], all_metrics={n: res[n] for n in top},
                candidates={n: dict(file=cands[n], cm0=res[n]["100000_5"]["cm0"], alpha0L_deg=res[n]["100000_5"]["alpha0L"],
                                    clmax=res[n]["100000_5"]["clmax"], cd_cruise=res[n]["100000_5"]["cd_cruise"],
                                    cd_cruise_75k=res[n]["75000_5"]["cd_cruise"])
                            for _, n in table[:8] if res[n].get("100000_5") and res[n].get("75000_5")})
    json.dump(summ, open(os.path.join(C.RES, "airfoil_summary.json"), "w"), indent=1)
    shutil.copy(cands[best], os.path.join(C.RES, "best_airfoil.dat"))
    print(f"\n>>> best: {best}  (Cm0={mb['cm0']:+.4f}, alpha_L0={mb['alpha0L']:+.1f} deg, clmax={mb['clmax']:.2f})")
    print("    saved results/airfoil_summary.json and results/best_airfoil.dat")
    try:
        plot(res, table[:4], cands)
    except Exception as e:
        print("plot skipped:", e)


def plot(res, top, cands):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for sc, n in top:
        pol = run_xfoil(cands[n], 100000, 5)
        if pol is None: continue
        ax[0].plot(pol[:, 2], pol[:, 1], label=n); ax[1].plot(pol[:, 0], pol[:, 4], label=n); ax[2].plot(pol[:, 1], pol[:, 1] / pol[:, 2])
        X = np.loadtxt(cands[n], skiprows=1)
    ax[0].set(xlabel="CD", ylabel="CL", title="Drag polar, Re=100k, Ncrit=5"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="alpha [deg]", ylabel="Cm (c/4)", title="Pitching moment"); ax[1].axhline(0, c="k", lw=.5)
    ax[2].set(xlabel="CL", ylabel="CL/CD", title="Efficiency")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.RES, "airfoil_report.png"), dpi=120)


if __name__ == "__main__":
    main()
