#!/usr/bin/env python3
"""
s07_ribs.py -- rib spacing / thickness / spar-hole fit for the 3D-printed PLA ribs with two 4x4 mm CF rods per half-wing.

Checks, rib by rib:
  1. FIT: local airfoil thickness at the front and rear rod positions - 4 mm rod = material left for printed skin (>= SKIN_WALL_MIN each side)
  2. LOAD: local lift per rib at ultimate load -> bearing stress on the rod hole (almost never governs)
  3. MASS: rib mass with lightening cut-outs vs the budget
Also searches the rear-rod offset behind the front rod (parallel rods) for the biggest spacing that fits everywhere.
Run after s02 (needs results/structure.json + design.json + best_airfoil.dat).
"""
import os, json
import numpy as np
import config as C
from geometry import Planform


def load_airfoil(path):
    X = np.loadtxt(path, skiprows=1)
    i = int(np.argmin(X[:, 0]))
    up, lo = X[: i + 1][::-1], X[i:]
    xs = np.linspace(0, 1, 201)
    yu = np.interp(xs, up[:, 0], up[:, 1]); yl = np.interp(xs, lo[:, 0], lo[:, 1])
    return xs, yu - yl, 0.5 * (yu + yl)


def main():
    D = json.load(open(os.path.join(C.RES, "design.json")))
    S2 = json.load(open(os.path.join(C.RES, "structure.json")))
    p = D["planform"]
    P = Planform(**{k: p[k] for k in ("span", "c_root", "c_kink", "c_tip", "yk_frac", "sweep_in", "sweep_out")}, twist_tip=0.0)
    s = P.s
    xs, th, cam = load_airfoil(D.get("airfoil_file") or os.path.join(C.RES, "best_airfoil.dat"))
    lam, x0, reach = S2["spar_sweep_deg"], S2["spar_x0"], S2["spar_reach"]
    b = C.ROD_B
    ribs = np.arange(C.RIB_PITCH / 2, s, C.RIB_PITCH)
    print(f"Airfoil max thickness {th.max()*100:.1f}% c at x/c={xs[np.argmax(th)]:.2f}; {len(ribs)} ribs per side at {C.RIB_PITCH*1000:.0f} mm pitch")
    print(f"Front rod: sweep {lam:.0f} deg, root x0 {x0*1000:.0f} mm, reaches y={reach*1000:.0f} mm\n")

    def fit(d):
        """spar clearance table for rear rod d metres behind the front rod (parallel)"""
        rows = []
        for y in ribs:
            c = P.chord(y); xle = P.xle(y)
            xf = x0 + np.tan(np.radians(lam)) * y
            for name, xr, on in (("front", xf, y <= reach), ("rear", xf + d, y <= reach)):
                xc = (xr - xle) / c
                if not on: continue
                if not 0.05 < xc < 0.95:
                    rows.append((y, name, xc, -1.0, -1.0)); continue
                t_loc = np.interp(xc, xs, th) * c
                wall = (t_loc - b - 0.2e-3) / 2           # 0.2 mm hole clearance
                rows.append((y, name, xc, t_loc, wall))
        return rows

    print("rear-rod offset search (parallel to front rod):")
    best = None
    for d in np.arange(C.REAR_SPAR_SPACING_RANGE[0], C.REAR_SPAR_SPACING_RANGE[1] + 1e-9, 0.01):
        rows = fit(d)
        xc_max = max(r[2] for r in rows if r[1] == "rear")
        wmin = min(r[4] for r in rows)
        okk = wmin >= C.SKIN_WALL_MIN and xc_max <= C.REAR_SPAR_MAX_XC
        print(f"   {d*1000:4.0f} mm: min printed wall {wmin*1000:5.2f} mm, rear rod up to {xc_max*100:3.0f}% chord  {'OK' if okk else 'no'}")
        if okk: best = d
    if best is None:
        print("\n   >> No parallel rear rod fits the whole span. Options: (a) end the rear rod where the wall hits the limit (it only needs to reach ~60-70% span),"
              "\n      (b) thicker airfoil outboard, (c) put both rods in the front 30-40% chord side by side, (d) use 3 mm rod outboard.")
        d = 0.05
        for lim in (0.5, 0.6, 0.7, 0.8, 0.9):
            rows = [r for r in fit(d) if r[1] == "rear" and r[0] <= lim * s]
            if rows: print(f"      rear rod {d*1000:.0f} mm behind front, ending at {lim*100:.0f}% semispan: min wall {min(r[4] for r in rows)*1000:.2f} mm")
    else:
        d = best
        print(f"\n   -> use rear rod {d*1000:.0f} mm behind the front rod")
    rows = fit(d)
    print(f"\nrib-by-rib at rear offset {d*1000:.0f} mm (y, rod, x/c, local thickness, printed wall each side):")
    bad = [r for r in rows if r[4] < C.SKIN_WALL_MIN]
    for r in rows[:: max(1, len(rows) // 14)]:
        print(f"   y={r[0]*1000:5.0f}  {r[1]:5s}  x/c={r[2]:.2f}  t={r[3]*1000:5.1f} mm  wall={r[4]*1000:5.2f} mm {'<-- too thin' if r[4] < C.SKIN_WALL_MIN else ''}")
    print(f"   {len(bad)} rod holes below the {C.SKIN_WALL_MIN*1000:.1f} mm wall limit")
    # loads
    W = D["mass"] * C.G
    L_half = C.N_LIMIT * C.FOS_ULT * W / 2
    l_loc = lambda y: 4 * L_half / (np.pi * s) * np.sqrt(max(0, 1 - (y / s) ** 2))
    P_rib = l_loc(ribs[0]) * C.RIB_PITCH
    print(f"\nLoad per rib at the root at ultimate: {P_rib:.2f} N  (split over 2 rods -> {P_rib/2:.2f} N each)")
    print("thickness option -> rod-hole bearing stress (vs %.0f MPa allowable)" % (C.SIGMA_BEARING_PLA / 1e6))
    for t in C.RIB_T_OPTIONS:
        print(f"   {t*1000:.1f} mm: {P_rib/2/(b*t)/1e6:5.2f} MPa   -> margin x{C.SIGMA_BEARING_PLA/(P_rib/2/(b*t)):.0f}")
    # mass
    print("\nrib mass (all %d ribs, both sides, lightening %.0f%% removed):" % (2 * len(ribs), 100 * C.LIGHTENING_REMOVED))
    for t in C.C_RIB if False else C.RIB_T_OPTIONS:
        m = 0
        for y in ribs:
            c = P.chord(y)
            m += np.trapezoid(th, xs) * c * c * t * C.RHO_PLA * (1 - C.LIGHTENING_REMOVED) if hasattr(np, "trapezoid") else np.trapz(th, xs) * c * c * t * C.RHO_PLA * (1 - C.LIGHTENING_REMOVED)
        print(f"   {t*1000:.1f} mm: {2*m*1000:5.1f} g   (budget in config: PLA+Monokote {C.STRUCT_AREAL_KG_M2*D['S']*1000:.0f} g total)")
    print("\nRecommendation: 1.0 mm general ribs (2.5 perimeters -> print as 0.8 or 1.2), 1.6 mm at the kink, elevon ends and hinge ribs,")
    print("2-3 mm root ribs with bolt bosses and square rod pockets 4.2 mm. Strength is not the driver: fit, printability and warping are.")
    print("PLA warning: Monokote shrinking (~100-120 C) is above PLA's ~55-60 C glass transition -> iron at the LOWEST setting that")
    print("shrinks, keep ribs supported, or print ribs in PETG / CF-PETG.")
    json.dump(dict(rear_offset_m=float(d), holes_too_thin=len(bad), n_ribs_per_side=len(ribs)), open(os.path.join(C.RES, "ribs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
