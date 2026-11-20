#!/usr/bin/env python3
"""
s02_structure.py -- 4x4 mm CF rod spar sizing for the 800 mm Beluga.

 * limit-load root bending moment from an elliptical lift distribution (n = N_LIMIT, ultimate = x FOS)
 * I = b^4/12 = 21.33 mm^4, bending stress, safety factor, tip deflection (numerical double integration)
 * two rod layouts:  A) butt joint + sleeve at the centreline,  B) side-by-side overlap (2 x 400 mm - overlap)
 * spar-line fit: because the planform is swept, a straight rod cannot stay at ~30 % chord  -> V-shaped spar,
   angled root joiner. Torsion that the swept rod sees is checked (square CF rods are weak in torsion!)
Reads results/design.json if present (script 03), otherwise uses the numbers below.
"""
import os, json
import numpy as np
import config as C

_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_design():
    p = os.path.join(C.RES, "design.json")
    if os.path.exists(p):
        return json.load(open(p))
    return dict(mass=0.30, planform=dict(span=0.80, c_root=0.26, c_kink=0.20, c_tip=0.09, yk_frac=0.25, sweep_in=45, sweep_out=25),
                S=0.135, xcg=0.14)


def elliptic_moment(s, L_half, y):
    """bending moment M(y) at station y from an elliptical load of total L_half over [0,s]"""
    eta = np.linspace(0, s, 4001)
    l = 4 * L_half / (np.pi * s) * np.sqrt(np.maximum(0, 1 - (eta / s) ** 2))
    M = np.zeros_like(y)
    for i, yy in enumerate(y):
        m = eta >= yy
        M[i] = _trapz(l[m] * (eta[m] - yy), eta[m]) if m.sum() > 1 else 0.0
    return M


def deflection(y, M, EI):
    kappa = M / EI
    theta = np.concatenate([[0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(y))])
    delta = np.concatenate([[0], np.cumsum(0.5 * (theta[1:] + theta[:-1]) * np.diff(y))])
    return theta, delta


def main():
    d = load_design()
    pf = d["planform"]; s = pf["span"] / 2
    W = d["mass"] * C.G
    b = C.ROD_B
    I = b ** 4 / 12; Z = I / (b / 2); A = b * b
    print(f"Rod 4x4 mm: I = {I*1e12:.2f} mm^4 (=b^4/12),  Z = {Z*1e9:.2f} mm^3,  EI = {C.E_CF*I:.2f} N m^2,  mass {C.RHO_CF*A*C.ROD_LEN*1000:.1f} g each")
    print(f"AUW {d['mass']*1000:.0f} g, semispan {s*1000:.0f} mm\n")
    for label, n in (("limit", C.N_LIMIT), ("ultimate", C.N_LIMIT * C.FOS_ULT)):
        L_half = n * W / 2
        M_root = L_half * 4 * s / (3 * np.pi)
        sig = M_root / Z
        print(f"[{label:8s}] n={n:.1f}  L_half={L_half:.2f} N  M_root={M_root:.3f} N m   sigma={sig/1e6:6.1f} MPa   SF vs {C.SIGMA_ALLOW/1e6:.0f} MPa = {C.SIGMA_ALLOW/sig:.1f}")
    n = C.N_LIMIT; L_half = n * W / 2
    out = {}
    # ------------------------------------------------ layouts
    y = np.linspace(0, s, 801)
    M = elliptic_moment(s, L_half, y)
    EI = C.E_CF * I
    print("\nLayout A: rods butt at the centreline, bonded sleeve (single rod EI along the whole span)")
    spar_reach = min(C.ROD_LEN, s)
    th, de = deflection(y, M, EI * np.ones_like(y))
    print(f"   tip deflection at limit load  = {de[-1]*1000:.1f} mm   ({de[-1]/s*100:.1f}% of semispan)")
    Lb = 0.030
    M_ult = M[0] * C.FOS_ULT
    F = M_ult / Lb
    print(f"   sleeve: each rod inserted {Lb*1000:.0f} mm -> couple force at ultimate ~ {F:.0f} N, glue shear ~ {F/(2*b*Lb)/1e6:.2f} MPa (2 faces), keep sleeve walls >=1.2 mm")
    out["A"] = dict(tip_defl_mm=de[-1] * 1000)
    Lo = 0.100
    print(f"\nLayout B: rods overlap {Lo*1000:.0f} mm side-by-side (chordwise) -> spanwise spar length {2*C.ROD_LEN*1000-Lo*1000:.0f} mm, EI doubled in the overlap")
    reach = (2 * C.ROD_LEN - Lo) / 2
    yB = np.linspace(0, min(reach, s), 801)
    MB = elliptic_moment(s, L_half, yB)
    EIB = np.where(yB < Lo / 2, 2 * EI, EI)
    thB, deB = deflection(yB, MB, EIB)
    tip = deB[-1] + thB[-1] * max(0, s - reach)
    print(f"   tip deflection at limit load  = {tip*1000:.1f} mm; stress at y={Lo/2*1000:.0f} mm (end of overlap, single rod) = {np.interp(Lo/2,yB,MB)/Z/1e6:.0f} MPa limit")
    tau = 0.125 * (np.interp(Lo / 2, yB, MB) * C.FOS_ULT / Z) * b / Lo
    print(f"   overlap bond shear (order of magnitude) ~ {tau/1e6:.2f} MPa -> epoxy/CA+kicker is fine; use 30 mm min; wrap with glass/carbon tow")
    out["B"] = dict(tip_defl_mm=tip * 1000)
    # ------------------------------------------------ spar line fit (V-shaped spar)
    from geometry import Planform
    P = Planform(**{k: pf[k] for k in ("span", "c_root", "c_kink", "c_tip", "yk_frac", "sweep_in", "sweep_out")}, twist_tip=0.0)
    mp = os.path.join(C.RES, "mass_props.json")
    x_front = (json.load(open(mp))["battery_x"] - C.BATT_LEN / 2) if os.path.exists(mp) else 0.06
    f_t = 0.25
    best = None
    rows = []
    for lam_c in np.arange(0, 41, 1.0):
        reach = min(0.92 * s, C.ROD_LEN * np.cos(np.radians(lam_c)))     # rod axis length is limited to 400 mm
        yy_c = np.linspace(0.0, reach, 50)
        for x0_c in np.arange(0.010, 0.200, 0.002):
            if x0_c > x_front - 0.006:                                       # spar must clear the battery's front edge at the root
                break
            frac_c = (x0_c + yy_c * np.tan(np.radians(lam_c)) - P.xle(yy_c)) / P.chord(yy_c)
            if frac_c.min() < 0.12 or frac_c.max() > 0.42:
                continue
            cost = np.sqrt(np.mean((frac_c - f_t) ** 2)) + 1.0 * max(0.0, (s - reach) / s - 0.08)
            if best is None or cost < best[0]:
                best = (cost, lam_c, x0_c, reach)
        if best and abs(best[1] - lam_c) < 1e-9:
            rows.append(best)
    if best is None:
        print("\nNo straight-rod spar line satisfies 12-42% chord + battery clearance -> move the battery aft or loosen limits.")
        return
    _, lam, x0, reach = best
    yy = np.linspace(0.0, reach, 60)
    xs = x0 + np.tan(np.radians(lam)) * yy
    frac = (xs - P.xle(yy)) / P.chord(yy)
    print(f"\nSpar line (swept rod, best fit to {f_t*100:.0f}% chord, 12..42% allowed, clear of battery front edge x={x_front*1000:.0f} mm):")
    print(f"   rod sweep {lam:.0f} deg (V-shape in plan view), root x0 = {x0*1000:.0f} mm from nose datum")
    print(f"   spar x/c over y = 0..{reach*1000:.0f} mm: {frac.min()*100:.0f}% .. {frac.max()*100:.0f}%   (root {frac[0]*100:.0f}%, outboard {frac[-1]*100:.0f}%)")
    print(f"   400 mm rod axis reaches y = {reach*1000:.0f} mm of the {s*1000:.0f} mm semispan -> {max(0,(s-reach))*1000:.0f} mm tip stub is foam/PLA-only (fine: it carries almost no load)")
    print(f"   => rods meet at the centreline at ±{lam:.0f} deg: print an ANGLED joiner (included angle {180-2*lam:.0f} deg in plan), not a straight sleeve.")
    out_spar = dict(spar_sweep_deg=float(lam), spar_x0=float(x0), spar_reach=float(reach))
    # bending + torsion in a swept rod
    M_ult = C.N_LIMIT * W / 2 * 4 * s / (3 * np.pi) * C.FOS_ULT
    Mb = M_ult * np.cos(np.radians(lam)); T = M_ult * np.sin(np.radians(lam))
    tau_t = T / (0.208 * b ** 3)
    print(f"\nTorsion in the swept rod at ultimate (rod-only, worst case): T = {T:.3f} N m  -> tau = {tau_t/1e6:.0f} MPa  "
          f"({'above' if tau_t>C.TAU_ALLOW else 'below'} the ~{C.TAU_ALLOW/1e6:.0f} MPa interlaminar shear allowable)")
    print(f"   bending about rod axis: sigma = {Mb/Z/1e6:.0f} MPa")
    print("   -> let the PLA leading-edge D-box/ribs + Monokote carry torsion; do NOT rely on the square rod for twist.")
    # lift-line eccentricity torsion
    ecc = (P.xle(yy) + 0.25 * P.chord(yy)) - xs   # lift line minus spar line
    print(f"   lift (quarter-chord) line is {ecc.min()*1000:+.0f} .. {ecc.max()*1000:+.0f} mm from the spar -> small twisting moments; washout helps, as long as ribs are stiff in torsion")
    out.update(out_spar); out.update(spar_xc_range=[float(frac.min()), float(frac.max())], tau_torsion_MPa=tau_t / 1e6)
    json.dump(out, open(os.path.join(C.RES, "structure.json"), "w"), indent=1)
    print("\nNOTE: a spanwise spar cannot pass through the battery AND the rear motor. The spar line above passes just ahead of the")
    print("      battery's front edge at the root; the motor/ESC sit behind on a short printed boom tied into the root rib.")


if __name__ == "__main__":
    main()
