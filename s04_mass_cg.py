#!/usr/bin/env python3
"""
s04_mass_cg.py -- mass budget, CG, inertias, battery placement and CG-sensitivity for the optimised Beluga.

Honest physics note: a LiPo does NOT get lighter as it discharges (the mass is the same at 12.6 V and 9.6 V),
so 'battery consumption' does not move the CG. What *does* move the CG: (a) where you strap the pack,
(b) which pack you fly (650 / 850 / 1000 mAh), (c) camera/GPS mounting changes. Those are what is modelled.

Reads results/design.json (script 03) -> writes results/mass_props.json (used by AVL + state-space scripts).
"""
import os, json
import numpy as np
import config as C
import massmodel as MM
from geometry import Planform


def main():
    d = json.load(open(os.path.join(C.RES, "design.json")))
    p = d["planform"]
    pf = Planform(**p)
    xnp, MAC, xle = d["xnp"], d["MAC"], d["xle_mac"]
    xcg_t = d["xcg"]
    xb = MM.battery_x_for_cg(pf, xcg_t)
    lo, hi = MM.batt_limits(pf)
    print(f"Target CG {xcg_t*1000:.1f} mm  (SM {C.STATIC_MARGIN*100:.0f}%)   -> battery centre x = {xb*1000:.1f} mm ; allowed {lo*1000:.0f}..{hi*1000:.0f} mm")
    if not lo <= xb <= hi:
        print("   !! outside nose-bay limits -> add nose ballast or move heavy items (see ballast line below)")
    xb_c = float(np.clip(xb, lo, hi))
    L = MM.lumps(pf, xb_c)
    pr = MM.props(L)
    sm = (xnp - pr["cg"][0]) / MAC
    print(f"\nWith battery at {xb_c*1000:.1f} mm:  mass = {pr['m']*1000:.0f} g, CG = ({pr['cg'][0]*1000:.1f}, {pr['cg'][1]*1000:.1f}, {pr['cg'][2]*1000:.1f}) mm, static margin = {sm*100:.1f}% MAC")
    print(f"Inertias about CG [kg m^2]: Ixx={pr['Ixx']:.5f}  Iyy={pr['Iyy']:.5f}  Izz={pr['Izz']:.5f}  Ixz={pr['Ixz']:.6f}")
    # ---- sensitivities
    print("\n-- CG sensitivity --")
    dsm_per_mm = C.MASS_ITEMS["battery"][0] / 1000 / pr["m"] / MAC * 0.001
    tol_mm = 0.02 * MAC / (C.MASS_ITEMS["battery"][0] / 1000 / pr["m"]) * 1000
    print(f"   moving the battery 10 mm changes SM by {10*dsm_per_mm*100:.1f}% MAC -> a +/-2% SM tolerance = +/-{tol_mm:.0f} mm of pack position")
    for mah, mg in ((650, 45.0), (850, 56.0), (1000, C.MASS_ITEMS["battery"][0])):
        # keep position, change pack
        Lp = MM.lumps(pf, xb_c, batt_mass_g=mg); q = MM.props(Lp)
        smp = (xnp - q["cg"][0]) / MAC
        xr = MM.battery_x_for_cg(pf, xcg_t, batt_mass_g=mg)
        print(f"   {mah:4d} mAh pack ({mg:4.1f} g): SM {smp*100:5.1f}% if left at {xb_c*1000:.0f} mm | to hold SM {C.STATIC_MARGIN*100:.0f}% move it to {xr*1000:6.1f} mm")
    print("   battery x window for SM 6% .. 15%  (allowable range):")
    for sm_t in (0.06, 0.08, 0.10, 0.12, 0.15):
        xr = MM.battery_x_for_cg(pf, xnp - sm_t * MAC)
        print(f"      SM {sm_t*100:4.0f}% -> battery centre at {xr*1000:7.1f} mm {'(inside bay)' if lo<=xr<=hi else '(OUTSIDE bay)'}")
    # extra 10 g camera/GPS tweak
    Lx = MM.lumps(pf, xb_c, extra=[[0.010, -0.06, 0, 0]])
    qx = MM.props(Lx)
    print(f"   +10 g of ballast/gear at x=-60 mm (nose): CG moves {1000*(qx['cg'][0]-pr['cg'][0]):+.1f} mm -> SM {(xnp-qx['cg'][0])/MAC*100:.1f}%")
    print(f"   +10 g at the motor (x=+{C.MASS_ITEMS['motor_prop'][1]*pf.c_root*1000:.0f} mm): "
          f"CG moves {1000*(MM.props(MM.lumps(pf, xb_c, extra=[[0.010, C.MASS_ITEMS['motor_prop'][1]*pf.c_root, 0, 0]]))['cg'][0]-pr['cg'][0]):+.1f} mm")
    # ---- save
    out = dict(mass=pr["m"], cg=pr["cg"].tolist(), Ixx=pr["Ixx"], Iyy=pr["Iyy"], Izz=pr["Izz"], Ixz=pr["Ixz"],
               battery_x=xb_c, static_margin=sm, xnp_vlm=xnp)
    json.dump(out, open(os.path.join(C.RES, "mass_props.json"), "w"), indent=1)
    print("\nsaved results/mass_props.json")
    print("\nMass budget (g):")
    tot = 0
    for k, (mg, *_r) in C.MASS_ITEMS.items():
        print(f"   {k:14s}{mg:6.1f}"); tot += mg
    print(f"   servos x2     {2*C.SERVO_G:6.1f}\n   CF spars x2   {C.SPAR_G:6.1f}\n   PLA ribs+Monokote {C.STRUCT_AREAL_KG_M2*d['S']*1000:5.1f}   (areal {C.STRUCT_AREAL_KG_M2} kg/m^2 x S)")
    print(f"   TOTAL         {pr['m']*1000:6.1f}  (target 200-400 g)")


if __name__ == "__main__":
    main()
