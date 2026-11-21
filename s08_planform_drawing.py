#!/usr/bin/env python3
"""s08_planform_drawing.py -- scale top-view drawing of the Beluga from results/*.json -> results/planform_drawing.png / .svg"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import config as C
from geometry import Planform

R = lambda f: json.load(open(os.path.join(C.RES, f)))
D, S2, MP = R("design.json"), R("structure.json"), R("mass_props.json")
RB = R("ribs.json") if os.path.exists(os.path.join(C.RES, "ribs.json")) else dict(rear_offset_m=0.03)
p = D["planform"]
P = Planform(**{k: p[k] for k in ("span", "c_root", "c_kink", "c_tip", "yk_frac", "sweep_in", "sweep_out")}, twist_tip=0.0)
s, mm = P.s, 1000.0
Y, Cc, X, _ = P.stations()
fin_h = D.get("fin_final", D["fin"]["height"]); el = D.get("elevon_final", D["elevon"]["chord_frac"])
tw = D.get("washout_deg_avl", D["washout_deg"])

fig, ax = plt.subplots(figsize=(13, 8.5))
# outline (plot coords: horizontal = y, vertical = -x so the nose points up)
yy = np.linspace(-s, s, 400)
le, te = P.xle(yy), P.xle(yy) + P.chord(yy)
ax.add_patch(Polygon(np.c_[np.r_[yy, yy[::-1]] * mm, -np.r_[le, te[::-1]] * mm], fc="#dbe9f6", ec="k", lw=1.8, zorder=1))
# elevons
for sg in (1, -1):
    y0, y1 = 0.35 * s, s
    ys = np.linspace(y0, y1, 30)
    xl = P.xle(ys) + (1 - el) * P.chord(ys); xt = P.xle(ys) + P.chord(ys)
    ax.add_patch(Polygon(np.c_[np.r_[ys, ys[::-1]] * sg * mm, -np.r_[xl, xt[::-1]] * mm], fc="#f4c27a", ec="#a05a00", lw=1.2, alpha=.9, zorder=2))
ax.text(0.68 * s * mm, -(P.xle(0.68 * s) + 0.9 * P.chord(0.68 * s)) * mm - 14, f"elevon ({el*100:.0f}% c)", ha="center", fontsize=9, color="#a05a00")
# ribs
ribs = np.arange(C.RIB_PITCH / 2, s, C.RIB_PITCH)
for sg in (1, -1):
    for y in ribs:
        ax.plot([sg * y * mm] * 2, [-P.xle(y) * mm, -(P.xle(y) + P.chord(y)) * mm], c="#6b8db5", lw=.7, zorder=3)
# spars (rods extend 45 mm past the centreline)
lam, x0, reach, d = S2["spar_sweep_deg"], S2["spar_x0"], S2["spar_reach"], RB["rear_offset_m"]
for off, col, lab in ((0, "#222", "front rod"), (d, "#7a1f1f", "rear rod")):
    for sg in (1, -1):
        y_a, y_b = 0.0, reach
        yv = np.array([y_a, y_b])
        xv = x0 + off + np.tan(np.radians(lam)) * yv
        ax.plot(sg * yv * mm, -xv * mm, c=col, lw=3, solid_capstyle="butt", zorder=5, label=lab if sg == 1 else None)
# fins (top view: a thin strip at each tip, root chord long)
xt0, ct = P.xle(s), P.chord(s)
for sg in (1, -1):
    ax.add_patch(Rectangle((sg * s * mm - 2, -(xt0 + ct) * mm), 4, ct * mm, fc="#2a7a3b", ec="k", zorder=6))
ax.text(s * mm + 6, -(xt0 + 0.5 * ct) * mm, f"fin\n{fin_h*mm:.0f} mm tall", fontsize=8, color="#2a7a3b", va="center")
# battery, motor/prop, nose pod
bx = MP["battery_x"]
ax.add_patch(Rectangle((-10, -(bx + C.BATT_LEN / 2) * mm), 20, C.BATT_LEN * mm, fc="#c44", ec="k", alpha=.85, zorder=6, label="3S battery"))
mx = C.MASS_ITEMS["motor_prop"][1] * P.c_root
ax.plot([-63.5, 63.5], [-mx * mm] * 2, c="gray", lw=2, ls=":", zorder=6, label='5" prop disk')
ax.add_patch(Rectangle((-8, -(mx + 0.01) * mm), 16, 22, fc="#555", zorder=6))
# MAC, NP, CG
ymac, xle_m, mac = D["y_mac"], D["xle_mac"], D["MAC"]
ax.plot([ymac * mm] * 2, [-xle_m * mm, -(xle_m + mac) * mm], c="m", lw=4, alpha=.6, zorder=4, label=f"MAC {mac*mm:.0f} mm")
ax.plot([-ymac * mm] * 2, [-xle_m * mm, -(xle_m + mac) * mm], c="m", lw=4, alpha=.6, zorder=4)
ax.axhline(-D["xnp"] * mm, c="r", ls="--", lw=1, zorder=4); ax.text(-s * mm, -D["xnp"] * mm + 3, f"neutral point  x={D['xnp']*mm:.0f} mm", color="r", fontsize=9)
cgx = MP["cg"][0]
ax.plot(0, -cgx * mm, "o", ms=13, mfc="w", mec="k", mew=2, zorder=9); ax.plot(0, -cgx * mm, "+", ms=13, c="k", zorder=10)
ax.text(22, -cgx * mm - 12, f"CG  x={cgx*mm:.0f} mm\nstatic margin {MP['static_margin']*100:.1f}% MAC", fontsize=9, zorder=10)
# dimensions
top = -(-20)
ax.annotate("", (-s * mm, 25), (s * mm, 25), arrowprops=dict(arrowstyle="<->")); ax.text(0, 31, f"span {p['span']*mm:.0f} mm", ha="center", fontsize=10)
ax.text(0, -P.xle(0) * mm + 8, "nose", ha="center", fontsize=9)
ax.text(0, -(p["c_root"]) * mm - 40, f"root chord {p['c_root']*mm:.0f} mm", ha="center", fontsize=8)
ax.annotate("rods end in an angled,\nbolted root block", (0, -x0*mm), (-250, -20), fontsize=8, arrowprops=dict(arrowstyle="->"))
for yv, c_, lab in ((Y[1], p["c_kink"], "kink"), (s, p["c_tip"], "tip")):
    ax.text(-yv * mm - 8, -(P.xle(yv) + c_ / 2) * mm, f"{lab}\n{c_*mm:.0f}", ha="right", fontsize=8, color="#333")
ax.set_title(f"Flying Beluga  -  planform (top view)   S = {D['S']*1e4:.0f} cm$^2$,  AR = {D['AR']:.2f},  LE sweep {p['sweep_in']:.0f}°/{p['sweep_out']:.0f}°,  "
             f"washout {tw:+.1f}°,  {D['airfoil']}", fontsize=11)
ax.set_aspect("equal"); ax.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
ax.set_xlabel("y  [mm]"); ax.set_ylabel("-x  [mm]   (nose up)"); ax.grid(alpha=.2)
ax.set_xlim(-s * mm - 40, s * mm + 60); ax.set_ylim(-(P.xle(s) + ct) * mm - 40, 50)
for ext in ("png", "svg"):
    plt.savefig(os.path.join(C.RES, f"planform_drawing.{ext}"), dpi=140, bbox_inches="tight")
print("saved results/planform_drawing.png and .svg")
