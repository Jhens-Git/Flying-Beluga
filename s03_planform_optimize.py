#!/usr/bin/env python3
"""
s03_planform_optimize.py -- size the Beluga: span, chords, kink, sweeps, washout, CG, fins, elevons.

For every candidate airfoil from script 01, a global optimizer (differential evolution) searches the planform that
minimises cruise drag subject to *all* of:
   * static margin = target (CG is placed at  x_NP - SM*MAC)
   * elevons neutral at cruise  -> the required tip washout is SOLVED (trim: CL = W/qS and Cm_cg = 0)
   * stall speed <= V_STALL_MAX using 3-D CLmax = factor * XFOIL clmax
   * washout within printable/sane limits
   * the battery can physically reach the required CG inside the nose bay (mass model, massmodel.py)
Output: results/design.json  (+ results/planform.png)
"""
import os, json, sys
import numpy as np
from scipy.optimize import differential_evolution
import config as C
import massmodel as MM
from geometry import Planform, VLM

# x = [span, c_root, c_tip, f_kink, yk_frac, sweep_in, sweep_out]
BOUNDS = [C.SPAN_BOUNDS, (0.18, C.C_ROOT_MAX), (C.C_TIP_MIN, 0.13), (0.2, 0.9), (0.20, 0.50), (20, 65), (10, 40)]
DEFAULT_AF = dict(file=None, cm0=0.0, alpha0L_deg=-0.1, clmax=0.86, cd_cruise=0.0125, cd_cruise_75k=0.0135)


def make_pf(x, tw=0.0):
    span, cr, ct, fk, yk, si, so = x
    return Planform(span, cr, ct + fk * (cr - ct), ct, yk, si, so, tw)


def evaluate(x, af, n_half=12, detail=False):
    pf = make_pf(x)
    pr = pf.props(401)
    S, MAC = pr["S"], pr["MAC"]
    L0 = MM.lumps(pf, 0.0)
    m = L0[:, 0].sum(); W = m * C.G
    q = 0.5 * C.RHO * C.V_CRUISE ** 2
    CL = W / (q * S)
    v = VLM(pf, n_half, af["alpha0L_deg"], af["cm0"])
    xnp = v.derivs(0.0)["xnp"]
    xcg = xnp - C.STATIC_MARGIN * MAC
    alpha, tw, d = v.solve_trim(xcg, CL)
    tw_deg = np.degrees(tw)
    G = v.gamma(alpha, tw)
    CDi, e = v.induced_drag_coeff(G)
    Re_mac = C.RHO * C.V_CRUISE * MAC / C.MU
    cd0 = af["cd_cruise"] * (Re_mac / 1e5) ** -0.3 + C.CD_EXTRA
    D = q * S * (cd0 + CDi)
    CLmax3d = C.CLMAX_3D_FACTOR * af["clmax"]
    Vs = np.sqrt(2 * W / (C.RHO * S * CLmax3d))
    xb = MM.battery_x_for_cg(pf, xcg)
    xb_lo, xb_hi = MM.batt_limits(pf)
    pen = 0.0
    pen += 0.05 * max(0.0, tw_deg - TW_HI) + 0.05 * max(0.0, TW_LO - tw_deg)
    pen += 0.4 * max(0.0, Vs - C.V_STALL_MAX)
    pen += 8.0 * (max(0.0, xb_lo - xb) + max(0.0, xb - xb_hi))
    if detail:
        return dict(pf=pf, props=pr, W=W, m=m, CL=CL, alpha_deg=np.degrees(alpha), twist_deg=tw_deg, xnp=xnp, xcg=xcg,
                    CDi=CDi, e=e, cd0=cd0, D=D, LD=W / D, Vs=Vs, xb=xb, xb_lim=(xb_lo, xb_hi), CLa=d["CLa"], Cma=d["Cma"],
                    CLmax3d=CLmax3d, Re_mac=Re_mac, Re_tip=C.RHO * C.V_CRUISE * pf.c_tip / C.MU, pen=pen, v=v, G=G)
    return D + pen


TW_LO, TW_HI = C.TWIST_BOUNDS


def optimise(af, seed=1, maxiter=35):
    r = differential_evolution(evaluate, BOUNDS, args=(af,), seed=seed, maxiter=maxiter, popsize=8, tol=1e-6,
                               mutation=(0.5, 1.0), recombination=0.8, polish=False, updating="immediate")
    return r.x, r.fun


def load_airfoils():
    p = os.path.join(C.RES, "airfoil_summary.json")
    if not os.path.exists(p):
        print("!! results/airfoil_summary.json missing -> using placeholder airfoil numbers. Run s01 first.")
        return {"placeholder": DEFAULT_AF}
    return json.load(open(p))["candidates"]


def sweep_study(x, af):
    print("\n-- sensitivity around the optimum (all else re-trimmed to SM = %.0f%%) --" % (100 * C.STATIC_MARGIN))
    print("   outer LE sweep [deg] | washout needed [deg] | drag [N] | L/D | battery x [mm]")
    for so in (10, 15, 20, 25, 30, 35, 40):
        xx = np.array(x, float); xx[6] = so
        r = evaluate(xx, af, detail=True)
        print(f"   {so:10.0f}           {r['twist_deg']:+8.2f}            {r['D']:.3f}    {r['LD']:5.1f}   {1000*r['xb']:7.1f}")


def main():
    cands = load_airfoils()
    best = None
    print("Optimising planform for each candidate airfoil (~15-30 s each) ...", flush=True)
    for name, af in cands.items():
        x, f = optimise(af)
        r = evaluate(x, af, detail=True)
        feas = r["pen"] < 1e-3
        print(f"  {name:18s} D={r['D']:.3f} N  L/D={r['LD']:5.1f}  washout={r['twist_deg']:+.2f} deg  Vs={r['Vs']:.1f}  "
              f"span={x[0]*1000:.0f}  {'OK' if feas else 'PENALISED (%.2f)' % r['pen']}", flush=True)
        if best is None or f < best[0]:
            best = (f, name, af, x)
    f, name, af, x = best
    r = evaluate(x, af, detail=True)
    pf, pr = r["pf"], r["props"]
    pf.twist_tip = r["twist_deg"]
    y, c, xle, tw = pf.stations()
    # ---- fin + elevon starting sizes
    l_f = (pf.xle(pf.s) + 0.55 * pf.c_tip) - r["xcg"]
    Vv = 0.020
    S_fin_tot = Vv * pr["S"] * pf.span / l_f
    fin_taper = 0.6
    h_fin = (S_fin_tot / 2) / (pf.c_tip * (1 + fin_taper) / 2)
    design = dict(
        airfoil=name, airfoil_file=af.get("file"), cm0=af["cm0"], alpha0L_deg=af["alpha0L_deg"],
        planform=dict(span=pf.span, c_root=pf.c_root, c_kink=pf.c_kink, c_tip=pf.c_tip, yk_frac=pf.yk_frac,
                      sweep_in=pf.sweep_in, sweep_out=pf.sweep_out, twist_tip=pf.twist_tip),
        stations=dict(y=y.tolist(), chord=c.tolist(), xle=xle.tolist(), twist=tw.tolist()),
        S=pr["S"], AR=pr["AR"], MAC=pr["MAC"], xle_mac=pr["xle_mac"], y_mac=pr["y_mac"],
        xnp=r["xnp"], xcg=r["xcg"], static_margin=C.STATIC_MARGIN, mass=r["m"], weight_N=r["W"],
        wing_loading_g_dm2=r["m"] * 1000 / (pr["S"] * 100),
        CL_cruise=r["CL"], alpha_trim_deg=r["alpha_deg"], washout_deg=r["twist_deg"], e_span=r["e"], CDi=r["CDi"], CD0=r["cd0"],
        LD_cruise=r["LD"], drag_N=r["D"], V_stall=r["Vs"], CLmax_3d=r["CLmax3d"], Re_mac=r["Re_mac"], Re_tip=r["Re_tip"],
        CLa_per_rad=r["CLa"], battery_x_required=r["xb"], battery_x_limits=list(r["xb_lim"]),
        fin=dict(height=h_fin, root_chord=pf.c_tip, taper=fin_taper, Vv=Vv, lever=l_f),
        elevon=dict(chord_frac=0.28, y_in_frac=0.35, y_out_frac=0.97),
        penalty=r["pen"])
    json.dump(design, open(os.path.join(C.RES, "design.json"), "w"), indent=1)
    print("\n================  OPTIMISED FLYING BELUGA  ================")
    print(f" airfoil            : {name}   (Cm0 {af['cm0']:+.4f}, alpha_L0 {af['alpha0L_deg']:+.2f} deg)")
    print(f" span               : {pf.span*1000:.0f} mm       area {pr['S']*1e4:.0f} cm^2     AR {pr['AR']:.2f}")
    print(f" chords root/kink/tip: {pf.c_root*1000:.0f} / {pf.c_kink*1000:.0f} / {pf.c_tip*1000:.0f} mm   (kink at y = {pf.yk_frac*pf.s*1000:.0f} mm)")
    print(f" LE sweep in / out  : {pf.sweep_in:.1f} / {pf.sweep_out:.1f} deg     MAC {pr['MAC']*1000:.0f} mm   (y_MAC {pr['y_mac']*1000:.0f} mm, LE at x={pr['xle_mac']*1000:.0f} mm)")
    print(f" washout (tip)      : {r['twist_deg']:+.2f} deg   trim alpha {r['alpha_deg']:+.2f} deg at CL {r['CL']:.2f}, elevons neutral")
    print(f" neutral point      : x = {r['xnp']*1000:.0f} mm  ({(r['xnp']-pr['xle_mac'])/pr['MAC']*100:.0f}% MAC)")
    print(f" CG target (SM {C.STATIC_MARGIN*100:.0f}%) : x = {r['xcg']*1000:.0f} mm  ({(r['xcg']-pr['xle_mac'])/pr['MAC']*100:.0f}% MAC)")
    print(f" battery centre must be at x = {r['xb']*1000:.0f} mm  (allowed {r['xb_lim'][0]*1000:.0f}..{r['xb_lim'][1]*1000:.0f} mm)")
    print(f" mass {r['m']*1000:.0f} g, wing loading {design['wing_loading_g_dm2']:.1f} g/dm^2, V_stall {r['Vs']:.1f} m/s, L/D {r['LD']:.1f}, e={r['e']:.2f}")
    print(f" Re_MAC {r['Re_mac']:.0f}, Re_tip {r['Re_tip']:.0f}")
    print(f" start fin height   : {h_fin*1000:.0f} mm (root chord {pf.c_tip*1000:.0f}, taper {fin_taper}), Vv={Vv}")
    print(f" start elevon       : 28% chord, y = 35%..97% semispan")
    print(" (final fin/elevon sizes are chosen in script 05 with AVL)")
    sweep_study(x, af)
    try:
        plot(design, r)
    except Exception as e:
        print("plot skipped", e)


def plot(d, r):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pf = r["pf"]; pf.twist_tip = d["washout_deg"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    y, c, xle, _ = pf.stations()
    yy = np.concatenate([-y[::-1], y[1:]]); le = np.concatenate([xle[::-1], xle[1:]]); cc = np.concatenate([c[::-1], c[1:]])
    ax[0].fill_between(yy * 1000, -(le) * 1000, -(le + cc) * 1000, alpha=.35)
    ax[0].plot(yy * 1000, -le * 1000, "k"); ax[0].plot(yy * 1000, -(le + cc) * 1000, "k")
    ax[0].plot([-pf.s * 1e3, pf.s * 1e3], [-d["xnp"] * 1e3] * 2, "r--", label="neutral point")
    ax[0].plot(0, -d["xcg"] * 1e3, "go", ms=10, label="CG (SM %.0f%%)" % (100 * d["static_margin"]))
    ax[0].plot(0, -d["battery_x_required"] * 1e3, "bs", label="battery centre")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_xlabel("y [mm]"); ax[0].set_ylabel("-x [mm] (nose up)"); ax[0].set_title("Planform")
    ym, sl = r["v"].spanload(r["G"])
    ax[1].plot(ym * 1000, sl, label="cl c / (CL MAC)"); ax[1].plot(ym * 1000, r["v"].cm / d["MAC"], "--", label="chord / MAC")
    ax[1].legend(); ax[1].set_xlabel("y [mm]"); ax[1].set_title("Trimmed span load (VLM)"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.RES, "planform.png"), dpi=120)


if __name__ == "__main__":
    main()
