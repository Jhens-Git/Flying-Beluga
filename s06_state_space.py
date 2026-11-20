#!/usr/bin/env python3
"""
s06_state_space.py -- AVL stability/control derivatives -> linear rigid-body state-space (A,B,C,D) for MATLAB/Simulink SIL.

Stability axes, level cruise (theta0 = 0, gamma = 0), U0 = V_CRUISE.
  Longitudinal  x = [u (m/s), w (m/s), q (rad/s), theta (rad)]   inputs [delta_e (rad), delta_T (N)]
                y = [u, alpha (rad), q, theta]
  Lateral       x = [v (m/s), p (rad/s), r (rad/s), phi (rad)]   input  [delta_a (rad)]
                y = [beta (rad), p, r, phi]
Elevon mixing (AVL sign: +deflection = trailing edge DOWN):  right = d_e + d_a ,  left = d_e - d_a
(AVL aileron d2 is defined with the +y surface going TE-down for positive d_a, so positive d_a rolls LEFT; Lda < 0 is expected.)
Not in AVL: alpha-dot derivatives (Cm_alphadot ~ 0 here -> short-period damping slightly pessimistic), and propwash/thrust-line effects.

Usage:  python3 s06_state_space.py             (uses results/avl/final_st.txt from s05)
        python3 s06_state_space.py --selftest  (uses a SYNTHETIC fixture: tests the code path only, numbers are NOT your aircraft)
"""
import os, sys, json
import numpy as np
import config as C
from s05_avl import parse

R2D = 180 / np.pi


def modes(A, label):
    ev = np.linalg.eigvals(A)
    print(f"\n{label} eigenvalues / modes")
    done = set()
    for i, l in enumerate(ev):
        if i in done: continue
        if abs(l.imag) > 1e-6:
            j = [k for k in range(len(ev)) if k != i and abs(ev[k] - l.conjugate()) < 1e-9]
            done |= set(j)
            wn = abs(l); z = -l.real / wn
            print(f"   {l.real:+8.3f} ±{abs(l.imag):6.3f}j   wn={wn:6.2f} rad/s  zeta={z:5.3f}  period={2*np.pi/abs(l.imag):6.2f} s")
        else:
            tc = -1 / l.real if l.real != 0 else np.inf
            kind = "STABLE  tau=%.2f s" % tc if l.real < 0 else "UNSTABLE t2=%.2f s" % (np.log(2) / l.real)
            print(f"   {l.real:+8.3f}            {kind}")
    return ev


def main():
    selftest = "--selftest" in sys.argv
    D = json.load(open(os.path.join(C.RES, "design.json")))
    mp = json.load(open(os.path.join(C.RES, "mass_props.json")))
    stfile = os.path.join(C.RES, "avl", "FIXTURE_st.txt" if selftest else "final_st.txt")
    d = parse(stfile)
    if "CLa" not in d:
        sys.exit(f"{stfile} not found / unparsable. Run s05_avl.py first (or --selftest).")
    scale = 1.0
    rp = os.path.join(C.RES, "avl_results.json")
    if not selftest and os.path.exists(rp):
        scale = json.load(open(rp)).get("ctrl_scale_to_per_rad", 1.0)
    U = C.V_CRUISE; q = 0.5 * C.RHO * U ** 2
    S, c, b, m, g = D["S"], D["MAC"], D["planform"]["span"], mp["mass"], C.G
    al = np.radians(D["alpha_trim_deg"])
    Ixx0, Iyy, Izz0, Ixz0 = mp["Ixx"], mp["Iyy"], mp["Izz"], mp["Ixz"]
    s2, c2, sa, ca = np.sin(2 * al), np.cos(2 * al), np.sin(al), np.cos(al)
    Ixx = Ixx0 * ca ** 2 + Izz0 * sa ** 2 - Ixz0 * s2
    Izz = Ixx0 * sa ** 2 + Izz0 * ca ** 2 + Ixz0 * s2
    Ixz = (Ixx0 - Izz0) * sa * ca + Ixz0 * c2
    CL0 = D["CL_cruise"]; CD0 = D["CD0"] + D["CDi"]
    CDa = 2 * CL0 * d["CLa"] / (np.pi * D["AR"] * D["e_span"])
    T = CD0 * q * S
    Tu = C.THRUST_U_FACTOR * T / U
    k = q * S / m
    # ------------------------------ longitudinal
    Xu = -2 * CD0 * k / U + Tu / m
    Xw = -(CDa - CL0) * k / U
    Zu = -2 * CL0 * k / U
    Zw = -(d["CLa"] + CD0) * k / U
    Zq = -d["CLq"] * k * c / (2 * U)
    Mw = d["Cma"] * q * S * c / (Iyy * U)
    Mq = d["Cmq"] * q * S * c ** 2 / (2 * Iyy * U)
    Zde = -d.get("CLd1", 0) * scale * k
    Mde = d["Cmd1"] * scale * q * S * c / Iyy
    A_lon = np.array([[Xu, Xw, 0, -g], [Zu, Zw, U + Zq, 0], [0, Mw, Mq, 0], [0, 0, 1, 0]])
    B_lon = np.array([[0, 1 / m], [Zde, 0], [Mde, 0], [0, 0]])
    C_lon = np.array([[1, 0, 0, 0], [0, 1 / U, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    D_lon = np.zeros((4, 2))
    # ------------------------------ lateral (moment form, handles Ixz)
    Yv = d["CYb"] * k / U; Yp = d["CYp"] * k * b / (2 * U); Yr = d["CYr"] * k * b / (2 * U)
    Lm = np.array([d["Clb"] * q * S * b / U, d["Clp"] * q * S * b ** 2 / (2 * U), d["Clr"] * q * S * b ** 2 / (2 * U)])
    Nm = np.array([d["Cnb"] * q * S * b / U, d["Cnp"] * q * S * b ** 2 / (2 * U), d["Cnr"] * q * S * b ** 2 / (2 * U)])
    Einv = np.linalg.inv(np.array([[Ixx, -Ixz], [-Ixz, Izz]]))
    pr = Einv @ np.vstack([Lm, Nm])                       # rows: pdot, rdot ; cols: v, p, r
    A_lat = np.array([[Yv, Yp, Yr - U, g], list(pr[0]) + [0], list(pr[1]) + [0], [0, 1, 0, 0]])
    Yda = d.get("CYd2", 0) * scale * k
    LNda = Einv @ np.array([d.get("Cld2", 0) * scale * q * S * b, d.get("Cnd2", 0) * scale * q * S * b])
    B_lat = np.array([[Yda], [LNda[0]], [LNda[1]], [0]])
    C_lat = np.array([[1 / U, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    D_lat = np.zeros((4, 1))
    # ------------------------------ report
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(f"{'SELFTEST (synthetic derivatives!)' if selftest else 'AVL derivatives'}   U0={U} m/s  q={q:.1f} Pa  m={m*1000:.0f} g  alpha_trim={D['alpha_trim_deg']:.2f} deg")
    print(f"stability-axis inertias: Ixx={Ixx:.5f} Iyy={Iyy:.5f} Izz={Izz:.5f} Ixz={Ixz:.6f}")
    print("\nA_lon =\n", A_lon, "\nB_lon [d_e, dT] =\n", B_lon)
    print("\nA_lat =\n", A_lat, "\nB_lat [d_a] =\n", B_lat)
    ev1 = modes(A_lon, "Longitudinal"); ev2 = modes(A_lat, "Lateral/directional")
    xnp = d.get("Xnp")
    if xnp: print(f"\nStatic margin from AVL: {(xnp-mp['cg'][0])/c*100:.1f}% MAC   Cm_alpha={d['Cma']:.3f}  Cn_beta={d['Cnb']:.3f}  Cl_beta={d['Clb']:.3f}")
    if d["Cma"] >= 0: print("!! Cm_alpha >= 0: statically UNSTABLE in pitch")
    # ------------------------------ export
    out = dict(A_lon=A_lon, B_lon=B_lon, C_lon=C_lon, D_lon=D_lon, A_lat=A_lat, B_lat=B_lat, C_lat=C_lat, D_lat=D_lat,
               U0=U, mass=m, Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixz=Ixz)
    tag = "state_space_SELFTEST" if selftest else "state_space"
    json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out.items()}, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
    try:
        from scipy.io import savemat
        savemat(os.path.join(C.RES, tag + ".mat"), out)
    except Exception as e:
        print("savemat skipped:", e)
    with open(os.path.join(C.RES, tag + ".m"), "w") as f:
        f.write("% generated by s06_state_space.py\n")
        for k_, v in out.items():
            if np.ndim(v):
                M = np.atleast_2d(np.array(v, float))
                f.write(f"{k_} = [" + "; ".join(" ".join(f"{x:.8g}" for x in row) for row in M) + "];\n")
            else:
                f.write(f"{k_} = {v};\n")
        f.write("sys_lon = ss(A_lon,B_lon,C_lon,D_lon,'StateName',{'u','w','q','theta'},'InputName',{'de','dT'},'OutputName',{'u','alpha','q','theta'});\n")
        f.write("sys_lat = ss(A_lat,B_lat,C_lat,D_lat,'StateName',{'v','p','r','phi'},'InputName',{'da'},'OutputName',{'beta','p','r','phi'});\n")
        f.write("damp(sys_lon); damp(sys_lat);\n")
    print(f"\nsaved results/{tag}.json / .mat / .m   (MATLAB: run('{tag}.m'))")


if __name__ == "__main__":
    main()
