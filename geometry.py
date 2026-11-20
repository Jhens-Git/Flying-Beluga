"""
geometry.py  --  BWB planform maths + a compact vortex-lattice solver (screening tool).

The VLM here is a 1-chordwise-panel horseshoe lattice (Weissinger-type). It is good for
neutral point, CL_alpha, twist trim and span efficiency of a swept planform -- exactly what the
optimizer needs thousands of times. AVL (script 05) is the authority for the final numbers,
control derivatives and lateral/directional stuff.
"""
import numpy as np
from dataclasses import dataclass, asdict

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class Planform:
    span: float = 0.80
    c_root: float = 0.26
    c_kink: float = 0.20
    c_tip: float = 0.09
    yk_frac: float = 0.25      # kink station as fraction of semispan
    sweep_in: float = 45.0     # deg, LE sweep, blended body panel
    sweep_out: float = 25.0    # deg, LE sweep, outer wing panel
    twist_tip: float = -2.0    # deg, linear from kink to tip (0 inboard)

    @property
    def s(self):
        return 0.5 * self.span

    def stations(self):
        s = self.s
        yk = self.yk_frac * s
        y = np.array([0.0, yk, s])
        c = np.array([self.c_root, self.c_kink, self.c_tip])
        x1 = yk * np.tan(np.radians(self.sweep_in))
        x2 = x1 + (s - yk) * np.tan(np.radians(self.sweep_out))
        return y, c, np.array([0.0, x1, x2]), np.array([0.0, 0.0, self.twist_tip])

    def chord(self, y):
        Y, c, _, _ = self.stations()
        return np.interp(np.abs(y), Y, c)

    def xle(self, y):
        Y, _, x, _ = self.stations()
        return np.interp(np.abs(y), Y, x)

    def twist_shape(self, y):
        """unit-twist distribution (0 inboard of kink, 1 at tip)"""
        Y, _, _, _ = self.stations()
        return np.interp(np.abs(y), [0, Y[1], Y[2]], [0, 0, 1])

    def props(self, n=2001):
        y = np.linspace(0, self.s, n)
        c, x = self.chord(y), self.xle(y)
        S = 2 * _trapz(c, y)
        mac = 2 / S * _trapz(c ** 2, y)
        xle_mac = 2 / S * _trapz(x * c, y)
        y_mac = 2 / S * _trapz(y * c, y)
        return dict(S=S, AR=self.span ** 2 / S, MAC=mac, xle_mac=xle_mac, y_mac=y_mac,
                    taper=self.c_tip / self.c_root, x_te_tip=x[-1] + c[-1])


def _seg(P, A, B):
    """Biot-Savart for finite segments A->B (unit circulation). P:(Ni,1,3) A,B:(1,Nj,3) -> (Ni,Nj,3)"""
    r1, r2, r0 = P - A, P - B, B - A
    cr = np.cross(r1, r2)
    n2 = np.sum(cr ** 2, axis=-1)
    d = np.sum(r0 * (r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
                     - r2 / np.linalg.norm(r2, axis=-1, keepdims=True)), axis=-1)
    with np.errstate(all="ignore"):
        f = np.where(n2 > 1e-14, d / (4 * np.pi * n2), 0.0)
    return cr * f[..., None]


class VLM:
    """Planar horseshoe lattice. alpha0L_deg: zero-lift angle of the airfoil (camber) -> effective incidence."""

    def __init__(self, pf: Planform, n_half=20, alpha0L_deg=0.0, cm0_af=0.0):
        self.pf = pf
        N = 2 * n_half
        s = pf.s
        ye = -s * np.cos(np.linspace(0, np.pi, N + 1))
        self.ye = ye
        self.dy = np.diff(ye)
        self.ym = 0.5 * (ye[:-1] + ye[1:])
        ce, xe = pf.chord(ye), pf.xle(ye)
        xa = xe + 0.25 * ce                     # bound vortex x at panel edges
        self.xb = 0.5 * (xa[:-1] + xa[1:])      # lift acts here (segment midpoint)
        cm_, xm_ = 0.5 * (ce[:-1] + ce[1:]), 0.5 * (xe[:-1] + xe[1:])
        self.cm = cm_
        X = 60 * s
        A = np.stack([xa[:-1], ye[:-1], 0 * ye[:-1]], -1)[None]
        B = np.stack([xa[1:], ye[1:], 0 * ye[1:]], -1)[None]
        A_inf = np.stack([X + 0 * ye[:-1], ye[:-1], 0 * ye[:-1]], -1)[None]
        B_inf = np.stack([X + 0 * ye[1:], ye[1:], 0 * ye[1:]], -1)[None]
        P = np.stack([xm_ + 0.75 * cm_, self.ym, 0 * self.ym], -1)[:, None]
        v = _seg(P, A_inf, A) + _seg(P, A, B) + _seg(P, B, B_inf)
        self.Aw = v[..., 2]
        self.th_twist = pf.twist_shape(self.ym)                        # unit shape: result is per rad of tip twist
        self.th_camber = -np.radians(alpha0L_deg) * np.ones(N)
        one = np.ones(N)
        self.Ga = np.linalg.solve(self.Aw, -one)                       # per rad alpha
        self.Gc = np.linalg.solve(self.Aw, -self.th_camber)            # camber offset
        self.Gt = np.linalg.solve(self.Aw, -self.th_twist)             # per rad of tip twist
        pr = pf.props()
        self.S, self.MAC, self.pr = pr["S"], pr["MAC"], pr
        self.cm0_af = cm0_af
        # airfoil pitching moment contribution about the section quarter-chord, referenced to MAC
        self.cm_af = cm0_af * np.sum(self.cm ** 2 * self.dy) / (self.S * self.MAC)

    # ---- helpers (V = 1 normalisation) ----
    def CL_of(self, G):
        return 2 * np.sum(G * self.dy) / self.S

    def Cm_of(self, G, xref):
        return -2 * np.sum(G * self.dy * (self.xb - xref)) / (self.S * self.MAC)

    def derivs(self, xref):
        CLa, CL0c, CLt = self.CL_of(self.Ga), self.CL_of(self.Gc), self.CL_of(self.Gt)
        Cma, Cm0c, Cmt = (self.Cm_of(g, xref) for g in (self.Ga, self.Gc, self.Gt))
        return dict(CLa=CLa, CL0=CL0c, CLt=CLt, Cma=Cma, Cm0=Cm0c + self.cm_af, Cmt=Cmt,
                    xnp=xref - Cma * self.MAC / CLa)

    def solve_trim(self, xcg, CL_target):
        """Find alpha [rad] and tip twist [rad] giving CL = CL_target and Cm_cg = 0 (elevons neutral)."""
        d = self.derivs(xcg)
        M = np.array([[d["CLa"], d["CLt"]], [d["Cma"], d["Cmt"]]])
        rhs = np.array([CL_target - d["CL0"], -d["Cm0"]])
        alpha, tw = np.linalg.solve(M, rhs)
        return alpha, tw, d

    def gamma(self, alpha, tw):
        return alpha * self.Ga + self.Gc + tw * self.Gt

    def induced_drag_coeff(self, G):
        """Trefftz-plane induced drag; returns CDi and span-efficiency e."""
        g = np.concatenate([[0], G]) - np.concatenate([G, [0]])         # trailing-vortex strength at each edge
        w = np.array([np.sum(g / (2 * np.pi * (y - self.ye + 1e-12 * (np.abs(y - self.ye) < 1e-12))))
                      for y in self.ym])
        Di = -0.5 * np.sum(G * w * self.dy)
        CDi = Di / (0.5 * self.S)
        CL = self.CL_of(G)
        return CDi, CL ** 2 / (np.pi * self.pf.span ** 2 / self.S * CDi + 1e-12)

    def spanload(self, G):
        """local cl*c/(CL*MAC) vs y"""
        cl_local = 2 * G / self.cm
        return self.ym, cl_local / self.CL_of(G)


if __name__ == "__main__":
    # --- self-test: rectangular AR=8 wing, unswept -> lifting-line says CLa=4.9, e~0.97, NP=25 %c
    pf = Planform(span=1.6, c_root=0.2, c_kink=0.2, c_tip=0.2, sweep_in=0, sweep_out=0, twist_tip=0)
    v = VLM(pf, 30)
    d = v.derivs(0.1)
    a = np.radians(5)
    CDi, e = v.induced_drag_coeff(v.gamma(a, 0))
    print(f"CLa={d['CLa']:.3f} (LL theory ~4.91)  NP/c={d['xnp']/0.2:.3f} (0.25)  e={e:.3f}")
