"""massmodel.py -- point-mass model of the Beluga: CG, inertias, and 'where must the battery go'."""
import numpy as np
import config as C
from geometry import Planform


def lumps(pf: Planform, batt_x, batt_mass_g=None, extra=None):
    """Return array rows [m(kg), x, y, z]. batt_x in metres from root-LE datum."""
    L = []
    s, cr = pf.s, pf.c_root
    for name, (mg, xf, y, z, _) in C.MASS_ITEMS.items():
        m = mg / 1000
        if name == "battery":
            m = (batt_mass_g if batt_mass_g is not None else mg) / 1000
            L.append([m, batt_x, 0.0, 0.0])
        elif name == "fins":
            xt = pf.xle(s) + 0.55 * pf.c_tip
            L += [[m / 2, xt, +s, z], [m / 2, xt, -s, z]]
        else:
            L.append([m, xf * cr, y, z])
    for sgn in (+1, -1):                                  # servos
        yy = sgn * 0.35 * s
        L.append([C.SERVO_G / 1000, pf.xle(yy) + 0.70 * pf.chord(yy), yy, 0.0])
    ys = np.linspace(0, s, 41); ym = 0.5 * (ys[1:] + ys[:-1]); dy = np.diff(ys)
    S = pf.props()["S"]
    w = pf.chord(ym) * dy; w = w / w.sum()                # structure follows chord*dy (area)
    for sgn in (+1, -1):
        for yy, wi in zip(ym, w):
            L.append([0.5 * C.STRUCT_AREAL_KG_M2 * S * wi, pf.xle(yy) + 0.45 * pf.chord(yy), sgn * yy, 0.0])
            L.append([0.5 * C.SPAR_G / 1000 * wi, pf.xle(yy) + 0.30 * pf.chord(yy), sgn * yy, 0.0])
    if extra:
        L += extra
    return np.array(L)


def props(L):
    m = L[:, 0].sum()
    cg = (L[:, 0:1] * L[:, 1:4]).sum(0) / m
    d = L[:, 1:4] - cg
    x, y, z = d.T
    mm = L[:, 0]
    Ixx = np.sum(mm * (y ** 2 + z ** 2)); Iyy = np.sum(mm * (x ** 2 + z ** 2)); Izz = np.sum(mm * (x ** 2 + y ** 2))
    Ixz = -np.sum(mm * x * z)
    return dict(m=m, cg=cg, Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixz=Ixz)


def battery_x_for_cg(pf: Planform, xcg_target, batt_mass_g=None):
    """Closed-form battery x that puts CG at xcg_target (CG is linear in battery position)."""
    L0 = lumps(pf, 0.0, batt_mass_g)
    mb = (batt_mass_g if batt_mass_g is not None else C.MASS_ITEMS["battery"][0]) / 1000
    M = L0[:, 0].sum()
    cg0 = (L0[:, 0] * L0[:, 1]).sum() / M          # with battery at x=0
    # xcg = cg0 + mb*xb/M
    return (xcg_target - cg0) * M / mb


def batt_limits(pf):
    return C.BATT_CENTER_MIN_X, C.BATT_CENTER_MAX_FRAC * pf.c_root


def total_mass(pf):
    return lumps(pf, 0.0)[:, 0].sum()
