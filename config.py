"""
config.py  --  single source of truth for the Flying Beluga design pipeline.
Edit numbers HERE (esp. the masses: replace every ESTIMATE with a kitchen-scale reading).
Coordinate system everywhere: x aft from the root leading edge / nose datum (m), y right (m), z up (m).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
AF_DIR = os.path.join(RES, "airfoils")
for d in (RES, AF_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- atmosphere / flight condition ----------------
RHO = 1.225          # kg/m^3 (sea level). West Lafayette is ~200 m -> 1.20; negligible
MU = 1.789e-5        # Pa s
G = 9.81
V_CRUISE = 12.0      # m/s
V_STALL_MAX = 8.0    # m/s  hand-launch/belly-land friendly (design requirement I chose)
N_LIMIT = 3.0        # limit load factor for structure (your VCRM says 2.5 -> we use 3.0, then x1.5 ultimate)
FOS_ULT = 1.5

# ---------------- design targets ----------------
SPAN_BOUNDS = (0.70, 0.80)    # m  (800 mm is the hard max: 2 x 400 mm rods)
STATIC_MARGIN = 0.10          # fraction of MAC, first-flight target (0.08-0.12)
CD_EXTRA = 0.010              # fins, skid, pod, gaps, interference drag on top of airfoil cd
CLMAX_3D_FACTOR = 0.75        # 3-D usable CLmax = factor * 2-D XFOIL clmax (swept, tailless, tip-stall limited)
RE_TIP_MIN = 60_000           # don't let tip chord drop into laminar-bubble hell
TWIST_BOUNDS = (-6.0, 0.0)    # deg, washout at tip (negative = tip LE down)
C_TIP_MIN = 0.075             # m   (printable rib + 4 mm spar + covering)
C_ROOT_MAX = 0.32             # m

# ---------------- carbon spar (4x4 mm pultruded) ----------------
ROD_B = 4.0e-3                # m
ROD_LEN = 0.420               # m
N_RODS = 2
E_CF = 120e9                  # Pa  conservative for pultruded square rod (spec sheets: 100-135 GPa)
G_CF = 4.5e9                  # Pa  shear modulus (UD CFRP is poor in torsion)
SIGMA_ALLOW = 600e6           # Pa  conservative flexural strength, pultruded rod (check vendor!)
TAU_ALLOW = 40e6              # Pa  interlaminar/torsional shear allowable, pultruded CF (conservative)
RHO_CF = 1600.0               # kg/m^3

# ---------------- nose bay / battery ----------------
BATT_LEN = 0.104              # m (4.1")
BATT_CENTER_MIN_X = -0.03     # m  front-most battery centre (nose pod may extend ahead of root LE)
BATT_CENTER_MAX_FRAC = 0.50   # of root chord

# ---------------- mass budget (grams) : ESTIMATE = replace with scale readings ----------------
# name: (mass_g, x as fraction of root chord (None = solved), y_m, z_m, note)
MASS_ITEMS = {
    "camera":        (10.0,   -0.25, 0.0,  0.0, "ESTIMATE nano FPV cam in nose pod"),
    "battery":       (63.5,  None,  0.0,  0.0, "BOM: 0.14 lb  -> position is SOLVED"),
    "stm32_board":   (21.0,   0.40, 0.0,  0.0, "ESTIMATE dev board"),
    "bmi270":        (2.0,    0.40, 0.0,  0.0, "ESTIMATE"),
    "gps_m100":      (5.0,   0.55, 0.0,  0.0, "ESTIMATE"),
    "elrs_rx":       (3.0,    0.35, 0.0,  0.0, "ESTIMATE"),
    "buck_5v":       (45.0,    0.45, 0.0,  0.0, "ESTIMATE"),
    "pdb_wiring":    (20.0,   0.50, 0.0,  0.0, "ESTIMATE PDB + wire harness"),
    "esc":           (21.0,   0.70, 0.0,  0.0, "ESTIMATE 15-20 A"),
    "motor_prop":    (25.0,   1.03, 0.0,  0.01,"ESTIMATE 1806 motor + 5in prop + mount, behind root TE"),
    "nose_pod_pla":  (20.0,  -0.10, 0.0,  0.0, "ESTIMATE printed nose pod + skid"),
    "fins":          (6.0,    None, None, 0.03,"ESTIMATE two printed fins, placed at wing tips"),
}
SERVO_G = 9.0                 # each, two of them; placed on y = +/-0.35 semispan, 70 % local chord
SPAR_G = 2 * 10.2             # 2 x (4x4x400 mm CF)
STRUCT_AREAL_KG_M2 = 0.40     # PLA ribs + Monokote per m^2 of planform (ESTIMATE; ~54 g on 0.135 m^2)

# ---------------- airfoil search ----------------
CRUISE_CL_GUESS = 0.35
RE_LIST = [50_000, 75_000, 100_000, 150_000, 200_000]
NCRIT_LIST = [5, 9]           # 5 = turbulent outdoor, 9 = clean air
UIUC_BASE = "https://m-selig.ae.illinois.edu/ads/coord/"
UIUC_NAMES = ["mh45", "mh49", "mh60", "mh61", "mh62", "mh64", "e186", "e184", "e387", "sd7037", "s5010", "ag35", "ag38"]

# ---------------- AVL / state-space ----------------
AVL_BIN = os.environ.get("AVL_BIN", "avl")     # path to the avl executable
FIN_HEIGHTS = [0.040, 0.060, 0.080, 0.100]      # m   candidate wingtip fin heights
ELEVON_CHORD_FRACS = [0.22, 0.28, 0.34]         # elevon chord / local chord
FIN_SWEEP_DEG = 25.0
FIN_TAPER = 0.6
CNB_MIN = 0.025                # per rad  weathercock stability target
DE_MAX_SLOW_DEG = 20.0         # max elevon needed to trim at 75 % of 3-D CLmax
DE_MAX_CRUISE_DEG = 3.0
THRUST_U_FACTOR = -1.0         # dT/dU = factor * T/U : -1 = constant power, 0 = constant thrust (fixed-throttle props are in between)

# ---------------- ribs (script 07) ----------------
RIB_PITCH = 0.030              # m  spacing (Monokote needs ~25-40 mm at these chords to hold shape)
RIB_T_OPTIONS = [0.8e-3, 1.0e-3, 1.2e-3, 1.6e-3]   # m  = 2,2.5,3,4 perimeters of a 0.4 mm nozzle
RHO_PLA = 1240.0
LIGHTENING_REMOVED = 0.55      # fraction of rib web area cut out
SKIN_WALL_MIN = 0.9e-3         # m  printed skin left above/below a spar hole (>= 2 perimeters)
SIGMA_BEARING_PLA = 15e6       # Pa conservative (layer-direction weak, glue/creep, warm day)
REAR_SPAR_SPACING_RANGE = (0.03, 0.12)   # m  chordwise distance front->rear rod to search
REAR_SPAR_MAX_XC = 0.65        # elevon hinge sits at 1 - elevon chord fraction (~72 %)
