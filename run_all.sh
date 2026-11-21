#!/usr/bin/env bash
# Full pipeline. Needs: xfoil xvfb xfonts-base xfonts-75dpi (apt), numpy scipy matplotlib (pip), avl binary (for 05/06).
set -e
python3 s01_airfoils.py            # XFOIL sweep          -> results/airfoil_summary.json
python3 s03_planform_optimize.py   # size the aircraft    -> results/design.json
python3 s04_mass_cg.py             # CG / inertias        -> results/mass_props.json
python3 s02_structure.py           # spar sizing (needs mass_props for battery clearance)
python3 s05_avl.py                 # AVL trim + fin/elevon sweep  (use --dry to just write the .avl)
python3 s06_state_space.py         # A,B,C,D -> results/state_space.{json,mat,m}
python3 s07_ribs.py                # rib spacing / thickness / spar-hole sizing
python3 s08_planform_drawing.py    # Plot all outputs 