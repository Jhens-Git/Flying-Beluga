# Flying Beluga Design Pipeline

Automated aerodynamic, structural, and flight dynamics optimization pipeline for the Flying Beluga RC aircraft.

## Usage
1. Edit masses and design targets in `config.py`.
2. Run the full pipeline:
   ```bash
   ./run_all.sh
   ```

## Pipeline Order
Execution order: `s01` -> `s03` -> `s04` -> `s02` -> `s05` -> `s06`

| Step | Module | Output |
| :--- | :--- | :--- |
| **s01** | XFOIL | `results/airfoil_summary.json`, `results/airfoil_report.png` |
| **s03** | Optimizer | `results/design.json`, `results/planform.png` |
| **s04** | Mass / CG | `results/mass_props.json` |
| **s02** | Structure | `results/structure.json` |
| **s05** | AVL | `results/avl/beluga.avl`, `.mass`, `avl_results.json` |
| **s06** | State Space | `results/state_space.mat`, `.m`, `.json` |
