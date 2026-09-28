#!/usr/bin/env bash
# Design A end-to-end: sweep then plot. Run from the repo root.
set -euo pipefail
python -m curvlearn.sweep --config experiments/designA.yaml --out results/designA_results.json "$@"
python -m curvlearn.plot  --results results/designA_results.json --outdir results
echo "Design A done -> results/designA_phase_diagram.png, results/designA_kappa_trajectories.png"
