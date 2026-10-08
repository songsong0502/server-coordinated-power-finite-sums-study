#!/bin/bash
# Runs every experiment and builds the figures (about twenty minutes on two cores).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$(dirname "$0")"
mkdir -p ../results ../figures
for s in exp_e1_validation exp_e1_dist exp_e2_stepsize_rules exp_e2b_kink_rank exp_e3_local_steps exp_e4_sampling exp_aux; do
  echo "== $s $(date)"; python3 $s.py > ../results/$s.log 2>&1; echo "done $s $(date)"
done
python3 analyze.py > ../results/analyze.log 2>&1
