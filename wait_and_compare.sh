#!/usr/bin/env bash
# Wait until the direct per-view run has merged, then run the full MLP-vs-direct
# comparison (summary + plots) on GPU 0.
set -u
cd /home/mirlab/Desktop/Digitally-Twinned-Neural-Geometry-Calibration
source /home/mirlab/anaconda3/etc/profile.d/conda.sh
conda activate dudodp
mkdir -p logs

DIRECT_DIR="./result_denseball/9DoF_analytic_nominalP_10/direct_param_vs1"

echo "[wait-compare] waiting for merge at $(date '+%F %T')" >> logs/compare_full.log
# Merge is done when loss_curves.npy exists AND the orchestrator logged ALL DONE.
while true; do
    if [ -f "${DIRECT_DIR}/loss_curves.npy" ] && \
       grep -q "ALL DONE" logs/orch_direct.log 2>/dev/null; then
        break
    fi
    sleep 60
done

echo "[wait-compare] merge detected, running comparison at $(date '+%F %T')" >> logs/compare_full.log
CUDA_VISIBLE_DEVICES=0 python -u compare_full.py >> logs/compare_full.log 2>&1
echo "[wait-compare] comparison done (exit $?) at $(date '+%F %T')" >> logs/compare_full.log
