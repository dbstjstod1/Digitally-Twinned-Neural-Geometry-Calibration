#!/usr/bin/env bash
# Direct per-view 9-DoF baseline over all 480 views, split across 2 GPUs.
#   GPU 0: views [0,240)
#   GPU 1: views [240,480)
# After both shards finish, merge into full arrays + loss_history + time.
set -u
cd /home/mirlab/Desktop/Digitally-Twinned-Neural-Geometry-Calibration
source /home/mirlab/anaconda3/etc/profile.d/conda.sh
conda activate dudodp
mkdir -p logs

NITERS=150
LR=5e-2
OUT="./result_denseball/9DoF_analytic_nominalP_10/direct_param_vs1"

echo "[orch-direct] start $(date '+%F %T')  niters=${NITERS} lr=${LR}" >> logs/orch_direct.log

CUDA_VISIBLE_DEVICES=0 python -u run_direct_param.py run --start 0   --stop 240 \
    --niters ${NITERS} --lr ${LR} --outdir "${OUT}" > logs/direct_gpu0.log 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python -u run_direct_param.py run --start 240 --stop 480 \
    --niters ${NITERS} --lr ${LR} --outdir "${OUT}" > logs/direct_gpu1.log 2>&1 &
PID1=$!

echo "[orch-direct] launched shard PIDs ${PID0} (GPU0), ${PID1} (GPU1)" >> logs/orch_direct.log

wait ${PID0}; R0=$?
wait ${PID1}; R1=$?
echo "[orch-direct] shards done (exit ${R0}/${R1}) at $(date '+%F %T')" >> logs/orch_direct.log

if [ "${R0}" -eq 0 ] && [ "${R1}" -eq 0 ]; then
    python -u run_direct_param.py merge --outdir "${OUT}" > logs/direct_merge.log 2>&1
    echo "[orch-direct] merge done (exit $?) at $(date '+%F %T')" >> logs/orch_direct.log
else
    echo "[orch-direct] SKIP merge: a shard failed" >> logs/orch_direct.log
fi
echo "[orch-direct] ALL DONE $(date '+%F %T')" >> logs/orch_direct.log
