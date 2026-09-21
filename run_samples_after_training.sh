#!/usr/bin/env bash
# Wait for each AI_Geocal training process to finish, then run its Sample export
# on the same GPU. Each view_step is handled in its own background subshell so
# a sample starts as soon as ITS training finishes (pipelined).
#
# Training PID / GPU mapping (set by the parallel launch):
#   vs1 -> PID 2650767, GPU 0
#   vs2 -> PID 2650768, GPU 0
#   vs4 -> PID 2650769, GPU 1
#   vs8 -> PID 2650770, GPU 1

set -u
cd /home/mirlab/Desktop/Digitally-Twinned-Neural-Geometry-Calibration

source /home/mirlab/anaconda3/etc/profile.d/conda.sh
conda activate dudodp
mkdir -p logs

run_after() {
    local vs="$1" pid="$2" gpu="$3"
    echo "[orch] vs${vs}: waiting for training PID ${pid} (GPU ${gpu}) to finish..." \
        >> logs/orch.log

    # Wait until the training process is gone.
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 30
    done

    echo "[orch] vs${vs}: training PID ${pid} ended at $(date '+%F %T')." >> logs/orch.log

    local train_dir="./result_denseball/9DoF_analytic_nominalP_10/no_initP_k_un_vn_SOD_SDD_vs${vs}"
    # Require the final checkpoint before exporting.
    if ! ls "${train_dir}"/motion_model_ep*.pth >/dev/null 2>&1; then
        echo "[orch] vs${vs}: NO checkpoint in ${train_dir}; skipping sample." >> logs/orch.log
        return 1
    fi

    echo "[orch] vs${vs}: launching sample on GPU ${gpu} at $(date '+%F %T')." >> logs/orch.log
    CUDA_VISIBLE_DEVICES="${gpu}" python -u run_single_sample.py "${vs}" \
        > "logs/sample_vs${vs}.log" 2>&1
    echo "[orch] vs${vs}: sample finished (exit $?) at $(date '+%F %T')." >> logs/orch.log
}

# Launch one waiter per view_step.
run_after 1 2650767 0 &
run_after 2 2650768 0 &
run_after 4 2650769 1 &
run_after 8 2650770 1 &

wait
echo "[orch] all samples done at $(date '+%F %T')." >> logs/orch.log
