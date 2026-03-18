#!/bin/bash
# Start N persistent eval server jobs on the MATX cluster.
# Each job holds one GPU and processes requests from the shared file queue.
#
# Usage:
#   ssh sc "cd /matx/u/knatalia/tttamd && bash scripts/cluster/start_eval_servers.sh [N_GPUS]"
#
# N_GPUS defaults to 8 (all GPUs on matx-amd-1).

set -euo pipefail

N_GPUS=${1:-8}
QUEUE_DIR="/matx/u/knatalia/eval_queue"
TASK_DIR="/matx/u/knatalia/tttamd/gpu_mode/mixed-mla"
EVAL_SERVER="/matx/u/knatalia/tttamd/tasks/gpu_mode/eval_server.py"
SIF="/matx/u/knatalia/rocm_pytorch.sif"
PYPACKAGES="/matx/u/knatalia/pypackages"
TRITON_CACHE="/matx/u/knatalia/.triton_cache"
TORCH_EXT="/matx/u/knatalia/.torch_extensions"
PYTHON="/opt/venv/bin/python3"
LOG_DIR="/matx/u/knatalia/eval_server_logs"

mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/processing" "$QUEUE_DIR/done"
mkdir -p "$LOG_DIR"

echo "Starting $N_GPUS eval server(s)..."

for ((i=0; i<N_GPUS; i++)); do
    LOG="$LOG_DIR/server_gpu${i}.log"
    JOB_NAME="eval_server_gpu${i}"

    sbatch \
        --account=matx \
        --partition=matx-interactive \
        --gres=gpu:1 \
        --time=8:00:00 \
        --job-name="$JOB_NAME" \
        --output="$LOG" \
        --error="$LOG" \
        --wrap="
APPTAINERENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
APPTAINERENV_PYTHONPATH=$PYPACKAGES \
APPTAINERENV_TRITON_CACHE_DIR=$TRITON_CACHE \
APPTAINERENV_TORCH_EXTENSIONS_DIR=$TORCH_EXT \
/usr/bin/apptainer exec --rocm --bind /matx $SIF \
$PYTHON $EVAL_SERVER $QUEUE_DIR $TASK_DIR $i
"
    echo "  Submitted gpu $i -> log: $LOG"
done

echo ""
echo "Monitor logs:   tail -f $LOG_DIR/server_gpu0.log"
echo "Check jobs:     squeue -u \$USER"
echo "Queue dir:      $QUEUE_DIR"
