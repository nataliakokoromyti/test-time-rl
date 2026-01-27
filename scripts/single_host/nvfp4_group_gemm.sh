#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if command -v cmd.exe >/dev/null 2>&1; then
  export PYTHONPATH="${PWD};${PWD}/tasks;${PWD}/gpu_mode;${PYTHONPATH:-}"
else
  export PYTHONPATH="${PWD}:${PWD}/tasks:${PWD}/gpu_mode:${PYTHONPATH:-}"
fi
unset RAY_ADDRESS

# Required API keys
# export TINKER_API_KEY="..."
# export WANDB_API_KEY="..."
# export WANDB_ENTITY="..."

# W&B config
wandb_project="ttt-discover-gpu"
wandb_name="nvfp4-group-gemm-seed0"
wandb_project="${wandb_project//$'\r'/}"
wandb_name="${wandb_name//$'\r'/}"

# Cluster config
nnodes=1
cpus_per_task=64
partition="default"
account="default"

if command -v cmd.exe >/dev/null 2>&1; then
  PYTHON_EXE="${PYTHON_EXE:-C:\\Users\\natal\\anaconda\\python.exe}"
  CMD="set KMP_DUPLICATE_LIB_OK=TRUE && ${PYTHON_EXE} -m tinker_cookbook.recipes.ttt.train"
  CMD="${CMD} env=nvfp4_group_gemm"
  CMD="${CMD} model_name=openai/gpt-oss-120b"
  CMD="${CMD} lora_rank=32"
  CMD="${CMD} learning_rate=4e-5"
  CMD="${CMD} temperature=1.0"
  CMD="${CMD} max_tokens=32768"
  CMD="${CMD} two_phase_sampling=true"
  CMD="${CMD} phase1_max_tokens=26000"
  CMD="${CMD} groups_per_batch=8"
  CMD="${CMD} group_size=64"
  CMD="${CMD} sampler_type=puct_backprop"
  CMD="${CMD} initial_exp_type=random"
  CMD="${CMD} num_epochs=50"
  CMD="${CMD} kl_penalty_coef=0.1"
  CMD="${CMD} adv_estimator=entropic_adaptive_beta"
  CMD="${CMD} adv_estimator_beta=0.693147"
  CMD="${CMD} seed=0"
  CMD="${CMD} wandb_project=${wandb_project}"
  CMD="${CMD} wandb_name=${wandb_name}"
  cmd.exe /c "${CMD}"
else
  PYTHON_EXE="${PYTHON_EXE:-python3.11}"
  if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
    PYTHON_EXE="python3"
  fi
  export KMP_DUPLICATE_LIB_OK=TRUE
  "${PYTHON_EXE}" -m tinker_cookbook.recipes.ttt.train \
    env=nvfp4_group_gemm \
    model_name=openai/gpt-oss-120b \
    lora_rank=32 \
    learning_rate=4e-5 \
    temperature=1.0 \
    max_tokens=32768 \
    dynamic_max_tokens=true \
    two_phase_sampling=true \
    phase1_max_tokens=26000 \
    groups_per_batch=8 \
    group_size=64 \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=5 \
    kl_penalty_coef=0.1 \
    adv_estimator=entropic_adaptive_beta \
    adv_estimator_beta=0.693147 \
    seed=0 \
    wandb_project="${wandb_project}" \
    wandb_name="${wandb_name}"
fi
