#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export PYTHONPATH="${PWD}:${PWD}/tasks:${PYTHONPATH:-}"
unset RAY_ADDRESS

# Required API keys
# export TINKER_API_KEY="..."
# export WANDB_API_KEY="..."
# export WANDB_ENTITY="..."

# W&B config
wandb_project="ttt-discover-gpu"
wandb_name="nvfp4-group-gemm-seed0"

# Cluster config
nnodes=4
cpus_per_task=100
partition="default"
account="default"

python main_tinker_submitit.py \
  --nodes "${nnodes}" \
  --partition "${partition}" \
  --account "${account}" \
  --cpus-per-task "${cpus_per_task}" \
  env=nvfp4_group_gemm \
  model_name="openai/gpt-oss-120b" \
  lora_rank=32 \
  learning_rate=4e-5 \
  temperature=1.0 \
  max_tokens=32768 \
  two_phase_sampling=true \
  phase1_max_tokens=26000 \
  groups_per_batch=8 \
  group_size=64 \
  sampler_type=puct_backprop \
  initial_exp_type=random \
  num_epochs=50 \
  kl_penalty_coef=0.1 \
  adv_estimator=entropic_adaptive_beta \
  adv_estimator_beta=0.693147 \
  seed=0 \
  wandb_project="${wandb_project}" \
  wandb_name="${wandb_name}"
