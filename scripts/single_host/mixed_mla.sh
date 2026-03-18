#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export PYTHONPATH="${PWD}:${PWD}/tasks:${PYTHONPATH:-}"
unset RAY_ADDRESS

# Required API keys — set these before running
# export TINKER_API_KEY="..."

# AMD cluster eval config
export MIXED_MLA_REMOTE_EVAL=1
export MIXED_MLA_REMOTE_DIR=/matx/u/knatalia/tttamd

export WANDB_MODE=disabled
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUTF8=1

PYTHONUTF8=1 KMP_DUPLICATE_LIB_OK=TRUE /mnt/c/Users/natal/anaconda/python.exe -m tinker_cookbook.recipes.ttt.train \
  env=mixed_mla \
  model_name="openai/gpt-oss-120b" \
  lora_rank=32 \
  learning_rate=4e-5 \
  temperature=1.0 \
  max_tokens=32768 \
  two_phase_sampling=true \
  phase1_max_tokens=26000 \
  groups_per_batch=8 \
  group_size=8 \
  sampler_type=puct_backprop \
  initial_exp_type=random \
  num_epochs=50 \
  eval_timeout=530 \
  dataset_timeout=530 \
  kl_penalty_coef=0.1 \
  adv_estimator=entropic_adaptive_beta \
  adv_estimator_beta=0.693147 \
  seed=0
