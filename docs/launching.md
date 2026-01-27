# Launching GPU Kernel Jobs

Scripts live in `scripts/` (trimul/mla only). If they are empty, use
`main_tinker_submitit.py` directly.

## Example (single node)

```bash
python main_tinker_submitit.py \
    --nodes 1 \
    --partition default \
    --cpus-per-task 64 \
    env=trimul \
    model_name="openai/gpt-oss-120b" \
    sampler_type=greedy \
    initial_exp_type=random \
    num_epochs=50
```

## Example (multi-node)

```bash
python main_tinker_submitit.py \
    --nodes 4 \
    --partition default \
    --cpus-per-task 100 \
    env=trimul \
    model_name="openai/gpt-oss-120b" \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=50
```

## GPU Mode Options

- `env`: `trimul`, `mla_decode_nvidia`, or `nvfp4_group_gemm`
- `problem_idx`: string identifier used for prompt variants
- `gpu_mode_score_scale`: reciprocal reward scale (see `tinker_cookbook/recipes/ttt/train.py`)
