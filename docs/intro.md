# GPU Kernel Optimization Overview

This repo is focused on reproducing the GPU kernel optimization experiments.
Shared training scaffolding remains intact, but non-GPU datasets/results were removed.

Key files:
- `tasks/gpu_mode/task.py`: async grading via GPUMode runners
- `tasks/gpu_mode/prompt_trimul.py`: TriMul prompt template
- `tasks/gpu_mode/prompt_mla_decode.py`: MLA decode prompt template
- `tinker_cookbook/recipes/ttt/env_gpu_mode.py`: environment wiring

Dataset names (set via `env=...`) include:
- `trimul`
- `mla_decode_nvidia`
- `nvfp4_group_gemm`
