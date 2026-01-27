<p align="center">
  <h1 align="center">TTT-Discover (GPU Kernel Optimization)</h1>
  <h3 align="center">Test-time training for GPUMode kernel search</h3>
</p>

This repo is focused on reproducing the GPU kernel optimization experiments only.
Non-GPU datasets/results and requirements were removed to slim the tree, but shared
training code and harnesses remain intact.

## Installation

```bash
pip install -r requirements/requirements-gpumode.txt
```

Set environment variables:

```bash
export TINKER_API_KEY="..."
export WANDB_API_KEY="..."
export WANDB_ENTITY="..."
```

## Quick Start

Use a preconfigured script:

```bash
bash scripts/single_host/trimul.sh
```

Multi-node (SLURM) example:

```bash
bash scripts/multi_host/trimul.sh
```

For advanced launch options, see `docs/launching.md`.

## GPU Paths

- GPU task: `tasks/gpu_mode/`
- GPU tooling: `gpu_mode/`
- TTT training loop: `tinker_cookbook/recipes/ttt/`
- Submitit launcher: `main_tinker_submitit.py`
- GPU results: `results/kernel-engineering/`

Available GPU tasks: `trimul`, `mla_decode_nvidia`, `nvfp4_group_gemm`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
