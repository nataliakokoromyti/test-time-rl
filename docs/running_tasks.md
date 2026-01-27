# Running GPU Kernel Jobs

## Setup

Create and activate a Python environment, then install GPU mode dependencies:

```bash
pip install -r requirements/requirements-gpumode.txt
```

GPU mode environments should use Python 3.13.11.

## Running

Use `main_tinker_submitit.py` (see `docs/launching.md`) or the scripts in `scripts/`.

## Security

Run jobs on an isolated network or VPN. Ray has minimal built-in security and
should not be exposed on a public or shared network.
