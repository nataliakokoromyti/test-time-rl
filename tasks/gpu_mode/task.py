import asyncio
import logging
import math
import os
import re
import shutil
import tempfile

logger = logging.getLogger(__name__)

# Lazy imports — only needed for Modal/Nvidia path, not popcorn
_compute_score = None
_run_on_modal = None

def _get_modal_deps():
    global _compute_score, _run_on_modal
    if _compute_score is None:
        from gpu_mode.libkernelbot.submission import compute_score
        from gpu_mode.run_modal import run_on_modal
        _compute_score = compute_score
        _run_on_modal = run_on_modal
    return _compute_score, _run_on_modal


def get_gpu_mode_error(msg):
    return {
        "score": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "performance": -1_000_000,
    }


# ---------------------------------------------------------------------------
# Popcorn-CLI leaderboard config per task
# ---------------------------------------------------------------------------
POPCORN_TASKS = {
    "mixed_mla": {
        "leaderboard": "amd-mixed-mla",
        "gpu": "MI355X",
        "task_dir": "gpu_mode/mixed-mla",
    },
}


# ---------------------------------------------------------------------------
# Local eval on AMD GPUs (opt-in via MIXED_MLA_LOCAL_EVAL=1)
# ---------------------------------------------------------------------------
_local_gpu_semaphores: dict[int, asyncio.Semaphore] = {}
_local_gpu_counter = 0
_local_gpu_lock = None  # initialized lazily


def _get_local_gpu_count() -> int:
    """Return number of local GPUs available for eval."""
    override = os.environ.get("MIXED_MLA_LOCAL_GPUS")
    if override:
        return int(override)
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _get_local_gpu_lock():
    global _local_gpu_lock
    if _local_gpu_lock is None:
        _local_gpu_lock = asyncio.Lock()
    return _local_gpu_lock


async def _acquire_gpu() -> int:
    """Acquire a GPU index for local eval, round-robin with concurrency limit."""
    global _local_gpu_counter, _local_gpu_semaphores
    n_gpus = _get_local_gpu_count()
    max_concurrent = int(os.environ.get("MIXED_MLA_LOCAL_CONCURRENT_PER_GPU", "1"))

    async with _get_local_gpu_lock():
        gpu_id = _local_gpu_counter % n_gpus
        _local_gpu_counter += 1
        if gpu_id not in _local_gpu_semaphores:
            _local_gpu_semaphores[gpu_id] = asyncio.Semaphore(max_concurrent)

    await _local_gpu_semaphores[gpu_id].acquire()
    return gpu_id


def _release_gpu(gpu_id: int):
    """Release a GPU back to the pool."""
    if gpu_id in _local_gpu_semaphores:
        _local_gpu_semaphores[gpu_id].release()


async def run_remote_eval_task(
    generation: str,
    task_name: str,
    score_scale: float,
) -> dict:
    """Run eval on a remote AMD cluster via SSH + SLURM. Training stays on laptop.

    SSH to login node (sc), then srun to get a GPU allocation on the compute node.

    Env vars:
        MIXED_MLA_SSH_HOST  — SSH config host alias (default: "sc")
        MIXED_MLA_REMOTE_DIR — path to tttamd repo on the cluster (required)
        MIXED_MLA_REMOTE_PYTHON — python exe on cluster (default: "python3")
        MIXED_MLA_SLURM_ACCOUNT — SLURM account (default: "matx")
        MIXED_MLA_SLURM_PARTITION — SLURM partition (default: "matx-interactive")
        MIXED_MLA_SLURM_TIME — SLURM time limit (default: "30:00")
    """
    ssh_host = os.environ.get("MIXED_MLA_SSH_HOST", "sc")
    remote_dir = os.environ.get("MIXED_MLA_REMOTE_DIR")
    if not remote_dir:
        return get_gpu_mode_error("MIXED_MLA_REMOTE_DIR not set")

    slurm_account = os.environ.get("MIXED_MLA_SLURM_ACCOUNT", "matx")
    slurm_partition = os.environ.get("MIXED_MLA_SLURM_PARTITION", "matx-interactive")
    slurm_time = os.environ.get("MIXED_MLA_SLURM_TIME", "30:00")
    sif = os.environ.get("MIXED_MLA_SIF", "/matx/u/knatalia/rocm_pytorch.sif")
    pypackages = os.environ.get("MIXED_MLA_PYPACKAGES", "/matx/u/knatalia/pypackages")

    eval_script = f"{remote_dir}/tasks/gpu_mode/eval_standalone.py"
    task_dir = f"{remote_dir}/gpu_mode/mixed-mla"
    apptainer_python = "/opt/venv/bin/python3"

    try:
        # Pipe submission code via stdin, write to /matx (shared across nodes).
        # srun allocates a GPU, apptainer runs eval inside ROCm container.
        triton_cache = os.environ.get("MIXED_MLA_TRITON_CACHE", "/matx/u/knatalia/.triton_cache")
        torch_ext_dir = os.environ.get("MIXED_MLA_TORCH_EXTENSIONS_DIR", "/matx/u/knatalia/.torch_extensions")
        remote_cmd = (
            f"TMPF=$(mktemp /matx/u/knatalia/submission_XXXXXX.py) && "
            f"cat > $TMPF && "
            f"srun --account={slurm_account} --partition={slurm_partition} "
            f"--gres=gpu:1 --time={slurm_time} "
            f'bash -c "'
            f"APPTAINERENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu "
            f"APPTAINERENV_PYTHONPATH={pypackages} "
            f"APPTAINERENV_TRITON_CACHE_DIR={triton_cache} "
            f"APPTAINERENV_TORCH_EXTENSIONS_DIR={torch_ext_dir} "
            f"/usr/bin/apptainer exec --rocm --bind /matx {sif} "
            f"{apptainer_python} {eval_script} $TMPF {task_dir}"
            f'" ; '
            f"rm -f $TMPF"
        )

        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            ssh_host, remote_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=generation.encode("utf-8")),
                timeout=900,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return get_gpu_mode_error("Remote eval timed out (900s)")

        output = stdout.decode("utf-8", errors="replace").strip()
        stderr_str = stderr.decode("utf-8", errors="replace").strip()

        if not output:
            return get_gpu_mode_error(f"Remote eval returned no output. stderr: {stderr_str[-500:]}")

        logger.info(f"Remote eval output: {output[:500]}")

    except Exception as e:
        return get_gpu_mode_error(f"SSH/SLURM error: {e}")

    # Parse JSON output from eval_standalone.py.
    # aiter prints log lines to stdout, so find the last line that parses as JSON.
    import json
    parsed = None
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None:
        return get_gpu_mode_error(f"Failed to parse remote eval JSON: {output[:500]}")

    if not parsed.get("success"):
        return get_gpu_mode_error(parsed.get("msg", "Remote eval failed"))

    geomean_us = parsed.get("geomean_us")
    if geomean_us is None:
        return get_gpu_mode_error("No benchmark timings obtained")

    score = score_scale / geomean_us

    return {
        "score": score,
        "msg": parsed["msg"],
        "correctness": 1.0,
        "performance": -geomean_us,
        "benchmark_details": "\n".join(
            f"  benchmark {i}: {t:.1f} µs" for i, t in enumerate(parsed.get("timings_us", []))
        ),
    }


def _parse_popcorn_benchmark_output(output: str) -> dict:
    """Parse popcorn-cli --no-tui benchmark stdout into structured results.

    Returns dict with keys: success, correctness, timings_us, geomean_us, msg
    """
    # Check for outright failure
    if "fail" in output.lower() and "success" not in output.lower():
        return {"success": False, "correctness": 0.0, "timings_us": [], "geomean_us": None, "msg": output[-500:]}

    # Check for benchmarking success
    if "Benchmarking successful" not in output and "Testing successful" not in output:
        # Could be an error
        if "Application error" in output:
            error_match = re.search(r"Application error:.*", output)
            err_msg = error_match.group(0) if error_match else "Unknown application error"
            return {"success": False, "correctness": 0.0, "timings_us": [], "geomean_us": None, "msg": err_msg}
        if "fail" in output.lower():
            return {"success": False, "correctness": 0.0, "timings_us": [], "geomean_us": None, "msg": "Benchmark failed"}

    # Parse mean timings: "⏱ 511 ± 2.2 µs" or "⏱ 3.48 ± 0.007 ms"
    timing_pattern = re.compile(r"⏱\s+([\d.]+)\s*±\s*[\d.]+\s*(µs|ms)")
    timings_us = []
    for match in timing_pattern.finditer(output):
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "ms":
            value *= 1000.0
        timings_us.append(value)

    if not timings_us:
        # Test-only mode — check for pass/fail
        if "Passed" in output:
            return {"success": True, "correctness": 1.0, "timings_us": [], "geomean_us": None, "msg": "Tests passed (no benchmark timings)"}
        return {"success": False, "correctness": 0.0, "timings_us": [], "geomean_us": None, "msg": "No benchmark timings found"}

    geomean_us = math.exp(sum(math.log(t) for t in timings_us) / len(timings_us))

    # Build per-benchmark details string
    details = "\n".join(f"  benchmark {i}: {t:.1f} µs" for i, t in enumerate(timings_us))
    msg = f"Geomean: {geomean_us:.1f} µs\n{details}"

    return {
        "success": True,
        "correctness": 1.0,
        "timings_us": timings_us,
        "geomean_us": geomean_us,
        "msg": msg,
    }


async def run_popcorn_task(
    generation: str,
    task_name: str,
    score_scale: float,
    mode: str = "benchmark",
) -> dict:
    """Run a submission via popcorn-cli and parse results."""
    popcorn_cfg = POPCORN_TASKS.get(task_name)
    if popcorn_cfg is None:
        return get_gpu_mode_error(f"No popcorn config for task: {task_name}")

    leaderboard = popcorn_cfg["leaderboard"]
    gpu = popcorn_cfg["gpu"]

    # Write generated code to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="popcorn_")
    tmp_file = os.path.join(tmp_dir, "submission.py")
    with open(tmp_file, "w") as f:
        f.write(generation)

    cmd = [
        "popcorn-cli", "submit",
        "--gpu", gpu,
        "--leaderboard", leaderboard,
        "--mode", mode,
        "--no-tui",
        tmp_file,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
        output = stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            # popcorn-cli may still print useful output on non-zero exit
            output += "\n" + stderr.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        return get_gpu_mode_error("Popcorn-cli submission timed out (900s)")
    except Exception as e:
        return get_gpu_mode_error(f"Failed to run popcorn-cli: {e}")
    finally:
        # Cleanup temp file
        try:
            os.remove(tmp_file)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    parsed = _parse_popcorn_benchmark_output(output)

    if not parsed["success"]:
        return get_gpu_mode_error(parsed["msg"])

    if parsed["geomean_us"] is None:
        return get_gpu_mode_error("No benchmark timings obtained")

    geomean_us = parsed["geomean_us"]
    score = score_scale / geomean_us

    return {
        "score": score,
        "msg": parsed["msg"],
        "correctness": 1.0,
        "performance": -geomean_us,
        "benchmark_details": "\n".join(
            f"  benchmark {i}: {t:.1f} µs" for i, t in enumerate(parsed["timings_us"])
        ),
    }


# ---------------------------------------------------------------------------
# Main dispatcher — routes to Modal or Popcorn based on task
# ---------------------------------------------------------------------------
async def run_gpu_mode_task(generation: str, gpu_type: str, task_name: str, score_scale: float, app_name: str):

    # Route AMD/popcorn tasks — remote SLURM eval if opted in, else popcorn-cli
    if task_name in POPCORN_TASKS:
        if os.environ.get("MIXED_MLA_REMOTE_EVAL", "0") == "1" and task_name == "mixed_mla":
            return await run_remote_eval_task(
                generation=generation,
                task_name=task_name,
                score_scale=score_scale,
            )
        return await run_popcorn_task(
            generation=generation,
            task_name=task_name,
            score_scale=score_scale,
            mode="benchmark",
        )

    # Otherwise use Modal (existing path for Nvidia tasks)
    compute_score, run_on_modal = _get_modal_deps()
    result, task = await run_on_modal(
        submission_code=generation,
        gpu_type=gpu_type,
        mode="leaderboard",
        task_name=task_name,
        app_name=app_name,
    )

    if not result.success:
        return get_gpu_mode_error(f"Error: Failed to run test: {result.error}.")

    # Unexpected
    if "test" not in result.runs:
        return get_gpu_mode_error(f"Unexpected result: Failed to find test results.")

    test_results = result.runs["test"]

    # Probably compile error
    if not test_results.run.success:
        return get_gpu_mode_error(f"Failed to run tests: {test_results.run.stderr}")

    # Failed test cases
    if not test_results.run.passed:
        return get_gpu_mode_error(f"Failed to pass test cases.")

    if task is not None and "leaderboard" in result.runs:
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000
            msg = f"\nOverall leaderboard score (microseconds, {task.ranking_by.value}): {score_us} us"
        except Exception as e:
            return get_gpu_mode_error(f"Could not compute leaderboard score: {e}")

    score = score_scale / score_us

    return {
        "score": score,
        "msg": msg,
        "correctness": 1.0,
        "performance": -score_us,
    }
