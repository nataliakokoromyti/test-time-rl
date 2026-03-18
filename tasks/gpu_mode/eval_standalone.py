"""Standalone eval runner — no POPCORN_FD needed. Prints JSON to stdout.

Usage:
    python eval_standalone.py /path/to/submission.py /path/to/task_dir [gpu_id]

Matches GPU Mode's official eval.py `benchmark` mode exactly:
  - spawn Pool(1) for process isolation
  - Same warmup (100 reps, 100ms cap)
  - Same timing loop (1000 reps, no data clone in timed region, 50s kernel-time cap)
  - Same convergence (err/mean < 0.1%, or mean*runs > max_time_ns, or wallclock > 2min)
  - Same L2 clearing, same CUDA event timing, same del output

Output:
  {"success": bool, "timings_us": [...], "geomean_us": float, "msg": str}
"""
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# Worker functions — run in a SPAWNED subprocess (clean HIP/CUDA context)
# ---------------------------------------------------------------------------

def _worker_init(task_dir, submission_dir):
    """Called once when the Pool(1) worker starts."""
    sys.path.insert(0, task_dir)
    sys.path.insert(0, submission_dir)  # submission_dir must win over task_dir


def _clone(data):
    import torch
    if isinstance(data, tuple):   return tuple(_clone(x) for x in data)
    if isinstance(data, list):    return [_clone(x) for x in data]
    if isinstance(data, dict):    return {k: _clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor): return data.clone()
    return data


def _warmup(benchmark, max_repeats=100, max_time_ns=10e7):
    """Warmup: up to 100 reps or 100ms of kernel time.  Matches GPU Mode."""
    import torch
    from reference import generate_input
    from submission import custom_kernel
    from utils import clear_l2_cache_large

    data = generate_input(**benchmark)
    # One correctness-free warmup pass (same as GPU Mode's run_single_benchmark
    # called with recheck=False from run_benchmarking warmup)
    durations_ns = []
    bm_start = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        clear_l2_cache_large()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        output = custom_kernel(data)
        end_ev.record()
        torch.cuda.synchronize()
        dur_ns = start_ev.elapsed_time(end_ev) * 1e6  # ms → ns
        durations_ns.append(dur_ns)
        del output
        if i > 1:
            mean = sum(durations_ns) / len(durations_ns)
            std  = math.sqrt(sum((d - mean)**2 for d in durations_ns) / (len(durations_ns) - 1))
            err  = std / math.sqrt(len(durations_ns))
            if err / mean < 0.001 or mean * len(durations_ns) > max_time_ns or (time.perf_counter_ns() - bm_start) > 120e9:
                break
    return True


def _run_single_benchmark(benchmark, max_repeats=1000, max_time_ns=50e9):
    """Run correctness + timing for one benchmark case.

    Matches GPU Mode eval.py _run_single_benchmark with recheck=False.
    """
    import torch
    from reference import check_implementation, generate_input
    from submission import custom_kernel
    from utils import clear_l2_cache_large

    # Generate input once, reuse for all timing iterations (no clone in timed region)
    data = generate_input(**benchmark)
    check_copy = _clone(data)

    # --- One obligatory correctness check (matches GPU Mode) ---
    output = custom_kernel(data)
    torch.cuda.synchronize()
    result = check_implementation(check_copy, output)
    if isinstance(result, tuple):
        good, msg = result
    else:
        good = not bool(result)
        msg = str(result)
    del output

    if not good:
        return {"ok": False, "msg": msg}

    # Re-generate data for timing (correctness check may have consumed it)
    data = generate_input(**benchmark)

    # --- Benchmark timing loop (exact match to GPU Mode) ---
    # Durations stored in nanoseconds internally (same as GPU Mode: elapsed_time * 1e6)
    durations_ns = []
    bm_start = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        clear_l2_cache_large()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        output = custom_kernel(data)       # NO clone — matches GPU Mode
        end_ev.record()
        torch.cuda.synchronize()

        del output
        durations_ns.append(start_ev.elapsed_time(end_ev) * 1e6)  # ms → ns

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start
            runs = len(durations_ns)
            mean = sum(durations_ns) / runs
            std  = math.sqrt(sum((d - mean)**2 for d in durations_ns) / (runs - 1))
            err  = std / math.sqrt(runs)
            # Same three stopping conditions as GPU Mode:
            # a) relative error < 0.1%
            # b) cumulative kernel time > max_time_ns (50s)
            # c) wall-clock > 2 minutes
            if err / mean < 0.001 or mean * runs > max_time_ns or total_bm_duration > 120e9:
                break

    # Convert mean to microseconds for output
    mean_ns = sum(durations_ns) / len(durations_ns)
    mean_us = mean_ns / 1000.0
    return {"ok": True, "mean_us": mean_us, "runs": len(durations_ns)}


# ---------------------------------------------------------------------------
# Main — runs in the parent process, delegates to subprocess pool
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "msg": "Usage: eval_standalone.py <submission.py> <task_dir> [gpu_id]"}))
        sys.exit(1)

    submission_path = sys.argv[1]
    task_dir = sys.argv[2]
    gpu_id = sys.argv[3] if len(sys.argv) > 3 else None

    if gpu_id is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Unique temp dir for this eval — avoids file races between concurrent evals
    eval_tmp = tempfile.mkdtemp(prefix="eval_sub_")
    shutil.copy2(submission_path, os.path.join(eval_tmp, "submission.py"))

    benchmarks = [
        {"batchsize": 4,   "qseqlen": 1, "kvseqlen": 1024, "seed": 4217},
        {"batchsize": 4,   "qseqlen": 1, "kvseqlen": 8192, "seed": 4220},
        {"batchsize": 32,  "qseqlen": 1, "kvseqlen": 1024, "seed": 5412},
        {"batchsize": 32,  "qseqlen": 1, "kvseqlen": 8192, "seed": 5415},
        {"batchsize": 64,  "qseqlen": 1, "kvseqlen": 1024, "seed": 1357},
        {"batchsize": 64,  "qseqlen": 1, "kvseqlen": 8192, "seed": 1360},
        {"batchsize": 256, "qseqlen": 1, "kvseqlen": 1024, "seed": 9823},
        {"batchsize": 256, "qseqlen": 1, "kvseqlen": 8192, "seed": 9826},
    ]

    try:
        # Spawn context → clean process, no inherited HIP/CUDA state
        mp_ctx = multiprocessing.get_context("spawn")
        pool = mp_ctx.Pool(
            1,
            initializer=_worker_init,
            initargs=(task_dir, eval_tmp),
        )

        try:
            # Warmup — same as GPU Mode: run_single_benchmark(tests[0], False, 100, 10e7)
            pool.apply(_warmup, (benchmarks[0],))

            timings_us = []
            for i, bm in enumerate(benchmarks):
                try:
                    result = pool.apply(_run_single_benchmark, (bm,))
                except Exception as e:
                    # Worker crashed (OOM, segfault, corrupt HIP state)
                    pool.terminate()
                    print(json.dumps({
                        "success": False,
                        "msg": f"Benchmark {i} crashed worker: {e}",
                    }))
                    return

                if not result["ok"]:
                    print(json.dumps({
                        "success": False,
                        "msg": f"Correctness failed on benchmark {i}: {result['msg']}",
                    }))
                    return

                timings_us.append(result["mean_us"])

        finally:
            pool.terminate()
            pool.join()

        geomean_us = math.exp(sum(math.log(t) for t in timings_us) / len(timings_us))
        details = [f"benchmark {i}: {t:.1f} us" for i, t in enumerate(timings_us)]
        print(json.dumps({
            "success": True,
            "timings_us": timings_us,
            "geomean_us": geomean_us,
            "msg": f"Geomean: {geomean_us:.1f} us\n" + "\n".join(f"  {d}" for d in details),
        }))

    except Exception as e:
        import traceback
        print(json.dumps({
            "success": False,
            "msg": f"Exception: {e}\n{traceback.format_exc()[-500:]}",
        }))
    finally:
        shutil.rmtree(eval_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
