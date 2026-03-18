"""
Persistent eval server for mixed-MLA benchmark.

Runs as a long-lived process inside an Apptainer container on a SLURM compute
node. Keeps aiter imported between evals — eliminates the ~5 min aiter import
overhead paid by the fresh-container approach.

Architecture:
  server process  — watches file queue, dispatches to worker
  worker process  — imports aiter ONCE, processes benchmark requests indefinitely
                    (spawn, not fork, so HIP context is clean)

File queue protocol (on shared /matx filesystem):
  pending/{uuid}.json    — client writes request:  {"code": "...", "timeout": 530}
  processing/{uuid}.json — server claims atomically (rename)
  done/{uuid}.json       — server writes result:   {"success": bool, ...}

Start:
    python eval_server.py <queue_dir> <task_dir> [gpu_id]
"""

import importlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path


BENCHMARKS = [
    {"batchsize": 4,   "qseqlen": 1, "kvseqlen": 1024, "seed": 4217},
    {"batchsize": 4,   "qseqlen": 1, "kvseqlen": 8192, "seed": 4220},
    {"batchsize": 32,  "qseqlen": 1, "kvseqlen": 1024, "seed": 5412},
    {"batchsize": 32,  "qseqlen": 1, "kvseqlen": 8192, "seed": 5415},
    {"batchsize": 64,  "qseqlen": 1, "kvseqlen": 1024, "seed": 1357},
    {"batchsize": 64,  "qseqlen": 1, "kvseqlen": 8192, "seed": 1360},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen": 1024, "seed": 9823},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen": 8192, "seed": 9826},
]


# ---------------------------------------------------------------------------
# Worker process — runs forever, imports aiter once
# ---------------------------------------------------------------------------

def _worker_main(task_dir, in_q, out_q):
    sys.path.insert(0, task_dir)

    print("[worker] Importing aiter + reference (one-time cost)...", flush=True)
    try:
        import torch
        from reference import generate_input, check_implementation
        from utils import clear_l2_cache_large
        print("[worker] Ready.", flush=True)
        out_q.put({"type": "ready"})
    except Exception as e:
        import traceback
        out_q.put({"type": "error", "msg": traceback.format_exc()})
        return

    while True:
        msg = in_q.get()
        if msg is None:
            break

        req_id = msg["id"]
        code = msg["code"]
        timeout = msg.get("timeout", 600)

        try:
            result = _run_eval(code, generate_input, check_implementation,
                               clear_l2_cache_large, torch, timeout)
        except Exception as e:
            import traceback
            result = {"success": False, "msg": f"{e}\n{traceback.format_exc()[-500:]}"}

        out_q.put({"type": "result", "id": req_id, "result": result})


def _clone(data):
    import torch
    if isinstance(data, tuple):        return tuple(_clone(x) for x in data)
    if isinstance(data, list):         return [_clone(x) for x in data]
    if isinstance(data, dict):         return {k: _clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor): return data.clone()
    return data


def _run_eval(code, generate_input, check_implementation, clear_l2_cache_large, torch, timeout):
    """Run full benchmark for one submission. Matches eval_standalone.py exactly."""
    # Write submission to temp file and import it dynamically
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                      dir="/tmp", delete=False, prefix="sub_")
    tmp.write(code)
    tmp.close()
    mod_name = f"_sub_{os.path.basename(tmp.name)[:-3]}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, tmp.name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        custom_kernel = mod.custom_kernel

        # --- Warmup (matches eval_standalone._warmup) ---
        data = generate_input(**BENCHMARKS[0])
        durations_ns = []
        bm_start = time.perf_counter_ns()
        for i in range(100):
            torch.cuda.synchronize()
            clear_l2_cache_large()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = custom_kernel(data)
            e.record()
            torch.cuda.synchronize()
            durations_ns.append(s.elapsed_time(e) * 1e6)
            del out
            if i > 1:
                mean = sum(durations_ns) / len(durations_ns)
                std = math.sqrt(sum((d - mean)**2 for d in durations_ns) / (len(durations_ns) - 1))
                err = std / math.sqrt(len(durations_ns))
                if (err / mean < 0.001
                        or mean * len(durations_ns) > 10e7
                        or time.perf_counter_ns() - bm_start > 120e9):
                    break

        # --- Benchmark each case (matches eval_standalone._run_single_benchmark) ---
        timings_us = []
        for i, bm in enumerate(BENCHMARKS):
            data = generate_input(**bm)
            check_copy = _clone(data)

            # Correctness check
            out = custom_kernel(data)
            torch.cuda.synchronize()
            result = check_implementation(check_copy, out)
            if isinstance(result, tuple):
                good, msg = result
            else:
                good = not bool(result)
                msg = str(result)
            del out
            if not good:
                return {"success": False, "msg": f"Correctness failed on benchmark {i}: {msg}"}

            # Re-generate for timing
            data = generate_input(**bm)
            durations_ns = []
            bm_start = time.perf_counter_ns()
            for j in range(1000):
                torch.cuda.synchronize()
                clear_l2_cache_large()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = custom_kernel(data)
                e.record()
                torch.cuda.synchronize()
                del out
                durations_ns.append(s.elapsed_time(e) * 1e6)
                if j > 1:
                    runs = len(durations_ns)
                    mean = sum(durations_ns) / runs
                    std = math.sqrt(sum((d - mean)**2 for d in durations_ns) / (runs - 1))
                    err = std / math.sqrt(runs)
                    if (err / mean < 0.001
                            or mean * runs > 50e9
                            or time.perf_counter_ns() - bm_start > 120e9):
                        break

            mean_us = sum(durations_ns) / len(durations_ns) / 1000.0
            timings_us.append(mean_us)

        geomean_us = math.exp(sum(math.log(t) for t in timings_us) / len(timings_us))
        details = "\n".join(f"  benchmark {i}: {t:.1f} us" for i, t in enumerate(timings_us))
        return {
            "success": True,
            "timings_us": timings_us,
            "geomean_us": geomean_us,
            "msg": f"Geomean: {geomean_us:.1f} us\n{details}",
        }
    finally:
        os.unlink(tmp.name)
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Server — file-queue dispatcher
# ---------------------------------------------------------------------------

def _start_worker(task_dir):
    ctx = mp.get_context("spawn")
    in_q  = ctx.Queue()
    out_q = ctx.Queue()
    proc  = ctx.Process(target=_worker_main, args=(task_dir, in_q, out_q), daemon=True)
    proc.start()
    print("[server] Waiting for worker to finish importing aiter...", flush=True)
    msg = out_q.get(timeout=900)   # aiter import can take ~5-7 min
    if msg["type"] != "ready":
        proc.terminate()
        raise RuntimeError(f"Worker failed to init: {msg.get('msg')}")
    return proc, in_q, out_q


def serve(queue_dir, task_dir, gpu_id=None):
    # Each SLURM job gets exactly 1 GPU; SLURM/ROCm masks it as device 0.
    # Don't override HIP_VISIBLE_DEVICES — that would mask the only available GPU.
    # gpu_id is kept for logging only.

    pending    = Path(queue_dir) / "pending"
    processing = Path(queue_dir) / "processing"
    done       = Path(queue_dir) / "done"
    for d in [pending, processing, done]:
        d.mkdir(parents=True, exist_ok=True)

    worker, in_q, out_q = _start_worker(task_dir)
    print(f"[server gpu={gpu_id}] Watching {pending}", flush=True)

    while True:
        # Atomically claim a pending request
        claimed = None
        for fpath in sorted(pending.glob("*.json")):
            dst = processing / fpath.name
            try:
                fpath.rename(dst)
                claimed = (fpath.stem, dst)
                break
            except OSError:
                continue

        if claimed is None:
            time.sleep(0.2)
            if not worker.is_alive():
                print("[server] Worker died — restarting...", flush=True)
                worker, in_q, out_q = _start_worker(task_dir)
            continue

        req_id, proc_path = claimed
        print(f"[server] Processing {req_id}", flush=True)

        try:
            request = json.loads(proc_path.read_text())
            timeout = request.get("timeout", 600)
            in_q.put({"id": req_id, "code": request["code"], "timeout": timeout})

            result = None
            deadline = time.time() + timeout + 60   # extra buffer
            while time.time() < deadline:
                try:
                    msg = out_q.get(timeout=5)
                    if msg.get("id") == req_id:
                        result = msg["result"]
                        break
                except Exception:
                    pass
                if not worker.is_alive():
                    result = {"success": False, "msg": "Worker crashed during eval"}
                    worker, in_q, out_q = _start_worker(task_dir)
                    break

            if result is None:
                result = {"success": False, "msg": "Eval timed out in server"}
                worker.terminate()
                worker.join(10)
                worker, in_q, out_q = _start_worker(task_dir)

        except Exception as e:
            result = {"success": False, "msg": str(e)}

        # Write result atomically
        tmp_path  = done / f"{req_id}.tmp"
        done_path = done / f"{req_id}.json"
        tmp_path.write_text(json.dumps(result))
        tmp_path.rename(done_path)
        proc_path.unlink(missing_ok=True)

        print(f"[server] Done {req_id}: geomean={result.get('geomean_us', result.get('msg', 'error'))}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: eval_server.py <queue_dir> <task_dir> [gpu_id]")
        sys.exit(1)
    serve(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
