MIXED_MLA_IMPROVEMENT_TEMPLATE_V0 = '''You are an expert AMD HIP kernel engineer tasked with implementing a highly optimized MLA (Multi-head Latent Attention) decode kernel for AMD MI355X.

This is the inner attention kernel from DeepSeek R1's forward_absorb MLA path.
The absorbed query and compressed KV cache are provided directly — you implement the
attention computation with variable-length batching.

The reference uses aiter's a8w8 persistent MLA kernel (fp8 Q + fp8 KV, mla_decode_fwd),
which is ~2-3x faster than bf16 on MI355X with negligible accuracy loss. However, you will have to unlock SOTA PERFORMANCE using HIP ONLY. The only deviation you are allowed to do is using -when necessary- inline AMD assembly to optimize even further.

## DeepSeek R1 Forward-Absorb MLA Config

| Parameter | Value | Notes |
|---|---|---|
| num_heads | 16 | Query heads (after TP split) |
| num_kv_heads | 1 | Single shared latent KV head |
| kv_lora_rank | 512 | Latent dimension |
| qk_rope_head_dim | 64 | RoPE embedding dimension |
| qk_head_dim | 576 | kv_lora_rank + qk_rope_head_dim (absorbed q/k dim) |
| v_head_dim | 512 | = kv_lora_rank (output dim) |
| sm_scale | 1/sqrt(576) | ~0.04167 |
| q dtype | bfloat16 | Input always bf16; reference quantizes to fp8 on-the-fly |
| kv dtype | bf16 / fp8 / mxfp4 | All three provided simultaneously |
| mode | decode | q_seq_len=1, kv_seq_len up to 8k |

## KV Buffer Format (forward_absorb)

- Full 576 dims used as keys (for Q@K^T score computation)
- First 512 dims (kv_lora_rank) used as values (for output computation)

## Input

A tuple `(q, kv_data, qo_indptr, kv_indptr, config)`:

```
q:          (total_q, 16, 576)     bfloat16  — absorbed queries
kv_data:    dict with three KV cache formats:
  "bf16":   Tensor (total_kv, 1, 576)              bfloat16
  "fp8":    (Tensor, Tensor)  kv_buffer fp8 + scalar scale
  "mxfp4":  (Tensor, Tensor)  kv_buffer fp4x2 + fp8_e8m0 scale
qo_indptr:  (batch_size + 1,)      int32     — query segment pointers
kv_indptr:  (batch_size + 1,)      int32     — KV segment pointers
config:     dict                              — MLA parameters
```

Config dict keys: batch_size, num_heads (16), num_kv_heads (1), qk_head_dim (576),
kv_lora_rank (512), qk_rope_head_dim (64), v_head_dim (512), q_seq_len, kv_seq_len, sm_scale.

## Output

```
attention_output: (total_q, 16, 512) bfloat16
```

## KV Cache Quantization

| dtype | kv_buffer | kv_scale | Bandwidth |
|---|---|---|---|
| bf16 | bfloat16 (total_kv, 1, 576) | None | 1x |
| fp8 | fp8 (total_kv, 1, 576) | scalar float32 | 2x savings |
| mxfp4 | fp4x2 (total_kv, 1, 288) | fp8_e8m0 (total_kv, N_blocks) | 4x savings |

For MXFP4: block_size=32, so 576/32=18 scale blocks per token. Two fp4 E2M1 values packed per byte.
Dequantize: unpack fp4->float, multiply by 2^(e8m0_exponent - 127) per block.

## Reference Implementation (aiter a8w8 persistent kernel)

This is the reference your submission is checked against (rtol=2e-02, atol=8e-03):

```python
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE = 1
NUM_KV_SPLITS = 32

def quantize_fp8(tensor):
    """Dynamic per-tensor FP8 quantization."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)

def ref_kernel(data):
    q, kv_data, qo_indptr, kv_indptr, config = data
    # Quantize Q to fp8
    q_input, q_scale = quantize_fp8(q)
    # Use fp8 KV cache
    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    # Build persistent-mode metadata, then call:
    # mla_decode_fwd(q, kv_buffer_4d, output, qo_indptr, kv_indptr, kv_indices,
    #                kv_last_page_len, max_q_len, page_size, nhead_kv, sm_scale,
    #                logit_cap, num_kv_splits, q_scale, kv_scale, intra_batch_mode, **meta)
    # Returns: (total_q, 16, 512) bfloat16
```

## Available Libraries and APIs

**aiter** (AMD's GPU kernel library):
- `from aiter import dtypes as aiter_dtypes` — `aiter_dtypes.fp8`, `aiter_dtypes.fp4x2`, `aiter_dtypes.fp8_e8m0`
- `from aiter import per_tensor_quant` — fast GPU-side FP8 quantization (no CPU sync)
- `from aiter.mla import mla_decode_fwd` — persistent MLA decode kernel
- `from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1` — metadata for persistent mode
- `from aiter.utility.fp4_utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32` — MXFP4 utilities

**HIP kernels** (custom C++ compiled at runtime):
- Write HIP C++ source, compile with hipcc (`/opt/rocm/bin/hipcc --offload-arch=gfx950`)
- Load via `ctypes.CDLL`, call kernel launch functions
- Use `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8` for MFMA fp8 matrix multiply
- gfx950 = MI355X architecture

**Triton** is also available for ROCm.

## MI355X-Specific Tips

- MI355X has MFMA (Matrix Fused Multiply-Add) instructions for fp8 and fp4
- `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8` does 32x32x16 fp8 matmul in hardware
- Use `per_tensor_quant` from aiter instead of manual quantize_fp8 to avoid CPU-GPU sync from `.item()`
- For split-K: use atomicAdd for work counter, threadfence + atomicAdd for done counter
- Shared memory (LDS) on gfx950: 64KB per workgroup
- 304 Compute Units on MI355X — cap grid size accordingly
- MXFP4 gives 4x bandwidth savings but requires dequant in registers or via LUT

## Optimization Ideas (pick ONE to try)

- Increase KV tile size (KVT) from 32 to 64 to reduce loop overhead
- Use ds_read_b64 or ds_read_b128 instead of ds_read_u8 for V accumulation
- Tune the split-K heuristic (_ns function) for specific batch/kv_len combos
- Unroll the inner token loop more aggressively
- Use async global loads (buffer_load) to overlap with compute
- Adjust MFMA scheduling (s_setprio, s_waitcnt tuning)
- Pack multiple V elements per LDS read to reduce LDS traffic

## Benchmark Cases (with seeds)

Your submission will be benchmarked on these 8 cases (ranking = geometric mean of latencies):

| batch_size | q_seq_len | kv_seq_len | seed |
|---|---|---|---|
| 4 | 1 | 1024 | 4217 |
| 4 | 1 | 8192 | 4220 |
| 32 | 1 | 1024 | 5412 |
| 32 | 1 | 8192 | 5415 |
| 64 | 1 | 1024 | 1357 |
| 64 | 1 | 8192 | 1360 |
| 256 | 1 | 1024 | 9823 |
| 256 | 1 | 8192 | 9826 |

## Accuracy

Submissions are checked against the a8w8 fp8 reference with `rtol=2e-02, atol=8e-03`.

Here is the last code we ran:

```python
<<<LAST_CODE>>>
```

<<<VALUE_CONTEXT>>>

Rules:
- Define all of your code in one final ```python ``` block.
- The entrypoint to your code must be named `custom_kernel`.
- You will be writing HIP C++ kernels compiled at runtime via `load_inline`, targeting AMD MI355X (gfx950, ROCm).
- All three KV cache formats (bf16, fp8, mxfp4) are available — choose the best strategy.
- Avoid `.item()` or `.cpu()` calls in the hot path — they cause CPU-GPU sync.
- Include a short docstring at the top summarizing your approach.
'''
