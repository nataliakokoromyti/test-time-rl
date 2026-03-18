"""
MLA Decode Triton Kernel for MI355X (gfx950).
Split-K with tiled V accumulation. Scores computed once per KV block.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

SM_SCALE = 1.0 / (576 ** 0.5)


@triton.jit
def _mla_phase1(
    Q, KV, KV_indptr,
    PO, PM, PL,
    sm_scale,
    num_splits: tl.constexpr,
    stride_q_batch, stride_q_head, stride_kv_tok,
    stride_po_batch, stride_po_head, stride_po_split,
    stride_pm_batch, stride_pm_head,
    BLOCK_KV: tl.constexpr,
    D_TILE: tl.constexpr,
    N_D_TILES: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    V_TILE: tl.constexpr,
    N_V_TILES: tl.constexpr,
    MAX_KV_BLOCKS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_split = tl.program_id(1)
    batch_idx = pid_bh // 16
    head_idx = pid_bh % 16

    kv_start = tl.load(KV_indptr + batch_idx)
    kv_end = tl.load(KV_indptr + batch_idx + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, num_splits)
    my_start = kv_start + pid_split * split_size
    my_end = tl.minimum(kv_start + pid_split * split_size + split_size, kv_end)

    q_ptr = Q + batch_idx * stride_q_batch + head_idx * stride_q_head

    tok_offs = tl.arange(0, BLOCK_KV)
    d_offs = tl.arange(0, D_TILE)
    vt_offs = tl.arange(0, V_TILE)

    m_i: tl.float32 = float("-inf")
    l_i: tl.float32 = 0.0

    # 8 V-tile accumulators (V_TILE=64 each, 512 total)
    acc0 = tl.zeros([V_TILE], dtype=tl.float32)
    acc1 = tl.zeros([V_TILE], dtype=tl.float32)
    acc2 = tl.zeros([V_TILE], dtype=tl.float32)
    acc3 = tl.zeros([V_TILE], dtype=tl.float32)
    acc4 = tl.zeros([V_TILE], dtype=tl.float32)
    acc5 = tl.zeros([V_TILE], dtype=tl.float32)
    acc6 = tl.zeros([V_TILE], dtype=tl.float32)
    acc7 = tl.zeros([V_TILE], dtype=tl.float32)

    for block_idx in range(MAX_KV_BLOCKS):
        tok_start = my_start + block_idx * BLOCK_KV
        tok_ids = tok_start + tok_offs
        tok_valid = tok_ids < my_end

        # Tiled QK dot product (576 dims in 9 tiles of 64)
        scores = tl.zeros([BLOCK_KV], dtype=tl.float32)
        for dt in range(N_D_TILES):
            d_start = dt * D_TILE
            d_off = d_start + d_offs
            d_mask = d_off < QK_DIM

            q_tile = tl.load(q_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
            kv_ptrs = KV + tok_ids[:, None] * stride_kv_tok + d_off[None, :]
            kv_tile = tl.load(kv_ptrs, mask=tok_valid[:, None] & d_mask[None, :],
                              other=0.0).to(tl.float32)
            scores += tl.sum(kv_tile * q_tile[None, :], axis=1)

        scores = scores * sm_scale
        scores = tl.where(tok_valid, scores, float("-inf"))

        # Online softmax
        block_max = tl.max(scores)
        new_m = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(scores - new_m)
        p = tl.where(tok_valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = new_m

        # Rescale all V accumulators
        acc0 *= alpha
        acc1 *= alpha
        acc2 *= alpha
        acc3 *= alpha
        acc4 *= alpha
        acc5 *= alpha
        acc6 *= alpha
        acc7 *= alpha

        # Accumulate V tiles (first 512 dims of KV buffer)
        v_base = KV + tok_ids[:, None] * stride_kv_tok
        v_mask = tok_valid[:, None]

        vd = vt_offs
        acc0 += tl.sum(p[:, None] * tl.load(v_base + (0 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc1 += tl.sum(p[:, None] * tl.load(v_base + (1 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc2 += tl.sum(p[:, None] * tl.load(v_base + (2 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc3 += tl.sum(p[:, None] * tl.load(v_base + (3 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc4 += tl.sum(p[:, None] * tl.load(v_base + (4 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc5 += tl.sum(p[:, None] * tl.load(v_base + (5 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc6 += tl.sum(p[:, None] * tl.load(v_base + (6 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)
        acc7 += tl.sum(p[:, None] * tl.load(v_base + (7 * V_TILE + vd)[None, :], mask=v_mask, other=0.0).to(tl.float32), axis=0)

    # Store partials
    po_base = (PO + batch_idx * stride_po_batch + head_idx * stride_po_head
               + pid_split * stride_po_split)
    tl.store(po_base + 0 * V_TILE + vt_offs, acc0)
    tl.store(po_base + 1 * V_TILE + vt_offs, acc1)
    tl.store(po_base + 2 * V_TILE + vt_offs, acc2)
    tl.store(po_base + 3 * V_TILE + vt_offs, acc3)
    tl.store(po_base + 4 * V_TILE + vt_offs, acc4)
    tl.store(po_base + 5 * V_TILE + vt_offs, acc5)
    tl.store(po_base + 6 * V_TILE + vt_offs, acc6)
    tl.store(po_base + 7 * V_TILE + vt_offs, acc7)

    pm_off = batch_idx * stride_pm_batch + head_idx * stride_pm_head + pid_split
    tl.store(PM + pm_off, m_i)
    tl.store(PL + pm_off, l_i)


@triton.jit
def _mla_phase2(
    PO, PM, PL, Out,
    num_splits: tl.constexpr,
    stride_po_batch, stride_po_head, stride_po_split,
    stride_pm_batch, stride_pm_head,
    stride_o_batch, stride_o_head,
    V_DIM: tl.constexpr,
    V_TILE: tl.constexpr,
    N_V_TILES: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    batch_idx = pid_bh // 16
    head_idx = pid_bh % 16

    pm_base = batch_idx * stride_pm_batch + head_idx * stride_pm_head

    # Find global max
    global_m: tl.float32 = float("-inf")
    for s in range(num_splits):
        global_m = tl.maximum(global_m, tl.load(PM + pm_base + s))

    # Compute normalizer
    global_l: tl.float32 = 0.0
    for s in range(num_splits):
        m_s = tl.load(PM + pm_base + s)
        l_s = tl.load(PL + pm_base + s)
        global_l += l_s * tl.exp(m_s - global_m)
    inv_l = 1.0 / global_l

    # Pre-compute per-split scales
    po_base = PO + batch_idx * stride_po_batch + head_idx * stride_po_head
    o_base = Out + batch_idx * stride_o_batch + head_idx * stride_o_head
    vt_offs = tl.arange(0, V_TILE)

    for vt in range(N_V_TILES):
        v_off = vt * V_TILE
        result = tl.zeros([V_TILE], dtype=tl.float32)
        for s in range(num_splits):
            m_s = tl.load(PM + pm_base + s)
            scale = tl.exp(m_s - global_m) * inv_l
            partial = tl.load(po_base + s * stride_po_split + v_off + vt_offs)
            result += partial * scale
        tl.store(o_base + v_off + vt_offs, result.to(tl.bfloat16))


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    kv_seq_len = config["kv_seq_len"]
    v_head_dim = config["v_head_dim"]
    qk_head_dim = config["qk_head_dim"]
    total_q = q.shape[0]

    kv_buffer = kv_data["bf16"].squeeze(1)  # (total_kv, 576)

    D_TILE = 64
    V_TILE = 64
    BLOCK_KV = 16
    N_V_TILES = v_head_dim // V_TILE       # 8
    N_D_TILES = (qk_head_dim + D_TILE - 1) // D_TILE  # 9

    # Adaptive splits — target ~2048+ thread blocks for good CU occupancy
    # MI355X has 304 CUs, want at least ~4 waves per CU
    total_heads = batch_size * num_heads  # grid dim 0
    if kv_seq_len <= 1024:
        if total_heads <= 64:
            num_splits = 32  # 64*32=2048 blocks
        elif total_heads <= 512:
            num_splits = 8
        else:
            num_splits = 4
    else:
        if total_heads <= 64:
            num_splits = 64  # 64*64=4096 blocks, each handles 128 tokens
        elif total_heads <= 512:
            num_splits = 16
        else:
            num_splits = 8

    # Max KV blocks any split could process
    max_split_tokens = (kv_seq_len + num_splits - 1) // num_splits
    MAX_KV_BLOCKS = (max_split_tokens + BLOCK_KV - 1) // BLOCK_KV

    partial_o = torch.empty((batch_size, num_heads, num_splits, v_head_dim),
                            dtype=torch.float32, device="cuda")
    partial_m = torch.full((batch_size, num_heads, num_splits),
                           float("-inf"), dtype=torch.float32, device="cuda")
    partial_l = torch.zeros((batch_size, num_heads, num_splits),
                            dtype=torch.float32, device="cuda")

    grid1 = (batch_size * num_heads, num_splits)
    _mla_phase1[grid1](
        q, kv_buffer, kv_indptr,
        partial_o, partial_m, partial_l,
        SM_SCALE, num_splits,
        q.stride(0), q.stride(1), kv_buffer.stride(0),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_m.stride(0), partial_m.stride(1),
        BLOCK_KV=BLOCK_KV,
        D_TILE=D_TILE,
        N_D_TILES=N_D_TILES,
        QK_DIM=qk_head_dim,
        V_DIM=v_head_dim,
        V_TILE=V_TILE,
        N_V_TILES=N_V_TILES,
        MAX_KV_BLOCKS=MAX_KV_BLOCKS,
    )

    output = torch.empty((total_q, num_heads, v_head_dim), dtype=torch.bfloat16, device="cuda")
    grid2 = (batch_size * num_heads,)
    _mla_phase2[grid2](
        partial_o, partial_m, partial_l, output,
        num_splits,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_m.stride(0), partial_m.stride(1),
        output.stride(0), output.stride(1),
        V_DIM=v_head_dim,
        V_TILE=V_TILE,
        N_V_TILES=N_V_TILES,
    )

    return output
