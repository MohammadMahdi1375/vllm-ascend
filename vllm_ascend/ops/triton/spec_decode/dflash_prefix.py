# SPDX-License-Identifier: Apache-2.0
"""Greedy full-prefix walk. Neural candidate interactions are prepared separately."""

import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit(do_not_specialize=["num_reqs"])
def prefix_greedy_walk_kernel(
    ids_ptr,
    unary_ptr,
    numerator_ptr,
    denominator_ptr,
    anchor_num_ptr,
    anchor_den_ptr,
    gate_ptr,
    output_ptr,
    num_reqs,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    tile_size: tl.constexpr,
):
    # Keep accumulators for all query candidates on the vector core. Pair
    # tables use [request, source_candidate, query_candidate] so each selected
    # source contributes a contiguous row. No per-token host dispatch.
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    offsets = tl.arange(0, tile_size)
    width = num_steps * top_k
    gate = tl.load(gate_ptr).to(tl.float32)
    req = pid
    while req < num_reqs:
        base = req * width + offsets
        unary = tl.load(unary_ptr + base, offsets < width, other=0).to(tl.float32)
        num = tl.load(anchor_num_ptr + base, offsets < width, other=0).to(tl.float32)
        den = tl.load(anchor_den_ptr + base, offsets < width, other=1).to(tl.float32)
        for step in range(num_steps):
            scores = unary + (gate * num) / den
            current = (offsets < width) & (offsets // top_k == step)
            scores = tl.where(current, scores, float("-inf"))
            maximum = tl.max(scores, 0)
            selected = tl.min(
                tl.where(current & (scores == maximum), offsets, tile_size), 0
            )
            token = tl.load(ids_ptr + req * width + selected)
            tl.store(output_ptr + req * num_steps + step, token)
            if step + 1 < num_steps:
                pair_base = (req * width + selected) * width + offsets
                future = (offsets < width) & (offsets // top_k > step)
                num += tl.load(numerator_ptr + pair_base, future, other=0).to(
                    tl.float32
                )
                den += tl.load(denominator_ptr + pair_base, future, other=0).to(
                    tl.float32
                )
        req += programs


def greedy_select_prefix(tables):
    """Run the fused NPU walk with the same first-index tie break as torch.argmax."""
    ids, unary, pair_num, pair_den, anchor_num, anchor_den, gate = tables
    n, steps, k = ids.shape
    if ids.device.type != "npu":
        raise ValueError("The fused prefix walk requires an Ascend NPU")
    output = torch.empty((n, steps), dtype=ids.dtype, device=ids.device)
    if n == 0:
        return output
    width = steps * k
    # Table construction is a parallel GEMM; this transpose makes the dependent
    # walk read contiguous source rows rather than strided candidate columns.
    pair_num = pair_num.reshape(n, width, width).transpose(1, 2).contiguous()
    pair_den = pair_den.reshape(n, width, width).transpose(1, 2).contiguous()
    init_device_properties_triton()
    prefix_greedy_walk_kernel[(min(n, get_vectorcore_num()),)](
        ids.contiguous(),
        unary.contiguous(),
        pair_num,
        pair_den,
        anchor_num.contiguous(),
        anchor_den.contiguous(),
        gate,
        output,
        n,
        num_steps=steps,
        top_k=k,
        tile_size=triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output
