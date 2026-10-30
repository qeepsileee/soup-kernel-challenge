"""FP16 input-gradient candidates for bnb 0.50.2 frozen weights/bias.

Default: dequantize once, then independently implemented tiled dense GEMM.
Experimental inline-decode GEMM remains available for tests/benchmarks only;
it was much slower on T4 and is NOT the default training path.
"""

from importlib.metadata import version
from math import prod

import torch
import triton
import triton.language as tl

from .nf4 import dequantize_nf4, supported


@triton.jit
def nf4_dense_dx_kernel(Gradient, Weight, Result,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + reduction
        left = tl.load(Gradient + row[:, None] * K + k[None, :],
                       mask=(row[:, None] < M) & (k[None, :] < K), other=0)
        right = tl.load(Weight + k[:, None] * N + col[None, :],
                        mask=(k[:, None] < K) & (col[None, :] < N), other=0)
        total = tl.dot(left, right, total)
    tl.store(Result + row[:, None] * N + col[None, :], total,
             mask=(row[:, None] < M) & (col[None, :] < N))


def input_grad_dense(gradient, packed, state):
    if (not supported(packed, state) or gradient.dtype != torch.float16
            or gradient.device != packed.device or gradient.ndim < 1
            or gradient.shape[-1] != state.shape[0]):
        raise ValueError("NF4 dX requires supported NF4 state and FP16 gradient")
    gradient = gradient.contiguous()
    k, n = state.shape
    m = prod(gradient.shape[:-1])
    result = torch.empty((*gradient.shape[:-1], n), device=gradient.device, dtype=gradient.dtype)
    if m == 0:
        return result
    # One temporary W, local to this call; never saved in autograd or shared.
    weight = dequantize_nf4(packed, state).to(gradient.dtype)
    with torch.cuda.device(gradient.device):
        nf4_dense_dx_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            gradient, weight, result, m, n, k, 32, 64, 64,
            num_warps=4, num_stages=2)
    return result


@triton.jit
def nf4_input_grad_kernel(Grad, Packed, Absmax, Code, ScaleCode, ScaleAbsmax,
                          Offset, Result, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, Q_BLOCK: tl.constexpr,
                          NESTED: tl.constexpr, SCALE_BLOCK: tl.constexpr,
                          SOURCE_HALF: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        rk = tile * BK + inner
        g = tl.load(Grad + rows[:, None] * K + rk[None, :],
                    mask=(rows[:, None] < M) & (rk[None, :] < K), other=0)
        flat = rk[:, None] * N + cols[None, :]
        valid = (rk[:, None] < K) & (cols[None, :] < N)
        byte = tl.load(Packed + flat // 2, mask=valid, other=0).to(tl.int32)
        idx = (byte >> tl.where(flat % 2 == 0, 4, 0)) & 15
        value = tl.load(Code + idx)
        block = flat // Q_BLOCK
        if NESTED:
            si = tl.load(Absmax + block, mask=valid, other=0).to(tl.int32)
            sc = tl.load(ScaleCode + si)
            outer = tl.load(ScaleAbsmax + block // SCALE_BLOCK, mask=valid, other=0)
            scale = sc * outer
            scale = scale + tl.load(Offset)
        else:
            scale = tl.load(Absmax + block, mask=valid, other=0)
        weight = value * scale
        if SOURCE_HALF:
            weight = weight.to(tl.float16)
        # Equivalent to bnb dequantize(...).to(grad_output.dtype).
        w = tl.where(valid, weight, 0).to(tl.float16)
        accumulator = tl.dot(g, w, accumulator)
    tl.store(Result + rows[:, None] * N + cols[None, :], accumulator,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def input_grad_nf4(gradient, packed, state):
    if (not supported(packed, state) or gradient.dtype != torch.float16
            or gradient.device != packed.device or gradient.ndim < 1
            or gradient.shape[-1] != state.shape[0]):
        raise ValueError("fused NF4 dX requires supported NF4 state and FP16 gradient")
    gradient = gradient.contiguous()
    k, n = state.shape
    m = prod(gradient.shape[:-1])
    result = torch.empty((*gradient.shape[:-1], n), device=gradient.device, dtype=gradient.dtype)
    if m == 0:
        return result
    second = state.state2 if state.nested else None
    with torch.cuda.device(gradient.device):
        nf4_input_grad_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            gradient, packed, state.absmax, state.code,
            second.code if second else state.code,
            second.absmax if second else state.absmax,
            state.offset if second else state.absmax,
            result, m, n, k, state.blocksize, state.nested,
            second.blocksize if second else 1, state.dtype == torch.float16,
            32, 64, 32, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return result


def patch_input_grad(*, fused=False):
    """Keep forward/LoRA intact. Never change weight or bias gradient semantics."""
    if version("bitsandbytes") != "0.50.2":
        return
    from bitsandbytes.autograd._functions import MatMul4Bit

    original = MatMul4Bit.backward
    if getattr(original, "_soup_fused_nf4_dx", False):
        return

    def backward(ctx, gradient):
        if (not ctx.is_empty and ctx.needs_input_grad[0]
                and not ctx.needs_input_grad[1] and not ctx.needs_input_grad[3]
                and ctx.dtype_A == torch.float16 and gradient.dtype == torch.float16):
            packed = ctx.tensors[1]
            if (supported(packed, ctx.state) and gradient.device == packed.device
                    and gradient.shape[-1] == ctx.state.shape[0]):
                implementation = input_grad_nf4 if fused else input_grad_dense
                dx = implementation(gradient, packed, ctx.state)
                return dx, None, None, None, None
        return original(ctx, gradient)

    backward._soup_fused_nf4_dx = True
    backward._nf4_dx_mode = "inline_decode" if fused else "decode_then_gemm"
    backward._original = original
    MatMul4Bit.backward = staticmethod(backward)


def unpatch_input_grad():
    from bitsandbytes.autograd._functions import MatMul4Bit

    current = MatMul4Bit.backward
    if getattr(current, "_soup_fused_nf4_dx", False):
        MatMul4Bit.backward = staticmethod(current._original)
