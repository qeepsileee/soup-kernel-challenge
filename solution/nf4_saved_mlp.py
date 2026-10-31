"""Standalone 4-bit SAVED ACTIVATION experiment; never imported by apply().

Forward uses original FP16 gate/up values. Backward reconstructs an approximate
pair using group-64 NF4 values. This does not change model weight quantization.
Formula-derived example_ kernels do NOT count as original-kernel contributions.
"""
import torch
import triton
import triton.language as tl
from .nf4_format import NF4_VALUES


@triton.jit
def _encode_normalized_nf4(x, VALUES: tl.constexpr):
    code = tl.full(x.shape, 0, tl.int32)
    for j in tl.static_range(15):
        code += (x > (VALUES[j] + VALUES[j + 1]) * 0.5).to(tl.int32)
    return code


@triton.jit
def _decode_normalized_nf4(code, VALUES: tl.constexpr):
    value = tl.full(code.shape, VALUES[0], tl.float32)
    for j in tl.static_range(1, 16):
        value = tl.where(code == j, VALUES[j], value)
    return value


@triton.jit
def example_nf4_saved_swiglu_fwd(G, U, Y, Packed, SG, SU,
                                 N, VALUES: tl.constexpr, TILE: tl.constexpr, GROUP: tl.constexpr):
    groups = tl.program_id(0) * (TILE // GROUP) + tl.arange(0, TILE // GROUP)
    i = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    g = tl.load(G + i, mask=i < N, other=0.).to(tl.float32)
    u = tl.load(U + i, mask=i < N, other=0.).to(tl.float32)
    # Store the original forward result BEFORE quantizing saved activations.
    tl.store(Y + i, g * tl.sigmoid(g) * u, mask=i < N)
    gs, us = tl.max(tl.abs(g), 1), tl.max(tl.abs(u), 1)
    gs, us = tl.where(gs > 0, gs, 1.), tl.where(us > 0, us, 1.)
    qg = _encode_normalized_nf4(g / gs[:, None], VALUES)
    qu = _encode_normalized_nf4(u / us[:, None], VALUES)
    tl.store(Packed + i, ((qg << 4) | qu).to(tl.uint8), mask=i < N)
    tl.store(SG + groups, gs, mask=groups < tl.cdiv(N, GROUP))
    tl.store(SU + groups, us, mask=groups < tl.cdiv(N, GROUP))


@triton.jit
def example_nf4_saved_swiglu_bwd(DY, Packed, SG, SU, DG, DU,
                                 N, VALUES: tl.constexpr, TILE: tl.constexpr, GROUP: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    packed = tl.load(Packed + i, mask=i < N, other=0x77).to(tl.int32)
    g = _decode_normalized_nf4((packed >> 4) & 15, VALUES)
    u = _decode_normalized_nf4(packed & 15, VALUES)
    g *= tl.load(SG + i // GROUP, mask=i < N, other=0)
    u *= tl.load(SU + i // GROUP, mask=i < N, other=0)
    dy = tl.load(DY + i, mask=i < N, other=0).to(tl.float32)
    sig = tl.sigmoid(g)
    tl.store(DG + i, dy * u * sig * (1. + g * (1. - sig)), mask=i < N)
    tl.store(DU + i, dy * g * sig, mask=i < N)


class NF4SavedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if (not gate.is_cuda or gate.dtype != torch.float16 or gate.dtype != up.dtype
                or gate.device != up.device or gate.shape != up.shape):
            raise ValueError("NF4 saved activations require matching FP16 CUDA gate/up")
        gate, up = gate.contiguous(), up.contiguous()
        n = gate.numel()
        packed = torch.empty(gate.shape, device=gate.device, dtype=torch.uint8)
        sg = torch.empty(triton.cdiv(n, 64), device=gate.device, dtype=torch.float32)
        su = torch.empty_like(sg)
        out = torch.empty_like(gate)
        if n:
            with torch.cuda.device(gate.device):
                example_nf4_saved_swiglu_fwd[(triton.cdiv(n, 1024),)](
                    gate, up, out, packed, sg, su, n, NF4_VALUES, 1024, 64, num_warps=4)
        ctx.save_for_backward(packed, sg, su)
        return out

    @staticmethod
    def backward(ctx, grad):
        packed, sg, su = ctx.saved_tensors
        dg = torch.empty(packed.shape, device=packed.device, dtype=torch.float16)
        du = torch.empty_like(dg)
        n = packed.numel()
        if n:
            with torch.cuda.device(packed.device):
                example_nf4_saved_swiglu_bwd[(triton.cdiv(n, 1024),)](
                    grad.to(torch.float16).contiguous(), packed, sg, su, dg, du,
                    n, NF4_VALUES, 1024, 64, num_warps=4)
        return dg, du
