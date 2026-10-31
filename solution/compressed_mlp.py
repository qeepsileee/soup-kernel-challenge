"""Block-INT8 saved activations, enabled by the validated production apply().

Forward SwiGLU is identical; only saved gate/up for backward are quantized.
The activation formulas derive from the shipped example: these example_-named
kernels are NOT original-kernel contributions for the >=5% rule.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def example_compressed_swiglu_fwd(G, U, Y, QG, QU, SG, SU,
                                 N, TILE: tl.constexpr, GROUP: tl.constexpr):
    groups = tl.program_id(0) * (TILE // GROUP) + tl.arange(0, TILE // GROUP)
    i = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    g = tl.load(G + i, mask=i < N, other=0.).to(tl.float32)
    u = tl.load(U + i, mask=i < N, other=0.).to(tl.float32)
    # Same arithmetic and FP16 store as the supplied example, BEFORE compression.
    y = g * tl.sigmoid(g) * u
    tl.store(Y + i, y, mask=i < N)
    gs = tl.max(tl.abs(g), 1) / 127.
    us = tl.max(tl.abs(u), 1) / 127.
    gs = tl.where(gs > 0, gs, 1.)
    us = tl.where(us > 0, us, 1.)
    qg = tl.floor(g / gs[:, None] + 0.5).to(tl.int8)
    qu = tl.floor(u / us[:, None] + 0.5).to(tl.int8)
    tl.store(QG + i, qg, mask=i < N)
    tl.store(QU + i, qu, mask=i < N)
    tl.store(SG + groups, gs, mask=groups < tl.cdiv(N, GROUP))
    tl.store(SU + groups, us, mask=groups < tl.cdiv(N, GROUP))


@triton.jit
def example_compressed_swiglu_bwd(DY, QG, QU, SG, SU, DG, DU,
                                 N, TILE: tl.constexpr, GROUP: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    g = tl.load(QG + i, mask=i < N, other=0).to(tl.float32)
    u = tl.load(QU + i, mask=i < N, other=0).to(tl.float32)
    g *= tl.load(SG + i // GROUP, mask=i < N, other=0)
    u *= tl.load(SU + i // GROUP, mask=i < N, other=0)
    dy = tl.load(DY + i, mask=i < N, other=0).to(tl.float32)
    sig = tl.sigmoid(g)
    du = dy * g * sig
    dg = dy * u * sig * (1. + g * (1. - sig))
    tl.store(DG + i, dg, mask=i < N)
    tl.store(DU + i, du, mask=i < N)


class CompressedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if (not gate.is_cuda or gate.dtype != torch.float16 or up.dtype != gate.dtype
                or gate.shape != up.shape or up.device != gate.device):
            raise ValueError("compression experiment requires matching FP16 CUDA activations")
        gate, up = gate.contiguous(), up.contiguous()
        n = gate.numel()
        qg = torch.empty(gate.shape, device=gate.device, dtype=torch.int8)
        qu = torch.empty_like(qg)
        sg = torch.empty(triton.cdiv(n, 64), device=gate.device, dtype=torch.float32)
        su = torch.empty_like(sg)
        out = torch.empty_like(gate)
        if n:
            with torch.cuda.device(gate.device):
                example_compressed_swiglu_fwd[(triton.cdiv(n, 1024),)](
                    gate, up, out, qg, qu, sg, su, n, 1024, 64, num_warps=4)
        ctx.save_for_backward(qg, qu, sg, su)
        return out

    @staticmethod
    def backward(ctx, grad):
        qg, qu, sg, su = ctx.saved_tensors
        dg = torch.empty(qg.shape, device=qg.device, dtype=torch.float16)
        du = torch.empty_like(dg)
        n = qg.numel()
        if n:
            with torch.cuda.device(qg.device):
                example_compressed_swiglu_bwd[(triton.cdiv(n, 1024),)](
                    grad.to(torch.float16).contiguous(), qg, qu, sg, su,
                    dg, du, n, 1024, 64, num_warps=4)
        return dg, du


def patch_compressed_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from .mlp_bridge import can_bridge, bridged_mlp
    original = Qwen2MLP.forward
    if getattr(original, "_int8_saved_activations", False):
        return

    def forward(self, x):
        if self.training and torch.is_grad_enabled() and can_bridge(self, x):
            return bridged_mlp(self, x, activation=CompressedSwiGLU.apply)
        return original(self, x)

    forward._int8_saved_activations = True
    forward._original = original
    Qwen2MLP.forward = forward


def unpatch_compressed_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    current = Qwen2MLP.forward
    if getattr(current, "_int8_saved_activations", False):
        Qwen2MLP.forward = current._original
