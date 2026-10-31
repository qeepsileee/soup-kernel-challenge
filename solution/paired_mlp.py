"""Standalone fused frozen MLP candidate. NOT enabled by solution.apply().

Pair gate/up weights in caller-owned temporary storage. One GEMM emits both
projections and one backward GEMM accumulates their input gradients. INT8 saved
activations match the current compressor. Formula-derived example_ kernels do
not count toward the original-kernel admission rule.
"""
import torch
import triton
import triton.language as tl
from .nf4 import dequantize_nf4


@triton.jit
def example_paired_compressed_fwd(Z, Y, QG, QU, SG, SU,
                                  N, H: tl.constexpr, TILE: tl.constexpr, GROUP: tl.constexpr):
    groups = tl.program_id(0) * (TILE // GROUP) + tl.arange(0, TILE // GROUP)
    i = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    offset = (i // H) * (2 * H) + i % H
    g = tl.load(Z + offset, mask=i < N, other=0.).to(tl.float32)
    u = tl.load(Z + offset + H, mask=i < N, other=0.).to(tl.float32)
    tl.store(Y + i, g * tl.sigmoid(g) * u, mask=i < N)
    gs = tl.max(tl.abs(g), 1) / 127.
    us = tl.max(tl.abs(u), 1) / 127.
    gs = tl.where(gs > 0, gs, 1.)
    us = tl.where(us > 0, us, 1.)
    tl.store(QG + i, tl.floor(g / gs[:, None] + 0.5).to(tl.int8), mask=i < N)
    tl.store(QU + i, tl.floor(u / us[:, None] + 0.5).to(tl.int8), mask=i < N)
    tl.store(SG + groups, gs, mask=groups < tl.cdiv(N, GROUP))
    tl.store(SU + groups, us, mask=groups < tl.cdiv(N, GROUP))


@triton.jit
def example_paired_compressed_bwd(DY, QG, QU, SG, SU, DZ,
                                  N, H: tl.constexpr, TILE: tl.constexpr, GROUP: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    g = tl.load(QG + i, mask=i < N, other=0).to(tl.float32)
    u = tl.load(QU + i, mask=i < N, other=0).to(tl.float32)
    g *= tl.load(SG + i // GROUP, mask=i < N, other=0)
    u *= tl.load(SU + i // GROUP, mask=i < N, other=0)
    dy = tl.load(DY + i, mask=i < N, other=0).to(tl.float32)
    sig = tl.sigmoid(g)
    du = dy * g * sig
    dg = dy * u * sig * (1. + g * (1. - sig))
    offset = (i // H) * (2 * H) + i % H
    tl.store(DZ + offset, dg, mask=i < N)
    tl.store(DZ + offset + H, du, mask=i < N)


def _decode_pair(gate, up, gate_state, up_state):
    h, d = gate_state.shape
    w = torch.empty((2 * h, d), device=gate.device, dtype=torch.float16)
    dequantize_nf4(gate, gate_state, out=w[:h])
    dequantize_nf4(up, up_state, out=w[h:])
    return w


class PairedFrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, up, down, gate_state, up_state, down_state):
        h, d = gate_state.shape
        shape = x.shape
        flat = x.contiguous().view(-1, d)
        rows = flat.shape[0]
        n = rows * h
        with torch.autocast("cuda", enabled=False):
            w = _decode_pair(gate, up, gate_state, up_state)
            z = torch.nn.functional.linear(flat, w)
            del w
            hidden = torch.empty((rows, h), device=x.device, dtype=torch.float16)
            qg = torch.empty((rows, h), device=x.device, dtype=torch.int8)
            qu = torch.empty_like(qg)
            sg = torch.empty(triton.cdiv(n, 64), device=x.device, dtype=torch.float32)
            su = torch.empty_like(sg)
            with torch.cuda.device(x.device):
                example_paired_compressed_fwd[(triton.cdiv(n, 1024),)](
                    z, hidden, qg, qu, sg, su, n, h, 1024, 64, num_warps=4)
            del z
            wd = dequantize_nf4(down, down_state)
            out = torch.nn.functional.linear(hidden, wd)
        ctx.shape = shape
        ctx.states = gate_state, up_state, down_state
        ctx.save_for_backward(gate, up, down, qg, qu, sg, su)
        return out.view(shape)

    @staticmethod
    def backward(ctx, grad):
        gate, up, down, qg, qu, sg, su = ctx.saved_tensors
        gate_state, up_state, down_state = ctx.states
        h, d = gate_state.shape
        rows = qg.numel() // h
        n = qg.numel()
        with torch.autocast("cuda", enabled=False):
            wd = dequantize_nf4(down, down_state)
            dh = grad.contiguous().view(rows, d) @ wd
            del wd
            dz = torch.empty((rows, 2 * h), device=grad.device, dtype=torch.float16)
            with torch.cuda.device(grad.device):
                example_paired_compressed_bwd[(triton.cdiv(n, 1024),)](
                    dh, qg, qu, sg, su, dz, n, h, 1024, 64, num_warps=4)
            del dh
            w = _decode_pair(gate, up, gate_state, up_state)
            dx = dz @ w
        return dx.view(ctx.shape), None, None, None, None, None, None


def paired_mlp(module, x):
    from .mlp_bridge import can_bridge
    if not can_bridge(module, x) or x.dtype != torch.float16:
        raise ValueError("paired MLP requires a supported frozen NF4 MLP and FP16 CUDA input")
    if x.numel() // x.shape[-1] < 512:
        raise ValueError("paired MLP is restricted to >=512 token rows; preserve tiny-M bnb dispatch")
    gate, up, down = (getattr(module, n).weight for n in ("gate_proj", "up_proj", "down_proj"))
    gs, us, ds = gate.quant_state, up.quant_state, down.quant_state
    if (tuple(gs.shape) != tuple(us.shape) or gs.shape[0] % 64
            or tuple(ds.shape) != (gs.shape[1], gs.shape[0])
            or any(s.dtype != torch.float16 for s in (gs, us, ds))):
        details = [(tuple(s.shape), str(s.dtype)) for s in (gs, us, ds)]
        raise ValueError("paired MLP requires FP16 quant states, matching gate/up and transposed down "
                         f"dimensions; gate/up/down states={details}")
    return PairedFrozenMLP.apply(x, gate, up, down, gs, us, ds)


def patch_paired_mlp():
    """Explicit diagnostic opt-in; not called by production apply()."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from .mlp_bridge import can_bridge
    original = Qwen2MLP.forward
    if getattr(original, "_paired_frozen_mlp", False):
        return

    def forward(self, x):
        if (self.training and torch.is_grad_enabled() and x.dtype == torch.float16
                and x.numel() // x.shape[-1] >= 512 and can_bridge(self, x)):
            g, u, d = (getattr(self, n).weight.quant_state for n in ("gate_proj", "up_proj", "down_proj"))
            if (tuple(g.shape) == tuple(u.shape) and g.shape[0] % 64 == 0
                    and tuple(d.shape) == (g.shape[1], g.shape[0])
                    and all(s.dtype == torch.float16 for s in (g, u, d))):
                return paired_mlp(self, x)
        return original(self, x)

    forward._paired_frozen_mlp = True
    forward._original = original
    Qwen2MLP.forward = forward
