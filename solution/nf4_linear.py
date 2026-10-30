"""NF4 linear with our decoder in forward AND backward, cuBLAS for GEMMs."""
from importlib.metadata import version
import torch
from .nf4 import dequantize_nf4, supported


class NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, bias, state):
        ctx.state = state
        ctx.save_for_backward(packed)
        w = dequantize_nf4(packed, state).to(x.dtype)
        return torch.nn.functional.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        packed, = ctx.saved_tensors
        w = dequantize_nf4(packed, ctx.state).to(grad.dtype)
        return torch.matmul(grad, w), None, None, None


def patch_linear():
    if version("bitsandbytes") != "0.50.2":
        return
    import bitsandbytes as bnb
    original = bnb.matmul_4bit
    if getattr(original, "_nf4_linear_candidate", False):
        return

    def linear(A, B, quant_state, out=None, bias=None):
        if (out is None and A.dtype == torch.float16 and A.numel() > 0
                and supported(B, quant_state) and A.device == B.device
                and A.shape[-1] == quant_state.shape[1] and not B.requires_grad
                and (bias is None or (not bias.requires_grad and bias.dtype == A.dtype))):
            return NF4Linear.apply(A, B, bias, quant_state)
        return original(A, B, quant_state, out=out, bias=bias)

    linear._nf4_linear_candidate = True
    linear._original = original
    bnb.matmul_4bit = linear


def unpatch_linear():
    import bitsandbytes as bnb
    if getattr(bnb.matmul_4bit, "_nf4_linear_candidate", False):
        bnb.matmul_4bit = bnb.matmul_4bit._original
