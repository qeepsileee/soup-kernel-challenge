"""Standalone diagnostics only. Never imported by solution.apply()."""
import os
import torch
from . import nf4_linear as production


class ExperimentalNF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, bias, state):
        ctx.state = state
        ctx.transpose_dx = os.environ.get("SOUP_EXPERIMENT_NF4_DX_LAYOUT") == "transposed"
        if production.uses_custom_gemm(x, state):
            ctx.save_for_backward(packed)
            return production.original_custom_forward(x, packed, bias, state)
        w = production.dequantize_nf4(packed, state).to(x.dtype)
        policy = os.environ.get("SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT", "0")
        retain = policy == "1" or (policy == "kv" and tuple(state.shape) == (256, 1536))
        if ctx.needs_input_grad[0] and retain:
            # This immutable tensor belongs to this autograd invocation, never a
            # shared scratch buffer. Release normally when backward is freed.
            ctx.save_for_backward(packed, w)
        else:
            ctx.save_for_backward(packed)
        return torch.nn.functional.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        saved = ctx.saved_tensors
        packed = saved[0]
        w = saved[1] if len(saved) == 2 else production.dequantize_nf4(packed, ctx.state).to(grad.dtype)
        if ctx.transpose_dx:
            # Same matrix product, different physical layout for cuBLAS.
            # Transposition cost is deliberately inside the measured backward.
            return torch.nn.functional.linear(grad, w.t().contiguous()), None, None, None
        return torch.matmul(grad, w), None, None, None

