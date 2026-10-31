"""NF4 linear: retain bnb's custom GEMM selection, accelerate its unfused path."""
from importlib.metadata import version
import os
import torch
from .nf4 import dequantize_nf4, supported


def uses_custom_gemm(x, state):
    """Mirror the installed 0.50.2 dispatch, without copying its heuristic code."""
    from bitsandbytes.backends.cuda import ops

    m = x.numel() // x.shape[-1]
    if m > ops._gemm_4bit_custom_max_m or x.shape[-1] % state.blocksize:
        return False
    return ops._gemm_4bit_use_custom_fn(
        x.device.index, x.dtype, m, state.shape[0], state.shape[1])


def original_custom_forward(x, packed, bias, state):
    # Preserve the fused GEMM's weight/bias arithmetic exactly. In particular,
    # materializing rounded FP16 W before cuBLAS is not equivalent at tiny M.
    if state.nested:
        if state.state2.blocksize != 256:
            raise NotImplementedError("bnb 0.50.2 fused GEMM requires scale blocksize 256")
        return torch.ops.bitsandbytes.gemm_4bit.default(
            x, packed.view(-1, 1), state.shape, state.state2.absmax,
            state.blocksize, state.quant_type, bias=bias,
            absmax_8bit=state.absmax, absmax_code=state.state2.code,
            absmax_offset=state.offset)
    return torch.ops.bitsandbytes.gemm_4bit.default(
        x, packed.view(-1, 1), state.shape, state.absmax,
        state.blocksize, state.quant_type, bias=bias)


class NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, bias, state):
        ctx.state = state
        if uses_custom_gemm(x, state):
            ctx.save_for_backward(packed)
            return original_custom_forward(x, packed, bias, state)
        w = dequantize_nf4(packed, state).to(x.dtype)
        if ctx.needs_input_grad[0] and os.environ.get("SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT") == "1":
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
        w = saved[1] if len(saved) == 2 else dequantize_nf4(packed, ctx.state).to(grad.dtype)
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
