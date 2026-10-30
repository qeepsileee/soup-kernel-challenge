"""Original tiled NF4 decoder, including fused reconstruction of nested scales.

First candidate: patch functional dequantization only. Keep bnb's forward GEMM
dispatcher and PEFT/dropout unchanged. No shared output storage escapes this API.
Requires CUDA testing before submission; no performance claim is made here.
"""

from math import prod

import torch
import triton
import triton.language as tl


@triton.jit
def nf4_decode_kernel(Packed, Absmax, Code, ScaleCode, ScaleAbsmax, Offset,
                      Output, N, Q_BLOCK: tl.constexpr,
                      NESTED: tl.constexpr, SCALE_BLOCK: tl.constexpr,
                      TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < N
    byte = tl.load(Packed + i // 2, mask=valid, other=0).to(tl.int32)
    shift = tl.where(i % 2 == 0, 4, 0)
    index = (byte >> shift) & 15
    value = tl.load(Code + index)
    block = i // Q_BLOCK
    if NESTED:
        scale_index = tl.load(Absmax + block, mask=valid, other=0).to(tl.int32)
        scale_value = tl.load(ScaleCode + scale_index)
        outer = tl.load(ScaleAbsmax + block // SCALE_BLOCK, mask=valid, other=0)
        # bnb reconstructs FP32 scales, then adds offset in a separate operation.
        scale = scale_value * outer
        scale = scale + tl.load(Offset)
    else:
        scale = tl.load(Absmax + block, mask=valid, other=0)
    tl.store(Output + i, value * scale, mask=valid)


def supported(packed, state):
    if state is None or not isinstance(packed, torch.Tensor):
        return False
    if not packed.is_cuda or packed.dtype != torch.uint8 or not packed.is_contiguous():
        return False
    if getattr(state, "quant_type", None) != "nf4":
        return False
    if getattr(state, "packing_format_for_cpu", False):
        return False
    shape = getattr(state, "shape", None)
    if shape is None or len(shape) != 2 or prod(shape) == 0:
        return False
    if packed.numel() != (prod(shape) + 1) // 2:
        return False
    if state.dtype not in (torch.float16, torch.float32):
        return False
    if state.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        return False
    tensors = [state.absmax, state.code]
    if state.code.numel() != 16 or state.code.dtype != torch.float32:
        return False
    blocks = triton.cdiv(prod(shape), state.blocksize)
    if state.absmax.numel() != blocks:
        return False
    if state.nested:
        second = state.state2
        if second is None or second.nested or second.dtype != torch.float32:
            return False
        if second.blocksize not in (64, 128, 256, 512, 1024, 2048, 4096):
            return False
        if state.absmax.dtype != torch.uint8 or second.code.numel() != 256:
            return False
        if second.code.dtype != torch.float32 or second.absmax.dtype != torch.float32:
            return False
        if second.absmax.numel() != triton.cdiv(blocks, second.blocksize):
            return False
        if not isinstance(state.offset, torch.Tensor) or state.offset.numel() != 1:
            return False
        if state.offset.dtype != torch.float32:
            return False
        tensors += [second.code, second.absmax, state.offset]
    elif state.absmax.dtype != torch.float32:
        return False
    return all(t.device == packed.device and t.is_contiguous() for t in tensors)


def dequantize_nf4(packed, state, out=None):
    """Return canonical [out_features, in_features], irrespective of packed view.

    Explicit out is caller-owned. The result may be saved by autograd: do not pass
    one global buffer here unless its entire lifetime is controlled by the caller.
    """
    if not supported(packed, state):
        raise ValueError("unsupported NF4 state; use the original bitsandbytes path")
    if out is None:
        out = torch.empty(tuple(state.shape), dtype=state.dtype, device=packed.device)
    if (tuple(out.shape) != tuple(state.shape) or out.dtype != state.dtype
            or out.device != packed.device or not out.is_contiguous()):
        raise ValueError("out must match canonical shape, dtype and device")
    inputs = [packed, state.absmax, state.code]
    if state.nested:
        inputs += [state.state2.code, state.state2.absmax, state.offset]
    # Prevent accidental destruction of quantized weights or statistics via out=.
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in inputs):
        raise ValueError("out aliases quantized inputs")
    second = state.state2 if state.nested else None
    with torch.cuda.device(packed.device):
        nf4_decode_kernel[(triton.cdiv(prod(state.shape), 1024),)](
            packed, state.absmax, state.code,
            second.code if second else state.code,
            second.absmax if second else state.absmax,
            state.offset if second else state.absmax,
            out, prod(state.shape), state.blocksize, state.nested,
            second.blocksize if second else 1, 1024,
            num_warps=4, enable_fp_fusion=False,
        )
    return out


def patch_dequantize():
    """Idempotent, narrow public-API patch; unsupported calls use bnb unchanged."""
    import bitsandbytes.functional as functional

    original = functional.dequantize_4bit
    if getattr(original, "_soup_nf4_candidate", False):
        return

    def replacement(A, quant_state=None, absmax=None, out=None,
                    blocksize=None, quant_type="fp4"):
        if supported(A, quant_state):
            decoded = dequantize_nf4(A, quant_state, out=out)
            # Preserve bnb 0.50.2's legacy transposed packed-input convention.
            return decoded.t() if A.shape[0] == 1 else decoded
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    replacement._soup_nf4_candidate = True
    replacement._original = original
    functional.dequantize_4bit = replacement


def unpatch_dequantize():
    import bitsandbytes.functional as functional

    current = functional.dequantize_4bit
    if getattr(current, "_soup_nf4_candidate", False):
        functional.dequantize_4bit = current._original
