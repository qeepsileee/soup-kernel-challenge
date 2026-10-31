"""Runtime trial: native dropout RNG, packed-mask adjoint and shared LoRA inputs.

Forward uses aten.native_dropout, the same CUDA operation as nn.Dropout. Its
mask is packed, not regenerated. Backward reads its bits directly. Within one
Qwen attention forward, A-projection saved inputs can be reconstructed exactly
from a shared FP16 input snapshot and each branch's own mask. No model parameter
or dropout probability is changed. The public full-task checks remain required.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import struct
import weakref

import torch
import triton
import triton.language as tl
from .lossless_saved import PackedMask, pack_mask, unpack_saved


@triton.jit
def packed_dropout_scale_kernel(X, Mask, Out, N, Scale, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(Mask + i // 8, mask=i < N, other=0).to(tl.int32)
    keep = ((bits >> (i % 8)) & 1).to(tl.float32)
    x = tl.load(X + i, mask=i < N, other=0).to(tl.float32)
    # Match the native floating-point multiplications, including masked NaNs.
    tl.store(Out + i, (keep * x) * Scale, mask=i < N)


@triton.jit
def mask_pack_snapshot_kernel(X, Mask, Packed, Snapshot, N, BLOCK: tl.constexpr):
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 8)
    i = byte[:, None] * 8 + bit[None, :]
    keep = tl.load(Mask + i, mask=i < N, other=0).to(tl.int32)
    value = tl.load(X + i, mask=i < N, other=0)
    tl.store(Snapshot + i, value, mask=i < N)
    bits = tl.sum(keep << bit[None, :], 1)
    tl.store(Packed + byte, bits.to(tl.uint8), mask=byte < tl.cdiv(N, 8))


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _forward_scale(probability):
    # Native CUDA forward casts the KEEP probability to its accumulator dtype
    # before computing the reciprocal. Backward receives a double reciprocal.
    return _float32(1.0 / _float32(1.0 - probability))


def _scale_packed(value, packed, scale):
    value = value.contiguous()
    out = torch.empty_like(value)
    n = value.numel()
    if n:
        with torch.cuda.device(value.device):
            packed_dropout_scale_kernel[(triton.cdiv(n, 1024),)](
                value, packed, out, n, scale, 1024, num_warps=4,
                enable_fp_fusion=False)
    return out


class PackedNativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probability, snapshot, copy_source):
        out, mask = torch.ops.aten.native_dropout.default(x, probability, True)
        if copy_source:
            n = x.numel()
            # Own the new buffer inside forward; do not mutate a Function input.
            snapshot = torch.empty_like(x)
            packed = torch.empty(triton.cdiv(n, 8), device=x.device, dtype=torch.uint8)
            with torch.cuda.device(x.device):
                mask_pack_snapshot_kernel[(triton.cdiv(n, 8 * 256),)](
                    x, mask, packed, snapshot, n, 256, num_warps=4)
            saved = PackedMask(packed, tuple(x.shape))
        else:
            saved = pack_mask(mask)
        if not isinstance(saved, PackedMask):
            raise ValueError("packed dropout requires a contiguous CUDA mask of at least 4096 values")
        ctx.save_for_backward(saved.packed)
        ctx.shape = tuple(x.shape)
        ctx.backward_scale = 1.0 / (1.0 - probability)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(saved.packed, *([snapshot] if snapshot is not None else []))
        return out, saved.packed, snapshot

    @staticmethod
    def backward(ctx, grad, unused_packed_grad, unused_snapshot_grad):
        if grad is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        packed, = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Retain the native higher-order derivative path for create_graph.
            mask = unpack_saved(PackedMask(packed, ctx.shape))
            dx = torch.ops.aten.native_dropout_backward.default(grad, mask, ctx.backward_scale)
        else:
            dx = _scale_packed(grad, packed, ctx.backward_scale)
        return dx, None, None, None


@dataclass
class SavedDropoutInput:
    source: torch.Tensor
    packed: torch.Tensor
    scale: float
    shape: tuple
    stride: tuple
    offset: int
    output_ref: object
    version: int


class DropoutInputContext:
    def __init__(self):
        self.sources = {}
        self.outputs = {}
        self.source_count = 0
        self.packed_count = 0

    def prepare_source(self, source, *, allocate_only=False):
        if source.dtype != torch.float16 or not source.is_contiguous() or source.numel() < 4096:
            return None, False
        key = (id(source), source._version)
        entry = self.sources.get(key)
        is_new = entry is None or entry[0]() is not source
        if is_new:
            if allocate_only:
                return None, True  # The fused forward will allocate/fill it.
            with torch.no_grad():
                snapshot = source.detach().clone(memory_format=torch.contiguous_format)
            entry = (weakref.ref(source), snapshot)
            self.sources[key] = entry
            self.source_count += 1
        return entry[1], is_new

    def register(self, source, output, packed, probability, snapshot=None):
        if (not output.is_contiguous() or source.shape != output.shape):
            return
        if snapshot is None:
            snapshot, _ = self.prepare_source(source)
        else:
            key = (id(source), source._version)
            entry = self.sources.get(key)
            if entry is None or entry[0]() is not source:
                self.sources[key] = (weakref.ref(source), snapshot)
                self.source_count += 1
            else:
                snapshot = entry[1]
        if snapshot is None:
            return
        self.outputs[output.untyped_storage().data_ptr()] = (
            weakref.ref(output), output._version, snapshot, packed,
            _forward_scale(probability), output.numel())

    def pack(self, tensor):
        if not tensor.is_cuda or tensor.dtype != torch.float16 or not tensor.numel():
            return tensor.detach()
        entry = self.outputs.get(tensor.untyped_storage().data_ptr())
        if entry is None:
            return tensor.detach()
        ref, version, source, packed, scale, size = entry
        original = ref()
        if original is None or original._version != version:
            return tensor.detach()  # No stale pointer reuse or changed forward values.
        stride, offset = tuple(tensor.stride()), tensor.storage_offset()
        last = offset + sum((s - 1) * t for s, t in zip(tensor.shape, stride))
        if offset < 0 or any(t < 0 for t in stride) or last >= size:
            return tensor.detach()
        self.packed_count += 1
        return SavedDropoutInput(source, packed, scale, tuple(tensor.shape),
                                 stride, offset, ref, version)

    def clear(self):
        self.sources.clear()
        self.outputs.clear()


def unpack_dropout_input(value):
    if not isinstance(value, SavedDropoutInput):
        return value
    original = value.output_ref()
    if original is not None and original._version != value.version:
        raise RuntimeError("a saved LoRA dropout output was modified in place")
    with torch.no_grad():
        restored = _scale_packed(value.source, value.packed, value.scale)
        return restored.as_strided(value.shape, value.stride, value.offset)


_input_context = ContextVar("runtime_trial_dropout_inputs", default=None)


@contextmanager
def shared_dropout_inputs():
    context = DropoutInputContext()
    token = _input_context.set(context)
    try:
        with torch.autograd.graph.saved_tensors_hooks(context.pack, unpack_dropout_input):
            yield context
    finally:
        context.clear()
        _input_context.reset(token)


def patch_fast_lora_dropouts(model):
    count = 0
    for name, module in model.named_modules():
        if (not isinstance(module, torch.nn.Dropout)
                or not (name.startswith("lora_dropout.") or ".lora_dropout." in name)
                or getattr(module.forward, "_packed_native_dropout", False)):
            continue

        def build(dropout):
            original = dropout.forward

            def forward(*args, **kwargs):
                x = args[0] if args else kwargs.get("input")
                if (len(args) > 1 or set(kwargs) - {"input"}
                        or not torch.is_grad_enabled() or not dropout.training
                        or dropout.inplace or not 0 < dropout.p < 1
                        or not isinstance(x, torch.Tensor) or not x.is_cuda
                        or x.dtype not in (torch.float16, torch.float32)
                        or not x.is_contiguous() or x.numel() < 4096):
                    return original(*args, **kwargs)
                context = _input_context.get()
                snapshot, copy_source = (context.prepare_source(x, allocate_only=True)
                                         if context is not None else (None, False))
                out, packed, snapshot = PackedNativeDropout.apply(x, dropout.p, snapshot, copy_source)
                if context is not None:
                    context.register(x, out, packed, dropout.p, snapshot=snapshot)
                return out

            forward._packed_native_dropout = True
            forward._original = original
            return forward

        module.forward = build(module)
        count += 1
    return count


def unpatch_fast_lora_dropouts(model):
    for module in model.modules():
        if getattr(module.forward, "_packed_native_dropout", False):
            module.forward = module.forward._original


def patch_shared_lora_inputs():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    original = Qwen2Attention.forward
    if getattr(original, "_shared_lora_saved_inputs", False):
        return

    def forward(self, *args, **kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        if (not self.training or not torch.is_grad_enabled()
                or not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.dtype != torch.float16):
            return original(self, *args, **kwargs)
        with shared_dropout_inputs():
            return original(self, *args, **kwargs)

    forward._shared_lora_saved_inputs = True
    forward._original = original
    Qwen2Attention.forward = forward


def unpatch_shared_lora_inputs():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    current = Qwen2Attention.forward
    if getattr(current, "_shared_lora_saved_inputs", False):
        Qwen2Attention.forward = current._original


def patch_fast_dropout():
    """Runtime-trial defaults; call after the lossless saved-data setup patch."""
    from soup_cli.trainer.sft import SFTTrainerWrapper
    patch_shared_lora_inputs()
    original = SFTTrainerWrapper._setup_transformers
    if getattr(original, "_packed_native_dropout", False):
        return

    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            print("[fast_dropout] patched LoRA modules:", patch_fast_lora_dropouts(model), flush=True)
        return result

    setup._packed_native_dropout = True
    setup._original = original
    SFTTrainerWrapper._setup_transformers = setup
