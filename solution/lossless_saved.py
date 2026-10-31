"""Standalone exact saved-data codecs. NOT enabled by solution.apply().

Pack LoRA dropout bool masks into bits. Retain original compact K/V rather than
their repeated head copies saved by native SDPA, reconstructing identical views
for native backward. Original forward operations, backend and RNG are unchanged.
"""
from contextvars import ContextVar
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@triton.jit
def mask_pack_kernel(X, Packed, N, BLOCK: tl.constexpr):
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 8)
    index = byte[:, None] * 8 + bit[None, :]
    value = tl.load(X + index, mask=index < N, other=0).to(tl.int32)
    packed = tl.sum(value << bit[None, :], 1)
    tl.store(Packed + byte, packed.to(tl.uint8), mask=byte < tl.cdiv(N, 8))


@triton.jit
def mask_unpack_kernel(Packed, X, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(Packed + index // 8, mask=index < N, other=0).to(tl.int32)
    tl.store(X + index, ((byte >> (index % 8)) & 1).to(tl.int1), mask=index < N)


@dataclass
class PackedMask:
    packed: torch.Tensor
    shape: tuple


@dataclass
class SavedKV:
    compact: torch.Tensor
    groups: int
    shape: tuple
    stride: tuple
    offset: int


def pack_mask(tensor):
    if (not tensor.is_cuda or tensor.dtype != torch.bool or not tensor.is_contiguous()
            or tensor.numel() < 4096):
        return tensor
    n = tensor.numel()
    packed = torch.empty(triton.cdiv(n, 8), device=tensor.device, dtype=torch.uint8)
    with torch.cuda.device(tensor.device):
        mask_pack_kernel[(triton.cdiv(n, 8 * 256),)](tensor, packed, n, 256, num_warps=4)
    return PackedMask(packed, tuple(tensor.shape))


def unpack_saved(value):
    with torch.no_grad():
        if isinstance(value, PackedMask):
            out = torch.empty(value.shape, device=value.packed.device, dtype=torch.bool)
            n = out.numel()
            if n:
                with torch.cuda.device(out.device):
                    mask_unpack_kernel[(triton.cdiv(n, 1024),)](value.packed, out, n, 1024, num_warps=4)
            return out
        if isinstance(value, SavedKV):
            expanded = value.compact.repeat_interleave(value.groups, dim=1).contiguous()
            return expanded.as_strided(value.shape, value.stride, value.offset)
    return value


class KVContext:
    def __init__(self):
        self.sources = {}

    def register(self, source, expanded, groups):
        if (groups <= 1 or not expanded.is_cuda or expanded.dtype != torch.float16
                or expanded.ndim != 4 or not expanded.is_contiguous()
                or expanded.storage_offset() != 0
                or expanded.untyped_storage().nbytes() != expanded.numel() * expanded.element_size()
                or source.untyped_storage().nbytes() >= expanded.untyped_storage().nbytes()):
            return
        # Keep no references to the expanded allocation in this descriptor.
        self.sources[expanded.untyped_storage().data_ptr()] = (
            source.detach().clone(memory_format=torch.contiguous_format), int(groups), expanded.numel())

    def pack(self, tensor):
        if not tensor.is_cuda or tensor.dtype != torch.float16 or not tensor.numel():
            return tensor
        descriptor = self.sources.get(tensor.untyped_storage().data_ptr())
        if descriptor is None:
            return tensor
        compact, groups, size = descriptor
        offset = tensor.storage_offset()
        stride = tuple(tensor.stride())
        last = offset + sum((s-1)*t for s, t in zip(tensor.shape, stride))
        if offset < 0 or any(t < 0 for t in stride) or last >= size:
            return tensor
        return SavedKV(compact, groups, tuple(tensor.shape), stride, offset)


_kv_context = ContextVar("lossless_saved_kv", default=None)


def patch_lossless_attention():
    from transformers.integrations import sdpa_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "_lossless_saved_kv", False):
        return
    original_repeat = sdpa_attention.repeat_kv

    def repeat(tensor, groups):
        result = original_repeat(tensor, groups)
        context = _kv_context.get()
        if context is not None:
            context.register(tensor, result, groups)
        return result

    def forward(module, query, key, value, *args, **kwargs):
        if (not torch.is_grad_enabled() or not getattr(module, "training", False) or not query.is_cuda
                or query.dtype != torch.float16
                or getattr(getattr(module, "config", None), "model_type", None) != "qwen2"
                or getattr(module, "num_key_value_groups", 1) <= 1):
            return original(module, query, key, value, *args, **kwargs)
        context = KVContext()
        token = _kv_context.set(context)
        try:
            with torch.autograd.graph.saved_tensors_hooks(context.pack, unpack_saved):
                return original(module, query, key, value, *args, **kwargs)
        finally:
            # PyTorch may retain hook callables. Empty this map so the pack
            # callback cannot keep unused tensors alive after the forward.
            context.sources.clear()
            _kv_context.reset(token)

    repeat._lossless_saved_kv = True
    repeat._original = original_repeat
    forward._lossless_saved_kv = True
    forward._original = original
    sdpa_attention.repeat_kv = repeat
    ALL_ATTENTION_FUNCTIONS.register("sdpa", forward)


def unpatch_lossless_attention():
    from transformers.integrations import sdpa_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(current, "_lossless_saved_kv", False):
        ALL_ATTENTION_FUNCTIONS.register("sdpa", current._original)
    current_repeat = sdpa_attention.repeat_kv
    if getattr(current_repeat, "_lossless_saved_kv", False):
        sdpa_attention.repeat_kv = current_repeat._original


def patch_lora_dropouts(model):
    count = 0
    for name, module in model.named_modules():
        if (not isinstance(module, torch.nn.Dropout)
                or not (name.startswith("lora_dropout.") or ".lora_dropout." in name)
                or getattr(module.forward, "_lossless_saved_mask", False)):
            continue

        def build(dropout):
            original = dropout.forward
            def forward(*args, **kwargs):
                x = args[0] if args else kwargs.get("input")
                if (not torch.is_grad_enabled() or not dropout.training
                        or not isinstance(x, torch.Tensor) or not x.is_cuda):
                    return original(*args, **kwargs)
                with torch.autograd.graph.saved_tensors_hooks(pack_mask, unpack_saved):
                    return original(*args, **kwargs)
            forward._lossless_saved_mask = True
            forward._original = original
            return forward

        module.forward = build(module)
        count += 1
    return count


def unpatch_lora_dropouts(model):
    for module in model.modules():
        if getattr(module.forward, "_lossless_saved_mask", False):
            module.forward = module.forward._original


def patch_lossless_saved():
    """Explicit diagnostic opt-in, not called by production apply()."""
    patch_lossless_attention()
    from soup_cli.trainer.sft import SFTTrainerWrapper
    original = SFTTrainerWrapper._setup_transformers
    if getattr(original, "_lossless_saved_mask", False):
        return

    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            print("[lossless_saved] LoRA dropout modules:", patch_lora_dropouts(model), flush=True)
        return result

    setup._lossless_saved_mask = True
    setup._original = original
    SFTTrainerWrapper._setup_transformers = setup
