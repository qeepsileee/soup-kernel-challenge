"""PEFT 0.20.0 vanilla NF4 LoRA epilogue; GEMMs/dropout remain in PEFT/PyTorch.

No in-place writes to base output. Forward rounds scaled adapter values BEFORE
addition, exactly as separate torch mul/cast/add would. No tensors saved by the
epilogue: its derivatives depend only on the scalar scale and dtypes.
"""
from importlib.metadata import version
import torch
import triton
import triton.language as tl


@triton.jit
def lora_add_kernel(Base, Adapter, Out, N: tl.constexpr, SCALE: tl.constexpr,
                    CAST_TO_BASE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(Base + i, mask=i < N, other=0).to(tl.float32)
    a = tl.load(Adapter + i, mask=i < N, other=0).to(tl.float32)
    scaled = (a * SCALE).to(Adapter.dtype.element_ty)
    if CAST_TO_BASE:
        scaled = scaled.to(Base.dtype.element_ty)
    tl.store(Out + i, b + scaled.to(tl.float32), mask=i < N)


@triton.jit
def lora_adapter_grad_kernel(Grad, DAdapter, N: tl.constexpr,
                             SCALE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # AddBackward/ToCopyBackward cast to adapter dtype before MulBackward.
    g = tl.load(Grad + i, mask=i < N, other=0).to(DAdapter.dtype.element_ty).to(tl.float32)
    tl.store(DAdapter + i, g * SCALE, mask=i < N)


class LoRAAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, adapter, scale, cast_to_base):
        base, adapter = base.contiguous(), adapter.contiguous()
        ctx.base_dtype, ctx.adapter_dtype = base.dtype, adapter.dtype
        ctx.scale = float(scale)
        dtype = base.dtype if cast_to_base else torch.promote_types(base.dtype, adapter.dtype)
        out = torch.empty(base.shape, device=base.device, dtype=dtype)
        with torch.cuda.device(base.device):
            lora_add_kernel[(triton.cdiv(base.numel(), 1024),)](
                base, adapter, out, base.numel(), ctx.scale, bool(cast_to_base),
                1024, num_warps=4, enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, grad):
        dbase = grad.to(ctx.base_dtype) if ctx.needs_input_grad[0] else None
        dadapter = None
        if ctx.needs_input_grad[1]:
            grad = grad.contiguous()
            dadapter = torch.empty(grad.shape, device=grad.device, dtype=ctx.adapter_dtype)
            with torch.cuda.device(grad.device):
                lora_adapter_grad_kernel[(triton.cdiv(grad.numel(), 1024),)](
                    grad, dadapter, grad.numel(), ctx.scale, 1024, num_warps=4,
                    enable_fp_fusion=False)
        return dbase, dadapter, None, None


def patch_lora():
    if version("peft") != "0.20.0":
        return
    from peft.tuners.lora.bnb import Linear4bit
    original = Linear4bit.forward
    if getattr(original, "_own_lora_epilogue", False):
        return

    def forward(self, x, *args, **kwargs):
        # Narrow, one-adapter rank-16 path; preserve all PEFT variants via fallback.
        if (args or kwargs or not x.is_cuda or x.numel() == 0
                or x.dtype not in (torch.float16, torch.float32)
                or self.disable_adapters or self.merged or len(self.active_adapters) != 1):
            return original(self, x, *args, **kwargs)
        key = self.active_adapters[0]
        if (key not in self.lora_A or key not in self.lora_B
                or key in self.lora_variant or self.r[key] != 16):
            return original(self, x, *args, **kwargs)
        a, b = self.lora_A[key], self.lora_B[key]
        if (a.weight.device != x.device or b.weight.device != x.device
                or a.weight.dtype not in (torch.float16, torch.float32)
                or b.weight.dtype not in (torch.float16, torch.float32)):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        base = self.base_layer(x)
        convert = not torch.is_autocast_enabled()
        branch_x = self._cast_input_dtype(x, a.weight.dtype) if convert else x
        # Keep the exact dropout module/call order/RNG consumption and both GEMMs.
        adapter = b(a(self.lora_dropout[key](branch_x)))
        return LoRAAdd.apply(base, adapter, self.scaling[key], convert)

    forward._own_lora_epilogue = True
    forward._original = original
    Linear4bit.forward = forward


def unpatch_lora():
    from peft.tuners.lora.bnb import Linear4bit
    current = Linear4bit.forward
    if getattr(current, "_own_lora_epilogue", False):
        Linear4bit.forward = current._original
