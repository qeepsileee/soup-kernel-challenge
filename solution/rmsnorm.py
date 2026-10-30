"""Frozen-weight Qwen RMSNorm, preserving normalized-value rounding and output dtype."""
import torch
import triton
import triton.language as tl


@triton.jit
def rms_forward_kernel(X, W, Y, R, D: tl.constexpr, EPS: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, T)
    x = tl.load(X + row * D + j, mask=j < D, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(R + row, r)
    w = tl.load(W + j, mask=j < D, other=0).to(tl.float32)
    z = (x * r).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * D + j, z * w, mask=j < D)


@triton.jit
def rms_backward_kernel(G, X, W, R, DX, D: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, T)
    x = tl.load(X + row * D + j, mask=j < D, other=0).to(tl.float32)
    w = tl.load(W + j, mask=j < D, other=0).to(tl.float32)
    g = tl.load(G + row * D + j, mask=j < D, other=0).to(tl.float32)
    g = (g * w).to(X.dtype.element_ty).to(tl.float32)
    r = tl.load(R + row)
    correction = tl.sum(g * x, 0) / D
    dx = r * g - x * (r * r * r * correction)
    tl.store(DX + row * D + j, dx, mask=j < D)


class FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        d = x.shape[-1]
        rows = x.numel() // d
        y = torch.empty(x.shape, device=x.device, dtype=torch.promote_types(x.dtype, weight.dtype))
        r = torch.empty(rows, device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            rms_forward_kernel[(rows,)](x, weight, y, r, d, eps,
                triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, r)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, w, r = ctx.saved_tensors
        dx = torch.empty_like(x)
        with torch.cuda.device(x.device):
            rms_backward_kernel[(r.numel(),)](grad.contiguous(), x, w, r, dx,
                x.shape[-1], triton.next_power_of_2(x.shape[-1]),
                num_warps=4, enable_fp_fusion=False)
        return dx, None, None


def patch_rmsnorm():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    original = Qwen2RMSNorm.forward
    if getattr(original, "_own_rmsnorm", False):
        return

    def forward(self, x):
        w = self.weight
        if (x.is_cuda and x.numel() > 0 and x.dtype in (torch.float16, torch.float32)
                and w.device == x.device and w.dtype in (torch.float16, torch.float32)
                and w.is_contiguous() and not w.requires_grad and x.shape[-1] <= 4096):
            return FrozenRMSNorm.apply(x, w, self.variance_epsilon)
        return original(self, x)

    forward._own_rmsnorm = True
    forward._original = original
    Qwen2RMSNorm.forward = forward
