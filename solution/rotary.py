"""Own RoPE fusion for the fixed Qwen2 FP16 training path.

Preserve separate product rounding before the addition in forward and backward.
Cos/sin are supplied by Qwen unchanged. No q/k activations are saved for backward.
Unsupported shapes/dtypes or differentiable position embeddings use HF unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_forward_kernel(X, C, S, Y, T, H, B,
                          XS0, XS1, XS2,
                          YS0, YS1, YS2,
                          CS0, CS1, SS0, SS1, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    b, t = row // (H * T), row % T
    h = row // T % H
    valid = row < B * H * T
    base = b * XS0 + h * XS1 + t * XS2
    x = tl.load(X + base[:, None] + d[None, :], mask=valid[:, None], other=0).to(tl.float32)
    partner = (d + 64) % 128
    r = tl.load(X + base[:, None] + partner[None, :], mask=valid[:, None], other=0).to(tl.float32)
    r = tl.where(d[None, :] < 64, -r, r)
    c = tl.load(C + b[:, None] * CS0 + t[:, None] * CS1 + d[None, :],
                mask=valid[:, None], other=0).to(tl.float32)
    s = tl.load(S + b[:, None] * SS0 + t[:, None] * SS1 + d[None, :],
                mask=valid[:, None], other=0).to(tl.float32)
    a = (x * c).to(Y.dtype.element_ty).to(tl.float32)
    z = (r * s).to(Y.dtype.element_ty).to(tl.float32)
    out_base = b * YS0 + h * YS1 + t * YS2
    tl.store(Y + out_base[:, None] + d[None, :], a + z, mask=valid[:, None])


@triton.jit
def rotary_backward_kernel(G, C, S, DX, T, H, B,
                           GS0, GS1, GS2, DS0, DS1, DS2,
                           CS0, CS1, SS0, SS1, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    b, t = row // (H * T), row % T
    h = row // T % H
    valid = row < B * H * T
    base = b * GS0 + h * GS1 + t * GS2
    partner = (d + 64) % 128
    g = tl.load(G + base[:, None] + d[None, :], mask=valid[:, None], other=0).to(tl.float32)
    gp = tl.load(G + base[:, None] + partner[None, :], mask=valid[:, None], other=0).to(tl.float32)
    c = tl.load(C + b[:, None] * CS0 + t[:, None] * CS1 + d[None, :],
                mask=valid[:, None], other=0).to(tl.float32)
    sp = tl.load(S + b[:, None] * SS0 + t[:, None] * SS1 + partner[None, :],
                 mask=valid[:, None], other=0).to(tl.float32)
    direct = (g * c).to(DX.dtype.element_ty).to(tl.float32)
    rotated = (gp * sp).to(DX.dtype.element_ty).to(tl.float32)
    rotated = tl.where(d[None, :] < 64, rotated, -rotated)
    out_base = b * DS0 + h * DS1 + t * DS2
    tl.store(DX + out_base[:, None] + d[None, :], direct + rotated, mask=valid[:, None])


def supported(q, k, cos, sin, unsqueeze_dim=1):
    if (unsqueeze_dim != 1 or any(not x.is_cuda for x in (q, k, cos, sin))
            or q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3
            or q.dtype not in (torch.float16, torch.float32)
            or any(x.dtype != q.dtype or x.device != q.device for x in (k, cos, sin))
            or cos.requires_grad or sin.requires_grad or q.shape[-1] != 128
            or k.shape[-1] != 128 or q.shape[0] != k.shape[0]
            or q.shape[2] != k.shape[2] or not q.numel() or not k.numel()
            or any(x.stride(-1) != 1 for x in (q, k, cos, sin))):
        return False
    # Qwen2Model's default position_ids are [1,T], shared across the batch.
    # Both [1,T,D] and per-example [B,T,D] embeddings are valid broadcasts.
    return all(value.shape[0] in (1, q.shape[0])
               and tuple(value.shape[1:]) == (q.shape[2], 128) for value in (cos, sin))


def _position_strides(value):
    return (0 if value.shape[0] == 1 else value.stride(0)), value.stride(1)


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        outputs = []
        for x in (q, k):
            y = torch.empty_like(x)
            b, h, t, _ = x.shape
            with torch.cuda.device(x.device):
                rotary_forward_kernel[(triton.cdiv(b * h * t, 4),)](
                    x, cos, sin, y, t, h, b, *x.stride()[:3], *y.stride()[:3],
                    *_position_strides(cos), *_position_strides(sin), 4, num_warps=4,
                    enable_fp_fusion=False)
            outputs.append(y)
        ctx.save_for_backward(cos, sin)
        ctx.layouts = [(tuple(x.shape), tuple(y.stride())) for x, y in zip((q, k), outputs)]
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*[y for i, y in enumerate(outputs) if not ctx.needs_input_grad[i]])
        return tuple(outputs)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        grads = []
        for index, grad in enumerate((dq, dk)):
            if grad is None or not ctx.needs_input_grad[index]:
                grads.append(None)
                continue
            shape, stride = ctx.layouts[index]
            if torch.is_grad_enabled():
                direct = grad * cos.unsqueeze(1)
                rotated = grad * sin.unsqueeze(1)
                grads.append(direct + torch.cat((rotated[..., 64:], -rotated[..., :64]), dim=-1))
                continue
            if grad.stride(-1) != 1:
                grad = grad.contiguous()
            dx = torch.empty_strided(shape, stride, device=grad.device, dtype=grad.dtype)
            b, h, t, _ = shape
            with torch.cuda.device(grad.device):
                rotary_backward_kernel[(triton.cdiv(b * h * t, 4),)](
                    grad, cos, sin, dx, t, h, b, *grad.stride()[:3], *dx.stride()[:3],
                    *_position_strides(cos), *_position_strides(sin), 4, num_warps=4,
                    enable_fp_fusion=False)
            grads.append(dx)
        return *grads, None, None


def patch_rotary():
    from transformers.models.qwen2 import modeling_qwen2
    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_runtime_rotary", False):
        return

    def forward(q, k, cos, sin, unsqueeze_dim=1):
        if supported(q, k, cos, sin, unsqueeze_dim):
            return Rotary.apply(q, k, cos, sin)
        return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    forward._runtime_rotary = True
    forward._original = original
    modeling_qwen2.apply_rotary_pos_emb = forward


def unpatch_rotary():
    from transformers.models.qwen2 import modeling_qwen2
    current = modeling_qwen2.apply_rotary_pos_emb
    if getattr(current, "_runtime_rotary", False):
        modeling_qwen2.apply_rotary_pos_emb = current._original
