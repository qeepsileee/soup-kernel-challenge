import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(
    R, X, W, S, Y, RS,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)

    r = tl.load(R + row * D + j, j < D, 0)
    x = tl.load(X + row * D + j, j < D, 0)

    summed = (r + x).to(R.dtype.element_ty)
    tl.store(S + row * D + j, summed, j < D)

    z = summed.to(tl.float32)
    variance = tl.sum(z * z, 0) / D
    rs = tl.rsqrt(variance + EPS)

    tl.store(RS + row, rs)

    w = tl.load(W + j, j < D, 0).to(tl.float32)
    normalized = (z * rs).to(R.dtype.element_ty).to(tl.float32)

    tl.store(Y + row * D + j, normalized * w, j < D)


@triton.jit
def fused_add_rmsnorm_backward_kernel(
    G, GS, W, S, RS, DR, DX,
    D: tl.constexpr,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)

    z = tl.load(S + row * D + j, j < D, 0).to(tl.float32)
    w = tl.load(W + j, j < D, 0).to(tl.float32)
    g = tl.load(G + row * D + j, j < D, 0).to(tl.float32)

    gs = tl.load(GS + row * D + j, j < D, 0).to(tl.float32)

    rs = tl.load(RS + row)

    g = (g * w).to(S.dtype.element_ty).to(tl.float32)
    correction = tl.sum(g * z, 0) / D

    grad = gs + rs * g - z * (rs * rs * rs * correction)

    tl.store(DR + row * D + j, grad, j < D)
    tl.store(DX + row * D + j, grad, j < D)


class FusedAddRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, residual, x, weight, eps):
        residual = residual.contiguous()
        x = x.contiguous()
        weight = weight.contiguous()

        d = x.shape[-1]
        rows = x.numel() // d

        summed = torch.empty_like(residual)
        output = torch.empty(
            x.shape,
            device=x.device,
            dtype=x.dtype
        )
        rs = torch.empty(rows, device=x.device, dtype=torch.float32)

        fused_add_rmsnorm_kernel[(rows,)](
            residual, x, weight, summed, output, rs,
            d, eps, triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False
        )

        ctx.save_for_backward(summed, weight, rs)
        ctx.d = d

        return summed, output

    @staticmethod
    def backward(ctx, grad_sum, grad_output):
        summed, weight, rs = ctx.saved_tensors

        if grad_sum is None:
            grad_sum = torch.zeros_like(summed)

        if grad_output is None:
            grad_output = torch.zeros_like(summed, dtype=summed.dtype)

        d = ctx.d
        rows = rs.numel()

        grad_residual = torch.empty_like(summed)
        grad_x = torch.empty_like(summed)

        fused_add_rmsnorm_backward_kernel[(rows,)](
            grad_output.contiguous(),
            grad_sum.contiguous(),
            weight,
            summed,
            rs,
            grad_residual,
            grad_x,
            d,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False
        )

        return grad_residual, grad_x, None, None


def fused_add_rmsnorm(residual, x, weight, eps):
    return FusedAddRMSNorm.apply(residual, x, weight, eps)



def patch_fused_add_rmsnorm(enabled=True):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    current = Qwen2DecoderLayer.forward

    if not enabled:
        if getattr(current, "_own_fused_add_rmsnorm", False):
            Qwen2DecoderLayer.forward = current._original
        return

    if getattr(current, "_own_fused_add_rmsnorm", False):
        return

    original = current

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states, normed = fused_add_rmsnorm(
            residual,
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )

        residual = hidden_states
        hidden_states = self.mlp(normed)
        hidden_states = residual + hidden_states

        return hidden_states

    forward._own_fused_add_rmsnorm = True
    forward._original = original
    Qwen2DecoderLayer.forward = forward