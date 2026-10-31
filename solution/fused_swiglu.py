import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def fused_swiglu_forward(G, U, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    gate = tl.load(G + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(U + offsets, mask=mask, other=0).to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    result = gate * sigmoid * up

    tl.store(Y + offsets, result, mask=mask)


@triton.jit
def fused_swiglu_backward(DY, G, U, DG, DU, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    grad = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
    gate = tl.load(G + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(U + offsets, mask=mask, other=0).to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate))

    dgate = grad * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    dup = grad * gate * sigmoid

    tl.store(DG + offsets, dgate, mask=mask)
    tl.store(DU + offsets, dup, mask=mask)


class FusedSwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()

        output = torch.empty_like(gate)
        n = gate.numel()

        if n > 0:
            fused_swiglu_forward[(triton.cdiv(n, BLOCK),)](
                gate, up, output, n, BLOCK
            )

        ctx.save_for_backward(gate, up)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)

        n = gate.numel()

        if n > 0:
            fused_swiglu_backward[(triton.cdiv(n, BLOCK),)](
                grad_output, gate, up, dgate, dup, n, BLOCK
            )

        return dgate, dup


def fused_swiglu(gate, up):
    return FusedSwiGLU.apply(gate, up)