"""Two-stage stable entropy: O(tokens * ceil(vocab/4096)) scratch, no softmax tensor.

Math only: local max, exp sum and shifted first moment; merge local distributions.
The training metric uses every original token and the original TRL attention mask.
"""
from math import prod
import torch
import triton
import triton.language as tl


@triton.jit
def entropy_tiles_kernel(X, Stats, V: tl.constexpr, C: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    j = chunk * T + tl.arange(0, T)
    z = tl.load(X + row * V + j, mask=j < V, other=-float('inf')).to(tl.float32)
    m = tl.max(z, 0)
    shifted = z - m
    p = tl.exp(shifted)
    s = tl.sum(p, 0)
    moment = tl.sum(p * tl.where(j < V, shifted, 0), 0)
    ptr = Stats + (row * C + chunk) * 3
    tl.store(ptr, m)
    tl.store(ptr + 1, s)
    tl.store(ptr + 2, moment)


@triton.jit
def entropy_merge_kernel(Stats, Out, C: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, T)
    ptr = Stats + (row * C + c) * 3
    m = tl.load(ptr, mask=c < C, other=-float('inf'))
    s = tl.load(ptr + 1, mask=c < C, other=0)
    t = tl.load(ptr + 2, mask=c < C, other=0)
    maximum = tl.max(m, 0)
    delta = tl.where(c < C, m - maximum, 0)
    factor = tl.where(c < C, tl.exp(delta), 0)
    total = tl.sum(factor * s, 0)
    moment = tl.sum(factor * (t + s * delta), 0)
    tl.store(Out + row, tl.log(total) - moment / total)


def entropy_from_logits(logits, chunk_size=128):
    """Forward-only metric. Callers needing derivatives use original TRL via patch."""
    if not logits.is_cuda or logits.dtype != torch.float32 or not logits.is_contiguous():
        raise ValueError("entropy candidate requires contiguous FP32 CUDA logits")
    v = logits.shape[-1]
    rows = prod(logits.shape[:-1])
    chunks = triton.cdiv(v, 4096)
    out = torch.empty(logits.shape[:-1], device=logits.device, dtype=logits.dtype)
    if rows == 0:
        return out
    scratch = torch.empty((rows, chunks, 3), device=logits.device, dtype=torch.float32)
    with torch.cuda.device(logits.device):
        entropy_tiles_kernel[(rows, chunks)](logits, scratch, v, chunks, 4096, num_warps=4)
        entropy_merge_kernel[(rows,)](scratch, out, chunks,
                                     triton.next_power_of_2(chunks), num_warps=4)
    return out


def patch_entropy():
    from trl.trainer import sft_trainer
    original = sft_trainer.entropy_from_logits
    if getattr(original, "_own_entropy", False):
        return

    def entropy(logits, chunk_size=128):
        if (logits.is_cuda and logits.dtype == torch.float32 and logits.is_contiguous()
                and logits.shape[-1] > 0
                and not (torch.is_grad_enabled() and logits.requires_grad)):
            return entropy_from_logits(logits, chunk_size)
        return original(logits, chunk_size)

    entropy._own_entropy = True
    entropy._original = original
    sft_trainer.entropy_from_logits = entropy
