"""Independent tiled CE/entropy/accuracy candidate; not enabled in apply().

Pass 1 parallelizes rows AND vocab tiles, pass 2 merges stable statistics.
Optional pass 3 writes gradients after every statistic has been finalized.
No third-party kernel code; standard stable reduction identities only.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def ce_tile_stats_kernel(Z, Stats, V: tl.constexpr, C: tl.constexpr,
                         S0: tl.constexpr, S1: tl.constexpr, T: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * T + tl.arange(0, T)
    valid = col < V
    z = tl.load(Z + row * S0 + col * S1, mask=valid, other=-float('inf')).to(tl.float32)
    maximum = tl.max(z, 0)
    shifted = z - maximum
    p = tl.exp(shifted)
    total = tl.sum(p, 0)
    moment = tl.sum(p * tl.where(valid, shifted, 0.0), 0)
    # PyTorch argmax returns the first matching index, including ties across tiles.
    first = tl.min(tl.where(valid & (z == maximum), col, 2147483647), 0)
    ptr = Stats + (row * C + chunk) * 4
    tl.store(ptr, maximum)
    tl.store(ptr + 1, total)
    tl.store(ptr + 2, moment)
    tl.store(ptr + 3, first.to(tl.float32))


@triton.jit
def ce_merge_stats_kernel(Z, Target, Stats, Loss, Entropy, Hits, LSE, Max, LogSum,
                          V: tl.constexpr, C: tl.constexpr,
                          S0: tl.constexpr, S1: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.arange(0, T)
    ptr = Stats + (row * C + chunk) * 4
    maximum = tl.load(ptr, mask=chunk < C, other=-float('inf'))
    total = tl.load(ptr + 1, mask=chunk < C, other=0)
    moment = tl.load(ptr + 2, mask=chunk < C, other=0)
    first = tl.load(ptr + 3, mask=chunk < C, other=16777215.).to(tl.int32)
    overall = tl.max(maximum, 0)
    delta = tl.where(chunk < C, maximum - overall, 0.0)
    factor = tl.where(chunk < C, tl.exp(delta), 0.0)
    summed = tl.sum(factor * total, 0)
    shifted_moment = tl.sum(factor * (moment + total * delta), 0)
    log_total = tl.log(summed)
    lse = overall + log_total
    predicted = tl.min(tl.where((chunk < C) & (maximum == overall), first, 2147483647), 0)
    target = tl.load(Target + row)
    labelled = (target >= 0) & (target < V)
    target_value = tl.load(Z + row * S0 + target * S1, mask=labelled, other=0).to(tl.float32)
    tl.store(Loss + row, tl.where(labelled, (overall - target_value) + log_total, 0.0))
    tl.store(Entropy + row, log_total - shifted_moment / summed)
    tl.store(Hits + row, (labelled & (predicted == target)).to(tl.int32))
    tl.store(LSE + row, lse)
    tl.store(Max + row, overall)
    tl.store(LogSum + row, log_total)


@triton.jit
def ce_write_grad_kernel(Z, Target, Max, LogSum, Grad, Upstream,
                         V: tl.constexpr, Z0: tl.constexpr, Z1: tl.constexpr,
                         G0: tl.constexpr, G1: tl.constexpr,
                         DENOM: tl.constexpr, SCALED: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * T + tl.arange(0, T)
    z = tl.load(Z + row * Z0 + col * Z1, mask=col < V, other=-float('inf')).to(tl.float32)
    maximum = tl.load(Max + row)
    log_sum = tl.load(LogSum + row)
    target = tl.load(Target + row)
    scale = 1.0 / DENOM
    if SCALED:
        scale = scale * tl.load(Upstream).to(tl.float32)
    grad = (tl.exp((z - maximum) - log_sum) - (col == target).to(tl.float32)) * scale
    grad = tl.where((target >= 0) & (target < V), grad, 0.0)
    tl.store(Grad + row * G0 + col * G1, grad, mask=col < V)


def parallel_cross_entropy(logits, targets, denom=None, needs_grad=True,
                           inplace_grad=True, grad_out=None, block_size=4096,
                           lse_out=None, upstream_scale=None):
    """Compatible tuple with fused_cross_entropy, plus optional pre-rounding scale.

    Metric entropy covers ALL positions. Caller applies its own attention mask.
    Inputs must be finite; targets are valid class IDs or -100. FP32 grad_out
    avoids FP16 underflow; alternatively pass real AMP upstream_scale BEFORE
    storing gradients. This primitive is not an autograd wrapper itself.
    """
    if (not logits.is_cuda or logits.ndim != 2
            or logits.dtype not in (torch.float16, torch.float32)):
        raise ValueError("expected 2D FP16/FP32 CUDA logits")
    n, v = logits.shape
    if not 0 < v < 16777216:
        raise ValueError("vocabulary must fit exact FP32 index representation")
    if (targets.shape != (n,) or targets.device != logits.device
            or targets.dtype not in (torch.int32, torch.int64)):
        raise ValueError("expected integer targets on logits device")
    if block_size not in (1024, 2048, 4096, 8192):
        raise ValueError("unsupported tile size")
    targets = targets.contiguous()
    if denom is None:
        denom = max(float((targets != -100).sum().item()), 1.)
    denom = float(denom)
    if not denom > 0:
        raise ValueError("denom must be positive")
    if lse_out is None:
        lse_out = torch.empty(n, device=logits.device, dtype=torch.float32)
    if (lse_out.shape != (n,) or lse_out.device != logits.device
            or lse_out.dtype != torch.float32 or not lse_out.is_contiguous()):
        raise ValueError("expected contiguous FP32 lse_out")
    if upstream_scale is not None and (upstream_scale.device != logits.device
            or upstream_scale.numel() != 1 or upstream_scale.dtype != torch.float32):
        raise ValueError("upstream_scale must be a scalar FP32 CUDA tensor")
    loss = torch.empty_like(lse_out)
    entropy = torch.empty_like(lse_out)
    hits = torch.empty(n, device=logits.device, dtype=torch.int32)
    grad = None
    if needs_grad:
        grad = logits if inplace_grad else grad_out
        if grad is None:
            grad = torch.empty_like(logits)
        if (grad.shape != logits.shape or grad.device != logits.device
                or grad.dtype not in (torch.float16, torch.float32)):
            raise ValueError("invalid gradient buffer")
        if not inplace_grad and grad.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr():
            raise ValueError("out-of-place gradient must not alias logits")
    if n:
        chunks = triton.cdiv(v, block_size)
        scratch = torch.empty((n, chunks, 4), device=logits.device, dtype=torch.float32)
        norms = torch.empty((2, n), device=logits.device, dtype=torch.float32)
        with torch.cuda.device(logits.device):
            ce_tile_stats_kernel[(n, chunks)](logits, scratch, v, chunks,
                logits.stride(0), logits.stride(1), block_size, num_warps=4,
                enable_fp_fusion=False)
            ce_merge_stats_kernel[(n,)](logits, targets, scratch, loss, entropy, hits,
                lse_out, norms[0], norms[1], v, chunks, logits.stride(0), logits.stride(1),
                triton.next_power_of_2(chunks), num_warps=4, enable_fp_fusion=False)
            if needs_grad:
                ce_write_grad_kernel[(n, chunks)](logits, targets, norms[0], norms[1], grad,
                    upstream_scale if upstream_scale is not None else lse_out,
                    v, logits.stride(0), logits.stride(1), grad.stride(0), grad.stride(1),
                    denom, upstream_scale is not None, block_size, num_warps=4,
                    enable_fp_fusion=False)
    return loss, entropy, hits, loss.sum() / denom, grad
