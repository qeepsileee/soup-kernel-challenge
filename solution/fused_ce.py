"""Fused Cross-Entropy + Metrics + Backward kernel in Triton.

Computes in a single fused operation over vocabulary:
1. LogSumExp and Cross-Entropy loss for labelled positions (target != -100).
2. Per-token Shannon entropy for ALL positions (preserves prompt-token entropy).
3. Online argmax token hit (accuracy metric for target != -100).
4. Analytical backward gradients d_logits = (softmax(z) - target) / denom.

Designed for large vocabulary (Qwen2.5 152k vocab) on SM75 (Kaggle T4).
Processes vocabulary in tiles of BLOCK_SIZE (e.g. 4096), keeping only registers in SRAM.
Supports in-place gradient overwrite to avoid allocating separate [chunk, V] gradient buffers.
"""

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def fused_ce_kernel(
    Logits, Targets,
    Loss, Entropy, Hits, Grad, LSE,
    stride_lz, stride_lv,
    stride_gz, stride_gv,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    denom: float,
    HAS_GRAD: tl.constexpr,
    INPLACE_GRAD: tl.constexpr,
    SAVE_LSE: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(Targets + row).to(tl.int64)
    valid_target = (target >= 0) & (target < V)

    # -------------------------------------------------------------------
    # Pass 1: Streaming reduction across vocabulary (LSE, entropy, argmax)
    # -------------------------------------------------------------------
    m = -float("inf")
    s = 0.0
    t = 0.0
    best_val = -float("inf")
    best_idx = 0

    for start in range(0, V, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        ptr = Logits + row * stride_lz + cols * stride_lv
        z = tl.load(ptr, mask=mask, other=-float("inf")).to(tl.float32)

        cur_m = tl.max(z, 0)
        cur_idx = start + tl.argmax(z, 0)

        # Update running argmax
        is_new_best = cur_m > best_val
        best_val = tl.maximum(best_val, cur_m)
        best_idx = tl.where(is_new_best, cur_idx, best_idx)

        # Update running online softmax and entropy moment
        new_m = tl.maximum(m, cur_m)
        alpha_old = tl.exp(m - new_m)
        alpha_cur = tl.exp(cur_m - new_m)

        shifted_cur = z - cur_m
        p_cur = tl.exp(shifted_cur)
        cur_s = tl.sum(tl.where(mask, p_cur, 0.0), 0)
        cur_t = tl.sum(tl.where(mask, p_cur * shifted_cur, 0.0), 0)

        s_old = s
        s = s_old * alpha_old + cur_s * alpha_cur
        old_delta = tl.where(s_old > 0, m - new_m, 0.0)
        t = alpha_old * (t + s_old * old_delta) + alpha_cur * (cur_t + cur_s * (cur_m - new_m))
        m = new_m

    lse = m + tl.log(s)
    if SAVE_LSE:
        tl.store(LSE + row, lse)
    # Entropy: H = log(s) - t / s (in nats)
    entropy_val = tl.log(s) - t / s
    tl.store(Entropy + row, entropy_val)

    # Per-token Cross-Entropy loss and accuracy hit
    if valid_target:
        target_ptr = Logits + row * stride_lz + target * stride_lv
        target_logit = tl.load(target_ptr).to(tl.float32)
        loss_val = lse - target_logit
        is_hit = (best_idx == target).to(tl.int32)
    else:
        loss_val = 0.0
        is_hit = 0

    tl.store(Loss + row, loss_val)
    tl.store(Hits + row, is_hit.to(tl.int32))

    # -------------------------------------------------------------------
    # Pass 2: Gradients d_logits = (softmax(z) - one_hot) / denom
    # -------------------------------------------------------------------
    if HAS_GRAD:
        scale = 1.0 / denom
        for start in range(0, V, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < V

            ptr_in = Logits + row * stride_lz + cols * stride_lv
            z = tl.load(ptr_in, mask=mask, other=-float("inf")).to(tl.float32)

            if valid_target:
                prob = tl.exp(z - lse)
                is_tgt = (cols == target)
                grad_val = (prob - tl.where(is_tgt, 1.0, 0.0)) * scale
            else:
                grad_val = 0.0

            if INPLACE_GRAD:
                ptr_out = ptr_in
            else:
                ptr_out = Grad + row * stride_gz + cols * stride_gv

            tl.store(ptr_out, grad_val.to(Grad.dtype.element_ty), mask=mask)


def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    denom: Optional[float] = None,
    needs_grad: bool = True,
    inplace_grad: bool = True,
    grad_out: Optional[torch.Tensor] = None,
    block_size: int = 4096,
    lse_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Runs fused cross-entropy + metrics + gradient.

    Args:
        logits: [N, V] float16 or float32 tensor on CUDA.
        targets: [N] int64 or int32 tensor on CUDA.
        denom: normalization denominator. If None, sum of (targets != -100) is used.
        needs_grad: whether to compute gradients w.r.t logits.
        inplace_grad: if True, writes gradients directly into `logits` buffer.
        grad_out: optional destination tensor for gradients if not inplace.
        block_size: tile size along vocabulary dimension (default: 4096).

    Returns:
        loss: [N] float32 per-token loss (0.0 for target == -100).
        entropy: [N] float32 per-token entropy (computed for all tokens).
        hits: [N] int32 per-token accuracy indicator (1 if argmax == target, else 0).
        total_loss: scalar float32 sum of loss / denom.
        grad: [N, V] gradients if needs_grad, or None.
    """
    assert logits.is_cuda, "logits must be on CUDA"
    assert targets.is_cuda, "targets must be on CUDA"
    assert logits.dim() == 2, f"expected 2D logits [N, V], got {logits.shape}"
    assert targets.dim() == 1, f"expected 1D targets [N], got {targets.shape}"
    assert logits.shape[0] == targets.shape[0], "batch size mismatch"

    n_rows, vocab_size = logits.shape
    if denom is None:
        valid_count = (targets >= 0).sum().item()
        denom = max(float(valid_count), 1.0)

    loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    entropy = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    hits = torch.empty(n_rows, device=logits.device, dtype=torch.int32)

    if needs_grad:
        if inplace_grad:
            grad = logits
        elif grad_out is not None:
            grad = grad_out
        else:
            grad = torch.empty_like(logits)
    else:
        grad = logits  # placeholder, unused

    num_warps = 8 if block_size >= 4096 else 4
    with torch.cuda.device(logits.device):
        fused_ce_kernel[(n_rows,)](
            logits, targets,
            loss, entropy, hits, grad, lse_out if lse_out is not None else loss,
            logits.stride(0), logits.stride(1),
            grad.stride(0), grad.stride(1),
            V=vocab_size,
            BLOCK_SIZE=block_size,
            denom=float(denom),
            HAS_GRAD=bool(needs_grad),
            INPLACE_GRAD=bool(inplace_grad and needs_grad),
            SAVE_LSE=lse_out is not None,
            num_warps=num_warps,
        )

    total_loss = loss.sum() / denom
    return loss, entropy, hits, total_loss, (grad if needs_grad else None)

