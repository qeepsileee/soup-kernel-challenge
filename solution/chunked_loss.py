"""Chunked frozen head with parallel CE statistics and AMP-safe recomputation.

Forward retains only hidden states and compact per-token statistics, not logits.
Backward recomputes labelled rows and applies the actual incoming loss scale
BEFORE storing FP16 dZ and multiplying by the head. Prompt-token entropy remains.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from .parallel_ce import parallel_cross_entropy as fused_cross_entropy

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChunkedStep:
    """Exact metrics container expected by Soup's RewindTrainer and TRL logger."""
    token_entropy: torch.Tensor  # [B, T] Shannon entropy of each position
    token_loss: torch.Tensor     # [B, T] float32 Cross-Entropy of each position (0.0 for ignore)
    correct: torch.Tensor        # scalar int32/int64: total labelled tokens whose argmax == target
    labelled: torch.Tensor       # scalar int32/int64: total positions where target != -100


def shift_for_next_token(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Aligns labels for causal LM next-token prediction: position t predicts token t+1."""
    return F.pad(labels, (0, 1), value=ignore_index)[..., 1:]


@triton.jit
def chunk_scaled_grad_kernel(Logits, Target, LSE, Upstream,
                             V: tl.constexpr, DENOM, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(Logits + row * V + col, mask=col < V, other=-float("inf")).to(tl.float32)
    norm = tl.load(LSE + row)
    target = tl.load(Target + row)
    scale = tl.load(Upstream).to(tl.float32) / DENOM
    value = (tl.exp(z - norm) - (col == target).to(tl.float32)) * scale
    tl.store(Logits + row * V + col, value, mask=col < V)


class _ChunkedHeadLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head_weight, target, chunk_size, denom, with_grad):
        batch, length, dim = hidden.shape
        vocab = head_weight.shape[0]
        compute = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else hidden.dtype
        # Cache the frozen head conversion for this invocation, not per chunk.
        # Do not save this dense converted head across forward/backward.
        weight = head_weight.to(compute)
        flat = hidden.reshape(-1, dim)
        labels = target.reshape(-1)
        rows = flat.shape[0]
        chunk_rows = batch * chunk_size
        losses = torch.empty(rows, device=hidden.device, dtype=torch.float32)
        entropies = torch.empty_like(losses)
        hits = torch.empty(rows, device=hidden.device, dtype=torch.int32)
        lse = torch.empty_like(losses)
        with torch.autocast("cuda", enabled=False):
            for start in range(0, rows, chunk_rows):
                end = min(rows, start + chunk_rows)
                logits = flat[start:end].to(compute) @ weight.t()
                loss, entropy, hit, _, _ = fused_cross_entropy(
                    logits, labels[start:end], denom=denom, needs_grad=False,
                    lse_out=lse[start:end])
                losses[start:end], entropies[start:end], hits[start:end] = loss, entropy, hit
                del logits
        ctx.compute, ctx.chunk_rows, ctx.denom = compute, chunk_rows, denom
        ctx.with_grad = with_grad
        if with_grad:
            # One synchronization to compact labels. Prompt/pad rows have zero
            # loss derivative; do not recompute their head or multiply their dZ.
            valid = (labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
            ctx.save_for_backward(hidden, head_weight, labels, lse, valid)
        token_loss = losses.view(batch, length)
        token_entropy = entropies.view(batch, length)
        total_hits = hits.sum().to(torch.int64)
        ctx.mark_non_differentiable(token_entropy, token_loss, total_hits)
        return losses.sum() / denom, token_entropy, token_loss, total_hits

    @staticmethod
    def backward(ctx, grad_loss, _entropy, _loss, _hits):
        if not ctx.with_grad or grad_loss is None:
            return None, None, None, None, None, None
        hidden, head_weight, labels, lse, valid = ctx.saved_tensors
        dim = hidden.shape[-1]
        vocab = head_weight.shape[0]
        flat = hidden.reshape(-1, dim)
        grad_hidden = torch.zeros_like(hidden)
        grad_flat = grad_hidden.view(-1, dim)
        if valid.numel():
            weight = head_weight.to(ctx.compute)
            with torch.autocast("cuda", enabled=False):
                for start in range(0, valid.numel(), ctx.chunk_rows):
                    index = valid[start:start + ctx.chunk_rows]
                    logits = flat[index].to(ctx.compute) @ weight.t()
                    targets = labels[index].contiguous()
                    norms = lse[index].contiguous()
                    with torch.cuda.device(hidden.device):
                        chunk_scaled_grad_kernel[(index.numel(), triton.cdiv(vocab, 4096))](
                            logits, targets, norms, grad_loss, vocab, ctx.denom, 4096,
                            num_warps=4, enable_fp_fusion=False)
                    # Loss scale is applied BEFORE rounding dZ to FP16/GEMM.
                    grad_flat[index] = (logits @ weight).to(hidden.dtype)
                    del logits
        return grad_hidden, None, None, None, None, None


class _PrecomputedHeadLossFunction(torch.autograd.Function):
    """Experimental first-order loss: reuse forward logits for dHidden.

    A power-of-two anchor prevents FP16 underflow before the head GEMM.
    Actual upstream scaling is still applied in backward. Rounding can differ
    from scaling dLogits first: this is opt-in and needs full quality validation.
    """
    @staticmethod
    def forward(ctx, hidden, head_weight, target, chunk_size, denom, with_grad, attention_mask):
        batch, length, dim = hidden.shape
        compute = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else hidden.dtype
        weight = head_weight.to(compute)
        flat, labels = hidden.reshape(-1, dim), target.reshape(-1)
        rows, vocab = flat.shape[0], weight.shape[0]
        # Labelled positions first: dHidden GEMMs never process prompt/pad rows.
        valid = (labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        entropy_rows = labels == IGNORE_INDEX
        if attention_mask is not None:
            # Entropy is consumed only where attention_mask is nonzero. Retain
            # EVERY labelled row above, even at a masked position (e.g. a left
            # padding boundary). Prompt entropy remains, including -100 labels.
            entropy_rows = entropy_rows & (attention_mask.reshape(-1) != 0)
        ignored = entropy_rows.nonzero(as_tuple=False).flatten()
        order = torch.cat((valid, ignored))
        # Unprojected padding has no loss, hit or consumed entropy. This sentinel
        # changes no logged/rewind value: only (mask=0 AND target=-100) is skipped.
        complete = order.numel() == rows
        allocate_stats = torch.empty if complete else torch.zeros
        losses = allocate_stats(rows, device=hidden.device, dtype=torch.float32)
        entropies = torch.empty_like(losses) if complete else torch.zeros_like(losses)
        hits = allocate_stats(rows, device=hidden.device, dtype=torch.int32)
        gradient = torch.zeros((rows, dim), device=hidden.device, dtype=torch.float32) if with_grad else None
        # Keep the negative target entry below FP16's maximum even when a
        # diagnostic contains only one labelled token.
        anchor = min(65536.0, 16384.0 * denom)
        scale = torch.tensor(anchor, device=hidden.device, dtype=torch.float32)
        with torch.autocast("cuda", enabled=False):
            for start in range(0, order.numel(), batch * chunk_size):
                index = order[start:start + batch * chunk_size]
                logits = flat[index].to(compute) @ weight.t()
                lse = torch.empty(index.numel(), device=hidden.device, dtype=torch.float32)
                loss, entropy, hit, _, _ = fused_cross_entropy(
                    logits, labels[index], denom=denom, needs_grad=False, lse_out=lse)
                losses[index], entropies[index], hits[index] = loss, entropy, hit
                count = min(index.numel(), max(valid.numel() - start, 0))
                if with_grad and count:
                    targets = labels[index[:count]].contiguous()
                    with torch.cuda.device(hidden.device):
                        chunk_scaled_grad_kernel[(count, triton.cdiv(vocab, 4096))](
                            logits, targets, lse, scale, vocab, denom, 4096,
                            num_warps=4, enable_fp_fusion=False)
                    gradient[index[:count]] = (logits[:count] @ weight).float() / anchor
                del logits
        if attention_mask is not None and not getattr(_PrecomputedHeadLossFunction, "_padding_reported", False):
            # One host-only message per process; sizes come from nonzero() which
            # this path already needs. No extra device synchronization or work.
            print(f"[head_padding] projected_rows={order.numel()} total_rows={rows} "
                  f"labelled_rows={valid.numel()}", flush=True)
            _PrecomputedHeadLossFunction._padding_reported = True
        ctx.dtype = hidden.dtype
        ctx.shape = hidden.shape
        ctx.with_grad = with_grad
        if with_grad:
            ctx.save_for_backward(gradient)
        total_hits = hits.sum().to(torch.int64)
        token_loss, token_entropy = losses.view(batch, length), entropies.view(batch, length)
        ctx.mark_non_differentiable(token_entropy, token_loss, total_hits)
        return losses.sum() / denom, token_entropy, token_loss, total_hits

    @staticmethod
    def backward(ctx, grad_loss, _entropy, _loss, _hits):
        if not ctx.with_grad or grad_loss is None:
            return None, None, None, None, None, None, None
        gradient, = ctx.saved_tensors
        grad = (gradient * grad_loss.float()).to(ctx.dtype).view(ctx.shape)
        return grad, None, None, None, None, None, None


def chunked_head_loss(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    chunk_size: int = 64,
    num_items_in_batch: Optional[Any] = None,
    shift_labels: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
    precompute_grad: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ChunkedStep]:
    """Computes Causal LM loss and compact metrics without full logits materialization.

    Args:
        hidden: [B, T, D] hidden states before output projection.
        lm_head: output projection module (nn.Linear).
        labels: [B, T] original sequence labels.
        chunk_size: number of sequence positions per chunk (default: 64).
        num_items_in_batch: optional batch item count for loss normalization.
        shift_labels: optional pre-shifted labels.
        ignore_index: ignored target index (default: -100).
        attention_mask: optional [B,T] mask used by the trainer's entropy metric.
            In precomputed training, skip only mask=0 AND ignored-target rows.
            Their unconsumed token_entropy entries are zero; every prompt/label
            row and all externally consumed metrics/gradients are retained.
            Without this argument, retain the original full-position behavior.

    Returns:
        loss: scalar tensor loss.
        step: ChunkedStep containing token_entropy, token_loss, correct, labelled.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")

    target = shift_labels if shift_labels is not None else shift_for_next_token(labels, ignore_index)
    target = target.to(hidden.device).contiguous()
    if target.shape != hidden.shape[:2]:
        raise ValueError("target shape must match hidden batch/sequence dimensions")
    if attention_mask is not None and (not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != hidden.shape[:2] or attention_mask.device != hidden.device):
        raise ValueError("attention_mask must match hidden batch/sequence dimensions and device")

    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        denom = float(max(labelled.item(), 1))
    elif torch.is_tensor(num_items_in_batch):
        denom = float(max(num_items_in_batch.item(), 1))
    else:
        denom = float(max(num_items_in_batch, 1))

    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    head_weight = lm_head.weight
    if head_weight.requires_grad:
        raise ValueError("chunked loss requires a frozen lm_head")
    if getattr(lm_head, "bias", None) is not None:
        raise ValueError("chunked loss currently requires a bias-free lm_head")
    if head_weight.ndim != 2 or head_weight.shape[1] != hidden.shape[-1]:
        raise ValueError("head shape must match hidden dimension")

    if precompute_grad and with_grad:
        loss, token_entropy, token_loss, correct = _PrecomputedHeadLossFunction.apply(
            hidden, head_weight, target, chunk_size, denom, with_grad, attention_mask)
    else:
        # The exact recompute and no-grad/eval path keeps its validated behavior.
        loss, token_entropy, token_loss, correct = _ChunkedHeadLossFunction.apply(
            hidden, head_weight, target, chunk_size, denom, with_grad)

    step = ChunkedStep(
        token_entropy=token_entropy,
        token_loss=token_loss,
        correct=correct,
        labelled=labelled,
    )
    return loss, step

