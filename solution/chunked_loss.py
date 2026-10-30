"""Chunked LM Head + Fused Cross-Entropy + Backward.

Eliminates full logits materialization:
Instead of computing full [B, T, V] logits (up to 1.25 GB in FP32),
this processes tokens in chunks (e.g. C=64 or 128 tokens) along sequence:
1. Z_chunk = H_chunk @ W_head.T (cuBLAS GEMM)
2. Fused CE Triton kernel computes loss, per-token entropy, accuracy hits,
   and in-place writes dZ_chunk = (softmax(z) - one_hot) / denom into Z_chunk buffer.
3. dH_chunk = dZ_chunk @ W_head (cuBLAS GEMM)
4. Deallocates Z_chunk immediately.

In backward:
- Incoming grad_loss contains PyTorch AMP loss scaler (e.g. 65536.0 or 1.0).
- Multiplies saved unscaled d_hidden by grad_loss.
- Returns loss and ChunkedStep for Soup RewindTrainer and TRL metrics.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn.functional as F
from .fused_ce import fused_cross_entropy

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


class _ChunkedHeadLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        head_weight: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int,
        denom: float,
        with_grad: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden.shape
        vocab_size = head_weight.shape[0]

        grad_hidden = torch.zeros_like(hidden) if with_grad else None
        total_loss = torch.zeros((), dtype=torch.float32, device=hidden.device)

        entropies = []
        losses = []
        total_hits = torch.zeros((), dtype=torch.int64, device=hidden.device)

        # Iterate over sequence in chunks of chunk_size tokens
        for start in range(0, seq_len, chunk_size):
            stop = min(start + chunk_size, seq_len)
            c_len = stop - start

            # [B, c_len, D] -> [B * c_len, D]
            piece = hidden[:, start:stop].reshape(-1, hidden_dim)
            labels_piece = target[:, start:stop].reshape(-1)

            # cuBLAS GEMM: [B * c_len, D] @ [D, V] -> [B * c_len, V]
            # Uses the dtype of hidden (FP16 on T4)
            chunk_logits = torch.matmul(piece, head_weight.t())

            # Fused Triton CE: computes loss, entropy, accuracy, and in-place writes dZ into chunk_logits
            chunk_loss, chunk_entropy, chunk_hits, _, _ = fused_cross_entropy(
                logits=chunk_logits,
                targets=labels_piece,
                denom=denom,
                needs_grad=with_grad,
                inplace_grad=True,  # In-place gradient overwrite saves allocating [chunk, V]
                block_size=4096,
            )

            total_loss = total_loss + chunk_loss.sum()
            total_hits = total_hits + chunk_hits.sum().to(torch.int64)

            # Reshape back to [B, c_len]
            losses.append(chunk_loss.view(batch_size, c_len))
            entropies.append(chunk_entropy.view(batch_size, c_len))

            if with_grad:
                # cuBLAS GEMM for d_hidden: [B * c_len, V] @ [V, D] -> [B * c_len, D]
                # chunk_logits now holds dZ = (softmax(z) - target) / denom
                d_piece = torch.matmul(chunk_logits, head_weight)
                grad_hidden[:, start:stop] = d_piece.view(batch_size, c_len, hidden_dim)

            # Explicitly drop the [B * c_len, V] temporary buffer
            del chunk_logits

        token_loss = torch.cat(losses, dim=1)
        token_entropy = torch.cat(entropies, dim=1)

        ctx.with_grad = with_grad
        if with_grad:
            ctx.save_for_backward(grad_hidden)
        ctx.mark_non_differentiable(token_entropy, token_loss, total_hits)

        normalized_loss = total_loss / denom
        return normalized_loss, token_entropy, token_loss, total_hits

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_hits):
        if not ctx.with_grad:
            return None, None, None, None, None, None
        (grad_hidden,) = ctx.saved_tensors
        # grad_loss comes from PyTorch autograd / AMP GradScaler and carries loss scaling.
        # Scale the unscaled d_hidden by grad_loss:
        scaled_grad_hidden = grad_hidden * grad_loss
        return scaled_grad_hidden, None, None, None, None, None


def chunked_head_loss(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    chunk_size: int = 64,
    num_items_in_batch: Optional[Any] = None,
    shift_labels: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
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

    Returns:
        loss: scalar tensor loss.
        step: ChunkedStep containing token_entropy, token_loss, correct, labelled.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")

    target = shift_labels if shift_labels is not None else shift_for_next_token(labels, ignore_index)
    target = target.to(hidden.device)

    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        denom = float(max(labelled.item(), 1))
    elif torch.is_tensor(num_items_in_batch):
        denom = float(max(num_items_in_batch.item(), 1))
    else:
        denom = float(max(num_items_in_batch, 1))

    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    head_weight = lm_head.weight

    loss, token_entropy, token_loss, correct = _ChunkedHeadLossFunction.apply(
        hidden, head_weight, target, chunk_size, denom, with_grad
    )

    step = ChunkedStep(
        token_entropy=token_entropy,
        token_loss=token_loss,
        correct=correct,
        labelled=labelled,
    )
    return loss, step

