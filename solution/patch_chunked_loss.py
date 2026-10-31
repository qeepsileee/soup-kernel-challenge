"""Soup/TRL trainer integration for chunked LM head + fused Triton CE.

We hook into the same two seams that Soup's own ``stream_loss_chunk`` uses:

1. ``model.forward`` — when called with ``labels``, the head is held back and
   the hidden states are passed to ``chunked_head_loss`` which runs our Triton
   fused_ce kernel chunk-by-chunk without ever materialising the full
   [B, T, 151936] logits tensor.

2. ``trainer.compute_loss`` — we subclass ``SFTTrainer`` exactly as Soup's
   ``make_chunked_loss_trainer_class`` does, reading the ``ChunkedStep`` stored
   by our hook and logging ``entropy``, ``mean_token_accuracy`` and
   ``num_tokens`` in the same way TRL/Soup would from full logits.

Integration with Soup's existing machinery:

* ``HOOK_ATTR = "_soup_chunked_head_loss"`` — Soup's ``RewindTrainer``
  already checks this attr. Setting it on the trainer makes ``rewind.jsonl``
  switch automatically to ``row_losses_from_token_losses(step.token_loss)``.

* ``STEP_ATTR = "_soup_chunked_step"`` — written by our compute_loss override
  after each labelled forward; read by ``RewindTrainer.compute_loss``.

Only Qwen-style models (no final_logit_softcapping, frozen head) qualify.
Fallback: if the model fails validation or is not a CausalLM with a frozen
output projection, the function is a no-op and training continues unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

# Must match Soup's attribute names exactly so RewindTrainer can find the step.
_HOOK_ATTR = "_soup_chunked_head_loss"
_STEP_ATTR = "_soup_chunked_step"

# Default chunk size in sequence-length tokens per GEMM call.
# At B=4, C=64: one chunk logits buffer = 4*64*151936*2 bytes ≈ 74 MB (FP16).
# At C=32 it halves to ~37 MB. Tune here without changing the kernel.
_DEFAULT_CHUNK = 128


# ---------------------------------------------------------------------------
# Hook — wraps the CausalLM forward to intercept hidden states
# ---------------------------------------------------------------------------

class _OwnChunkedHeadLossHook:
    """Attaches to a frozen-head CausalLM and replaces labelled forwards.

    Compatible with Soup's ``ChunkedHeadLoss`` duck-type: exposes
    ``take_step()`` and stores the last ``ChunkedStep`` in ``last_step``.
    """

    def __init__(self, causal_lm: Any, chunk: int) -> None:
        from .chunked_loss import chunked_head_loss

        head = causal_lm.get_output_embeddings()
        if head is None:
            raise ValueError(
                f"{type(causal_lm).__name__} has no output embeddings "
                "(get_output_embeddings returned None)."
            )
        if getattr(head, "bias", None) is not None:
            raise ValueError("chunked head loss requires a bias-free output projection")
        # Qwen2.5 has no soft-cap; bail early if the model does.
        cap = getattr(getattr(causal_lm, "config", None), "final_logit_softcapping", None)
        if cap is not None:
            raise ValueError(
                f"{type(causal_lm).__name__} uses final_logit_softcapping={cap}; "
                "chunked head loss cannot be applied."
            )
        trainable = [n for n, p in head.named_parameters() if p.requires_grad]
        if trainable:
            raise ValueError(
                "chunked head loss requires a frozen output head; "
                f"trainable params: {', '.join(trainable)}"
            )

        self.chunk = chunk
        self.last_step = None
        self._chunked_head_loss = chunked_head_loss
        self._head = head
        self._causal_lm = causal_lm
        self._original_forward = causal_lm.forward
        self._holding = False
        self._held: Optional[Any] = None  # (hidden, placeholder) when _holding

        # Intercept head.forward to capture hidden states before projection.
        self._original_head_forward = head.forward
        self._before_head = head.__dict__.get("forward")
        self._before_lm = causal_lm.__dict__.get("forward")

        head.forward = self._head_or_hold
        causal_lm.forward = self._chunked_forward

    def remove(self) -> None:
        """Restore original forwards."""
        for module, saved in (
            (self._head, self._before_head),
            (self._causal_lm, self._before_lm),
        ):
            if saved is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = saved

    def take_step(self):
        """Return and clear the last labelled step (called once per forward)."""
        step, self.last_step = self.last_step, None
        return step

    def _head_or_hold(self, hidden: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._holding:
            return self._original_head_forward(hidden, *args, **kwargs)
        # Return a cost-free placeholder (no vocab axis) so the model can
        # complete its forward pass and return outputs.logits = placeholder.
        placeholder = hidden.new_empty((*hidden.shape[:-1], 0))
        self._held = (hidden, placeholder)
        return placeholder

    def _chunked_forward(self, *args: Any, **kwargs: Any) -> Any:
        labels = kwargs.pop("labels", None)
        if labels is None:
            # Generation / evaluation without labels: pass through unchanged.
            return self._original_forward(*args, **kwargs)

        self._holding, self._held = True, None
        try:
            outputs = self._original_forward(*args, **kwargs)
        finally:
            self._holding = False

        held, self._held = self._held, None

        # Validate that the model did not transform logits after the head
        # (e.g. a softcap we didn't catch at init time).
        if held is None or getattr(outputs, "logits", None) is not held[1]:
            raise ValueError(
                f"{type(self._causal_lm).__name__} modified the head output "
                "after our placeholder was returned — chunked loss cannot be applied."
            )

        hidden_states = held[0]
        import torch
        attention_mask = kwargs.get("attention_mask")
        if (not isinstance(attention_mask, torch.Tensor)
                or attention_mask.shape != hidden_states.shape[:2]
                or attention_mask.device != hidden_states.device):
            attention_mask = None  # Unsupported layouts retain the full head path.
        loss, step = self._chunked_head_loss(
            hidden_states,
            self._head,
            labels,
            chunk_size=self.chunk,
            num_items_in_batch=kwargs.get("num_items_in_batch"),
            shift_labels=kwargs.get("shift_labels"),
            precompute_grad=True,
            attention_mask=attention_mask,
        )
        self.last_step = step

        # Return same output class with loss and no logits (saves memory).
        rest = {k: v for k, v in outputs.items() if k not in ("loss", "logits")}
        return type(outputs)(loss=loss, **rest)


# ---------------------------------------------------------------------------
# Trainer subclass — mirrors make_chunked_loss_trainer_class exactly
# ---------------------------------------------------------------------------

def _make_own_chunked_trainer_class(base_cls: type) -> type:
    """Subclass of ``base_cls`` that reads metrics from our hook's ChunkedStep."""
    from transformers import Trainer

    class OwnChunkedLossTrainer(base_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            model = getattr(self, "model", None)
            if callable(getattr(model, "get_output_embeddings", None)) or callable(getattr(model, "get_base_model", None)):
                attach_hook(model, self, chunk_size=getattr(type(self), "_chunk_size", _DEFAULT_CHUNK))

        def compute_loss(
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            hook = getattr(self, _HOOK_ATTR, None)
            if hook is None or "labels" not in inputs:
                return super().compute_loss(
                    model, inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            mode = "train" if self.model.training else "eval"
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False
            loss, outputs = Trainer.compute_loss(
                self, model, inputs, return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            step = hook.take_step()
            setattr(self, _STEP_ATTR, step)
            if step is not None:
                _log_trl_metrics(self, mode, inputs, step)
            return (loss, outputs) if return_outputs else loss

    OwnChunkedLossTrainer.__name__ = f"OwnChunked{base_cls.__name__}"
    OwnChunkedLossTrainer.__qualname__ = OwnChunkedLossTrainer.__name__
    return OwnChunkedLossTrainer


def _log_trl_metrics(trainer: Any, mode: str, inputs: Any, step: Any) -> None:
    """Log entropy, num_tokens and mean_token_accuracy from ChunkedStep.

    Applies separate masks per metric, matching TRL's exact formulas:
    - entropy: attention_mask (includes prompt tokens, excludes pad).
    - accuracy: labelled positions only (target != -100).
    """
    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:
        return

    # --- Entropy: mask by attention_mask (all non-pad tokens, including prompt) ---
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
        tokens = gather(attention_mask.sum()).sum().item()
    else:
        entropy = step.token_entropy.mean()
        tokens = None

    metrics[mode]["entropy"].append(gather(entropy).mean().item())

    if mode == "train" and tokens is not None and hasattr(trainer, "_total_train_tokens"):
        trainer._total_train_tokens += tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]

    # --- Accuracy: mask by labelled positions (target != -100) ---
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


# ---------------------------------------------------------------------------
# Public patch / unpatch API
# ---------------------------------------------------------------------------

def patch_chunked_loss(chunk_size: int = _DEFAULT_CHUNK) -> None:
    """Patch SFTTrainer to use our chunked Triton CE kernel.

    Must be called from ``apply()`` AFTER the model is loaded — specifically
    inside the trainer's ``setup()`` cycle. We hook Soup's SFT trainer
    ``_build_trainer``-equivalent path by patching SFTTrainer at class level.

    Safe to call multiple times (idempotent).
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    try:
        from trl import SFTTrainer
    except ImportError:
        return  # TRL not present; skip silently

    if getattr(SFTTrainer, "_own_chunked_loss_patched", False):
        return

    # Replace SFTTrainer with our subclass in TRL's module namespace so Soup
    # picks it up when it does ``from trl import SFTTrainer``.
    import trl as _trl
    import trl.trainer.sft_trainer as _sft_mod

    _orig_cls = SFTTrainer
    _new_cls = _make_own_chunked_trainer_class(_orig_cls)
    _new_cls._own_chunked_loss_patched = True
    _new_cls._original_cls = _orig_cls
    _new_cls._chunk_size = chunk_size

    _trl.SFTTrainer = _new_cls
    if hasattr(_sft_mod, "SFTTrainer"):
        _sft_mod.SFTTrainer = _new_cls

    # Also patch soup_cli's import alias if already imported.
    try:
        from soup_cli.trainer import sft as _soup_sft
        if hasattr(_soup_sft, "SFTTrainer"):
            _soup_sft.SFTTrainer = _new_cls
    except ImportError:
        pass


def attach_hook(model: Any, trainer: Any, chunk_size: int = _DEFAULT_CHUNK) -> None:
    """Attach the chunked head loss hook to an already-loaded model and trainer.

    Call this from a Soup callback or directly after model loading if you have
    access to both objects. ``patch_chunked_loss()`` patches the trainer class
    so this function only needs to wire the runtime hook.

    No-op if:
    - model has no frozen output head.
    - model has final_logit_softcapping.
    - hook already attached.
    """
    if getattr(trainer, _HOOK_ATTR, None) is not None:
        return  # Already attached

    # Unwrap PEFT wrapper to reach the CausalLM.
    causal_lm = model
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        causal_lm = get_base()

    try:
        hook = _OwnChunkedHeadLossHook(causal_lm, chunk=chunk_size)
    except ValueError:
        # Unsupported model; fall back to full logits silently.
        return

    setattr(trainer, _HOOK_ATTR, hook)


def detach_hook(trainer: Any) -> None:
    """Remove the hook and restore the original forward methods."""
    hook = getattr(trainer, _HOOK_ATTR, None)
    if hook is not None:
        hook.remove()
        try:
            delattr(trainer, _HOOK_ATTR)
        except AttributeError:
            pass

