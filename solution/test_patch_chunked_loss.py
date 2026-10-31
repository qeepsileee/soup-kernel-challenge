"""Unit tests for patch_chunked_loss Soup/TRL integration."""

import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class PatchChunkedLossTests(unittest.TestCase):

    def test_default_configuration_ignores_stale_environment(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from solution.patch_chunked_loss import attach_hook, detach_hook, _HOOK_ATTR

        model = self._make_tiny_causal_lm()
        trainer = SimpleNamespace()
        with patch.dict(os.environ, {"SOUP_EXPERIMENT_HEAD_CHUNK": "1",
                                     "SOUP_EXPERIMENT_PRECOMPUTE_HEAD": "0"}):
            attach_hook(model, trainer)
            hook = getattr(trainer, _HOOK_ATTR)
            try:
                with patch.object(hook, "_chunked_head_loss", wraps=hook._chunked_head_loss) as loss:
                    model(labels=torch.randint(512, (2, 32), device="cuda"))
                self.assertEqual(loss.call_args.kwargs["chunk_size"], 128)
                self.assertTrue(loss.call_args.kwargs["precompute_grad"])
            finally:
                detach_hook(trainer)

    def _make_tiny_causal_lm(self, batch=2, seq=32, hidden=128, vocab=512):
        """Build a minimal CausalLM-like object for testing the hook."""
        import torch
        import torch.nn as nn

        class FakeCausalLMOutputs:
            def __init__(self, **kw):
                self._d = kw
            def __getattr__(self, k):
                return self._d.get(k)
            def items(self):
                return self._d.items()

        class FakeHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(vocab, hidden, device="cuda", dtype=torch.float16),
                    requires_grad=False,  # frozen
                )
            def forward(self, x):
                return torch.matmul(x, self.weight.t())

        class FakeCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = FakeHead()
                self._b, self._s, self._h = batch, seq, hidden

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, input_ids=None, labels=None, **kw):
                # Simulate a decoder: output random hidden states, then head.
                hidden = torch.randn(
                    self._b, self._s, self._h,
                    device="cuda", dtype=torch.float16, requires_grad=True
                )
                logits = self.lm_head(hidden)
                return FakeCausalLMOutputs(logits=logits, loss=None)

        return FakeCausalLM().cuda()

    def test_hook_intercepts_and_returns_loss(self):
        import torch
        from solution.patch_chunked_loss import _OwnChunkedHeadLossHook, _HOOK_ATTR, _STEP_ATTR

        model = self._make_tiny_causal_lm()
        hook = _OwnChunkedHeadLossHook(model, chunk=8)

        batch, seq, vocab = 2, 32, 512
        input_ids = torch.zeros(batch, seq, dtype=torch.long, device="cuda")
        labels = torch.randint(0, vocab, (batch, seq), device="cuda", dtype=torch.long)
        labels[:, :8] = -100  # prompt tokens

        # Forward through hooked model
        outputs = model(input_ids=input_ids, labels=labels)

        # logits must be absent (None), loss must be a scalar
        self.assertIsNone(getattr(outputs, "logits", None))
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.shape, ())
        self.assertTrue(torch.isfinite(outputs.loss))

        # ChunkedStep must be stored
        step = hook.take_step()
        self.assertIsNotNone(step)
        self.assertEqual(step.token_entropy.shape, (batch, seq))
        self.assertEqual(step.token_loss.shape, (batch, seq))
        self.assertTrue((step.token_entropy > 0).all())

        hook.remove()

    def test_hook_passthrough_without_labels(self):
        """Without labels, the hook must not intercept (generation mode)."""
        import torch
        from solution.patch_chunked_loss import _OwnChunkedHeadLossHook

        model = self._make_tiny_causal_lm()
        hook = _OwnChunkedHeadLossHook(model, chunk=8)

        input_ids = torch.zeros(2, 16, dtype=torch.long, device="cuda")
        outputs = model(input_ids=input_ids)  # no labels

        # In passthrough mode logits are still present
        self.assertIsNotNone(getattr(outputs, "logits", None))
        self.assertIsNone(hook.take_step())  # no step stored

        hook.remove()

    def test_hook_remove_restores_originals(self):
        """After remove(), model.forward and head.forward are restored."""
        import torch
        from solution.patch_chunked_loss import _OwnChunkedHeadLossHook

        model = self._make_tiny_causal_lm()
        original_fwd = model.forward
        hook = _OwnChunkedHeadLossHook(model, chunk=8)

        # Forward is replaced while hook is active
        self.assertIsNot(model.forward, original_fwd)

        hook.remove()

        # Forward is restored after remove
        self.assertIs(model.forward.__func__, original_fwd.__func__)
        self.assertIs(model.forward.__self__, original_fwd.__self__)

    def test_chunked_loss_agrees_with_full_logits(self):
        """Chunked loss must match full PyTorch CE to within FP16 rounding."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from solution.chunked_loss import chunked_head_loss

        b, t, d, v = 2, 64, 128, 1024
        head = nn.Linear(d, v, bias=False, device="cuda", dtype=torch.float16)
        head.weight.requires_grad_(False)

        for chunk in (16, 32, 64):
            with self.subTest(chunk=chunk):
                hidden_ref = torch.randn((b, t, d), device="cuda", dtype=torch.float16,
                                        requires_grad=True)
                hidden_cand = hidden_ref.detach().clone().requires_grad_(True)

                labels = torch.randint(0, v, (b, t), device="cuda", dtype=torch.long)
                labels[:, :16] = -100

                # Reference
                full_logits = head(hidden_ref)
                shift_logits = full_logits[:, :-1].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                denom = float(max((shift_labels != -100).sum().item(), 1))
                ref_loss = F.cross_entropy(
                    shift_logits.view(-1, v), shift_labels.view(-1),
                    ignore_index=-100, reduction="sum"
                ) / denom
                (ref_grad,) = torch.autograd.grad(ref_loss, hidden_ref)

                # Candidate
                cand_loss, step = chunked_head_loss(
                    hidden_cand, head, labels,
                    chunk_size=chunk, num_items_in_batch=denom
                )
                (cand_grad,) = torch.autograd.grad(cand_loss, hidden_cand)

                torch.testing.assert_close(cand_loss, ref_loss, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(cand_grad, ref_grad, rtol=3e-3, atol=1e-2)


if __name__ == "__main__":
    unittest.main()

