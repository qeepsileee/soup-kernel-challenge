"""Unit tests for Chunked LM Head + Fused CE + Backward."""

import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class ChunkedLossTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(42)

    def test_chunked_vs_full_logits_loss_and_grad(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from solution.chunked_loss import chunked_head_loss, shift_for_next_token

        b, t, d = 2, 64, 128
        vocab_size = 2048

        lm_head = nn.Linear(d, vocab_size, bias=False, device="cuda", dtype=torch.float16)
        lm_head.weight.requires_grad_(False)  # Frozen head in LoRA/QLoRA

        for chunk_size in (16, 32, 64):
            with self.subTest(chunk_size=chunk_size):
                # Hidden states
                hidden_ref = torch.randn((b, t, d), device="cuda", dtype=torch.float16, requires_grad=True)
                hidden_cand = hidden_ref.detach().clone().requires_grad_(True)

                # Labels: mix of prompt (-100) and response tokens
                labels = torch.randint(0, vocab_size, (b, t), device="cuda", dtype=torch.long)
                # First 20 tokens are prompt
                labels[:, :20] = -100

                # ----------------- Reference PyTorch Full Logits -----------------
                full_logits = lm_head(hidden_ref)
                shift_logits = full_logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()

                valid_tokens = (shift_labels != -100).sum()
                denom = float(max(valid_tokens.item(), 1))

                ref_per_token = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(b, t - 1)

                ref_loss = ref_per_token.sum() / denom

                # Upstream loss scale (simulate AMP GradScaler e.g. 65536.0)
                loss_scale = 1024.0
                (ref_grad_hidden,) = torch.autograd.grad(ref_loss * loss_scale, hidden_ref)

                # Reference per-token Shannon entropy
                logp = full_logits.float().log_softmax(-1)
                ref_entropy = -(logp.exp() * logp).sum(-1)

                # ----------------- Candidate Chunked Head Loss -----------------
                cand_loss, step = chunked_head_loss(
                    hidden_cand,
                    lm_head,
                    labels,
                    chunk_size=chunk_size,
                    num_items_in_batch=denom,
                )

                (cand_grad_hidden,) = torch.autograd.grad(cand_loss * loss_scale, hidden_cand)

                # 1. Total loss comparison
                torch.testing.assert_close(cand_loss, ref_loss, rtol=1e-3, atol=1e-3)

                # 2. Gradient on hidden states comparison (with AMP scaling!)
                torch.testing.assert_close(cand_grad_hidden, ref_grad_hidden, rtol=3e-3, atol=1e-2)

                # 3. Per-token loss comparison on shifted tokens
                # step.token_loss[:, :-1] corresponds to shift_labels
                torch.testing.assert_close(
                    step.token_loss[:, :-1], ref_per_token, rtol=1e-3, atol=1e-3
                )

                # 4. Entropy comparison for all positions
                torch.testing.assert_close(
                    step.token_entropy, ref_entropy, rtol=2e-4, atol=2e-4
                )

                # 5. Accuracy hits comparison
                ref_hits = ((full_logits.argmax(-1) == shift_for_next_token(labels)) & (shift_for_next_token(labels) != -100)).sum()
                self.assertEqual(int(step.correct.item()), int(ref_hits.item()))

    def test_amp_loss_scaler_consistency(self):
        """Validates that varying grad_loss scale produces strictly proportional gradients."""
        import torch
        import torch.nn as nn
        from solution.chunked_loss import chunked_head_loss

        b, t, d, v = 2, 32, 64, 512
        lm_head = nn.Linear(d, v, bias=False, device="cuda", dtype=torch.float16)
        labels = torch.randint(0, v, (b, t), device="cuda", dtype=torch.long)

        h1 = torch.randn((b, t, d), device="cuda", dtype=torch.float16, requires_grad=True)
        h2 = h1.detach().clone().requires_grad_(True)

        loss1, _ = chunked_head_loss(h1, lm_head, labels, chunk_size=16)
        loss2, _ = chunked_head_loss(h2, lm_head, labels, chunk_size=16)

        (g1,) = torch.autograd.grad(loss1 * 1.0, h1)
        (g2,) = torch.autograd.grad(loss2 * 65536.0, h2)

        torch.testing.assert_close(g1 * 65536.0, g2, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

