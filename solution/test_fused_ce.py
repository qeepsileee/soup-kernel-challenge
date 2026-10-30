"""Unit tests for Triton fused Cross-Entropy + Metrics + Backward kernel."""

import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class FusedCrossEntropyTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(42)

    def test_numerical_loss_and_grad_against_pytorch(self):
        import torch
        import torch.nn.functional as F
        from solution.fused_ce import fused_cross_entropy

        test_shapes = [
            (4, 256),       # small vocabulary
            (8, 4096),      # exact single block
            (16, 5000),     # non-multiple of block size
            (32, 151936),   # full Qwen2.5 vocabulary size!
        ]

        for n, v in test_shapes:
            for dtype in (torch.float16, torch.float32):
                with self.subTest(n=n, v=v, dtype=dtype):
                    logits_ref = torch.randn((n, v), device="cuda", dtype=dtype, requires_grad=True)
                    logits_cand = logits_ref.detach().clone()

                    # Target tokens: ~30% -100, remainder valid random tokens
                    targets = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)
                    mask_ignore = torch.rand(n, device="cuda") < 0.3
                    targets[mask_ignore] = -100

                    valid_count = (targets != -100).sum().item()
                    denom = max(float(valid_count), 1.0)

                    # Reference PyTorch loss & autograd gradient
                    # Note: PyTorch cross_entropy upcasts to float32 internally
                    ref_per_token = F.cross_entropy(
                        logits_ref.float(), targets, ignore_index=-100, reduction="none"
                    )
                    ref_loss = ref_per_token.sum() / denom
                    (ref_grad,) = torch.autograd.grad(ref_loss, logits_ref)

                    # Candidate fused CE (out-of-place first to keep logits_cand intact)
                    loss, entropy, hits, cand_loss, grad = fused_cross_entropy(
                        logits_cand, targets, denom=denom, needs_grad=True, inplace_grad=False
                    )

                    # 1. Total loss comparison
                    torch.testing.assert_close(
                        cand_loss, ref_loss,
                        rtol=1e-3 if dtype == torch.float16 else 1e-5,
                        atol=1e-3 if dtype == torch.float16 else 1e-5,
                    )

                    # 2. Per-token loss comparison
                    torch.testing.assert_close(
                        loss, ref_per_token,
                        rtol=1e-3 if dtype == torch.float16 else 1e-5,
                        atol=1e-3 if dtype == torch.float16 else 1e-5,
                    )

                    # 3. Gradient comparison
                    torch.testing.assert_close(
                        grad, ref_grad.to(dtype),
                        rtol=3e-3 if dtype == torch.float16 else 1e-5,
                        atol=3e-3 if dtype == torch.float16 else 1e-5,
                    )

                    # 4. In-place gradient overwrite verification
                    logits_inplace = logits_ref.detach().clone()
                    fused_cross_entropy(
                        logits_inplace, targets, denom=denom, needs_grad=True, inplace_grad=True
                    )
                    torch.testing.assert_close(logits_inplace, grad)

    def test_entropy_against_reference(self):
        import torch
        from solution.fused_ce import fused_cross_entropy

        for v in (256, 4097, 151936):
            n = 8
            logits = torch.randn((n, v), device="cuda", dtype=torch.float32)
            targets = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)
            # Make half of the targets -100 (prompt tokens)
            targets[: n // 2] = -100

            # Mathematical Shannon entropy
            log_probs = logits.log_softmax(-1)
            ref_entropy = -(log_probs.exp() * log_probs).sum(-1)

            _, cand_entropy, _, _, _ = fused_cross_entropy(
                logits, targets, needs_grad=False
            )

            # Check that entropy matches reference for ALL positions, including -100
            torch.testing.assert_close(cand_entropy, ref_entropy, rtol=2e-5, atol=2e-5)
            self.assertTrue(torch.all(cand_entropy > 0.0))

    def test_accuracy_hits_against_reference(self):
        import torch
        from solution.fused_ce import fused_cross_entropy

        n, v = 32, 1024
        logits = torch.randn((n, v), device="cuda", dtype=torch.float32)
        # Force some positions to have predicted argmax equal to target
        preds = logits.argmax(dim=-1)
        targets = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)
        targets[:10] = preds[:10]  # first 10 match
        targets[20:] = -100        # last 12 ignored

        labelled = targets != -100
        ref_hits = ((preds == targets) & labelled).to(torch.int32)

        _, _, cand_hits, _, _ = fused_cross_entropy(
            logits, targets, needs_grad=False
        )

        torch.testing.assert_close(cand_hits, ref_hits)

    def test_all_ignore_tokens(self):
        import torch
        from solution.fused_ce import fused_cross_entropy

        n, v = 16, 512
        logits = torch.randn((n, v), device="cuda", dtype=torch.float16)
        targets = torch.full((n,), -100, device="cuda", dtype=torch.long)

        loss, entropy, hits, total_loss, grad = fused_cross_entropy(
            logits, targets, denom=1.0, needs_grad=True, inplace_grad=False
        )

        self.assertEqual(float(total_loss.item()), 0.0)
        self.assertTrue(torch.all(loss == 0.0))
        self.assertTrue(torch.all(hits == 0))
        self.assertTrue(torch.all(grad == 0.0))
        self.assertTrue(torch.all(entropy > 0.0))


if __name__ == "__main__":
    unittest.main()

