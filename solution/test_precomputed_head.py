import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class PrecomputedHeadTests(unittest.TestCase):
    def test_real_vocab_under_amp(self):
        import torch
        from solution.chunked_loss import chunked_head_loss
        torch.manual_seed(911)
        head = torch.nn.Linear(64, 151936, bias=False, device="cuda", dtype=torch.float32)
        head.weight.requires_grad_(False)
        labels = torch.randint(151936, (2, 17), device="cuda")
        labels[:, :8] = -100
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                a = torch.randn(2, 17, 64, device="cuda", dtype=dtype, requires_grad=True)
                b = a.detach().clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.float16):
                    ref, rs = chunked_head_loss(a, head, labels, chunk_size=3)
                    out, step = chunked_head_loss(b, head, labels, chunk_size=16, precompute_grad=True)
                ga, = torch.autograd.grad(ref * 65536., a)
                gb, = torch.autograd.grad(out * 65536., b)
                torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(step.token_entropy, rs.token_entropy, rtol=2e-4, atol=2e-4)
                relative = (gb.float() - ga.float()).norm() / ga.float().norm()
                self.assertTrue(torch.isfinite(gb).all())
                self.assertLess(relative.item(), .005)

    def test_metrics_and_scaled_gradients(self):
        import torch
        from solution.chunked_loss import chunked_head_loss
        torch.manual_seed(910)
        head = torch.nn.Linear(128, 2048, bias=False, device="cuda", dtype=torch.float32)
        head.weight.requires_grad_(False)
        values = torch.randn(2, 129, 128, device="cuda", dtype=torch.float32)
        labels = torch.randint(2048, (2, 129), device="cuda")
        labels[:, :53] = -100
        labels[0, 91:105] = -100
        for chunk in (16, 64, 256):
            for scale in (1., 1024., 32768., 65536.):
                with self.subTest(chunk=chunk, scale=scale):
                    a = values.clone().requires_grad_(True)
                    b = values.clone().requires_grad_(True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        ref, rs = chunked_head_loss(a, head, labels, chunk_size=16)
                        out, step = chunked_head_loss(b, head, labels, chunk_size=chunk,
                                                     precompute_grad=True)
                    ga, = torch.autograd.grad(ref * scale, a)
                    gb, = torch.autograd.grad(out * scale, b)
                    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(step.token_loss, rs.token_loss, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(step.token_entropy, rs.token_entropy, rtol=2e-4, atol=2e-4)
                    self.assertEqual(step.correct.item(), rs.correct.item())
                    self.assertTrue(torch.isfinite(gb).all())
                    # Anchored FP16 dZ changes rounding; bound the full gradient
                    # error, not unstable relative errors near individual zeros.
                    relative = (gb - ga).norm() / ga.norm().clamp_min(1e-20)
                    self.assertLess(relative.item(), 0.005)
                    self.assertTrue(torch.all(gb[:, :52] == 0))
                    self.assertTrue(torch.all(gb[:, -1] == 0))

    def test_all_ignored_and_one_label(self):
        import torch
        from solution.chunked_loss import chunked_head_loss
        head = torch.nn.Linear(16, 64, bias=False, device="cuda", dtype=torch.float16)
        head.weight.requires_grad_(False)
        for count in (0, 1):
            x = torch.randn(1, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
            labels = torch.full((1, 8), -100, device="cuda")
            if count:
                labels[0, -1] = 4
            loss, _ = chunked_head_loss(x, head, labels, chunk_size=3, precompute_grad=True)
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            if not count:
                self.assertEqual(loss.item(), 0)
                self.assertTrue(torch.all(x.grad == 0))
