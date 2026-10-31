import unittest
from solution.cuda_test_support import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class ParallelCETests(unittest.TestCase):
    def test_loss_entropy_grad_and_amp_before_rounding(self):
        import torch
        from solution.parallel_ce import parallel_cross_entropy
        torch.manual_seed(607)
        for v in (1, 17, 4097, 151936):
            for dtype in (torch.float16, torch.float32):
                with self.subTest(v=v, dtype=dtype):
                    z = torch.randn((8, v), device="cuda", dtype=dtype)
                    y = torch.randint(v, (8,), device="cuda")
                    y[::3] = -100
                    denom = max(int((y != -100).sum()), 1)
                    scale = torch.tensor(1024., device="cuda", dtype=torch.float32)
                    r = z.float().detach().requires_grad_(True)
                    losses = torch.nn.functional.cross_entropy(r, y, reduction="none")
                    lp = r.log_softmax(-1)
                    ent = -(lp.exp() * lp).sum(-1)
                    expected_grad, = torch.autograd.grad(losses.sum() / denom * scale, r)
                    out = torch.empty_like(z, dtype=torch.float32)
                    norms = torch.empty(8, device="cuda", dtype=torch.float32)
                    loss, entropy, hit, total, grad = parallel_cross_entropy(
                        z, y, denom=denom, inplace_grad=False, grad_out=out,
                        lse_out=norms, upstream_scale=scale)
                    torch.testing.assert_close(loss, losses, rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(entropy, ent, rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(grad, expected_grad, rtol=3e-5, atol=3e-5)
                    torch.testing.assert_close(total, losses.sum() / denom, rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(norms, z.float().logsumexp(-1), rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(hit, ((z.argmax(-1) == y) & (y != -100)).int())

    def test_inplace_strides_and_ignored_rows(self):
        import torch
        from solution.parallel_ce import parallel_cross_entropy
        torch.manual_seed(608)
        z = torch.randn((9, 10002), device="cuda", dtype=torch.float16)[:, ::2]
        before = z.clone()
        target = torch.randint(5001, (18,), device="cuda")[::2]
        target[::2] = -100
        denom = max(int((target != -100).sum()), 1)
        reference = before.float().requires_grad_(True)
        expected, = torch.autograd.grad(
            torch.nn.functional.cross_entropy(reference, target, reduction="sum") / denom,
            reference)
        result = parallel_cross_entropy(z, target, denom=denom, inplace_grad=True)
        self.assertEqual(result[4].data_ptr(), z.data_ptr())
        torch.testing.assert_close(z, expected.half(), rtol=1e-3, atol=6e-8)
        self.assertTrue(torch.all(z[::2] == 0))

    def test_uniform_ties_and_high_offset(self):
        import torch
        from solution.parallel_ce import parallel_cross_entropy
        for v in (4097, 151936):
            for random in (False, True):
                z = (torch.randn((4, v), device="cuda") * 4 if random else
                     torch.zeros((4, v), device="cuda")) + 10000
                y = torch.tensor([0, 0, -100, v - 1], device="cuda")
                lp = z.log_softmax(-1)
                expected_entropy = -(lp.exp() * lp).sum(-1)
                result = parallel_cross_entropy(z, y, denom=3, inplace_grad=False)
                torch.testing.assert_close(result[0], torch.nn.functional.cross_entropy(
                    z, y, reduction="none"), rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(result[1], expected_entropy, rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(result[2], ((z.argmax(-1) == y) & (y != -100)).int())

    def test_all_ignore_metrics_only_and_no_input_mutation(self):
        import torch
        from solution.parallel_ce import parallel_cross_entropy
        z = torch.randn((4, 151936), device="cuda")
        before = z.clone()
        y = torch.full((4,), -100, device="cuda", dtype=torch.long)
        result = parallel_cross_entropy(z, y, needs_grad=False)
        torch.testing.assert_close(z, before, rtol=0, atol=0)
        self.assertIsNone(result[4])
        self.assertEqual(result[3].item(), 0.)
        self.assertTrue(torch.all(torch.isfinite(result[1]) & (result[1] > 0)))


if __name__ == "__main__":
    unittest.main()
