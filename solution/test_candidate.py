import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class CandidateTests(unittest.TestCase):
    def test_nf4_linear_forward_backward_bias_and_loss_scale(self):
        import bitsandbytes as bnb
        import bitsandbytes.functional as functional
        import torch
        from solution.nf4_linear import NF4Linear, unpatch_linear
        unpatch_linear()
        torch.manual_seed(52)
        for shape in ((256, 1536), (1536, 1536), (8960, 1536), (1536, 8960)):
            with self.subTest(shape=shape):
                w = torch.randn(shape, device="cuda", dtype=torch.float16)
                packed, state = functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
                bias = torch.randn(shape[0], device="cuda", dtype=torch.float16)
                x = torch.randn((2, 17, shape[1]), device="cuda", dtype=torch.float16)
                g = torch.randn((2, 17, shape[0]), device="cuda", dtype=torch.float16) * 8
                a = x.detach().clone().requires_grad_(True)
                ref = bnb.matmul_4bit(a, packed, state, bias=bias)
                ref.backward(g)
                b = x.detach().clone().requires_grad_(True)
                actual = NF4Linear.apply(b, packed, bias, state)
                actual.backward(g)
                torch.testing.assert_close(actual, ref, rtol=3e-3, atol=1e-2)
                torch.testing.assert_close(b.grad, a.grad, rtol=3e-3, atol=1e-2)

    def test_rmsnorm_rounding_and_backward(self):
        import torch
        from solution.rmsnorm import FrozenRMSNorm
        torch.manual_seed(53)
        for dtype in (torch.float16, torch.float32):
            for wdtype in (torch.float16, torch.float32):
                for d in (65, 1536):
                    with self.subTest(dtype=dtype, wdtype=wdtype, d=d):
                        a = torch.randn((2, 17, d), device="cuda", dtype=dtype, requires_grad=True)
                        b = a.detach().clone().requires_grad_(True)
                        w = torch.randn(d, device="cuda", dtype=wdtype)
                        af = a.float()
                        y = w * (af * torch.rsqrt(af.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
                        actual = FrozenRMSNorm.apply(b, w, 1e-6)
                        self.assertEqual(actual.dtype, y.dtype)
                        g = torch.randn_like(y) * 8
                        y.backward(g)
                        actual.backward(g)
                        tol = 5e-3 if dtype == torch.float16 else 2e-5
                        torch.testing.assert_close(actual, y, rtol=tol, atol=tol)
                        torch.testing.assert_close(b.grad, a.grad, rtol=tol, atol=tol)

    def test_entropy_tail_uniform_extreme_and_offsets(self):
        import torch
        from solution.entropy import entropy_from_logits
        torch.manual_seed(54)
        for v in (1, 17, 4097, 151936):
            for scale, offset in ((0.0, 0.0), (1.0, 0.0), (50.0, 10000.0)):
                with self.subTest(v=v, scale=scale, offset=offset):
                    x = torch.randn((2, 3, v), device="cuda", dtype=torch.float32) * scale + offset
                    logp = x.log_softmax(-1)
                    expected = -(logp.exp() * logp).sum(-1)
                    torch.testing.assert_close(entropy_from_logits(x), expected,
                                               rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
