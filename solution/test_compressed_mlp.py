import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class CompressedMLPTests(unittest.TestCase):
    def test_exact_forward_bounded_gradient_and_saved_bytes(self):
        import torch
        from solution.compressed_mlp import CompressedSwiGLU
        from solution.swiglu import swiglu
        torch.manual_seed(801)
        for shape in ((17,), (1031,), (2, 17, 8960), (4, 512, 8960)):
            with self.subTest(shape=shape):
                base_g = torch.randn(shape, device="cuda", dtype=torch.float16)
                base_u = torch.randn_like(base_g)
                dy = torch.randn_like(base_g) * 8
                saved = [0, 0]

                def run(fn, slot):
                    g = base_g.detach().clone().requires_grad_(True)
                    u = base_u.detach().clone().requires_grad_(True)

                    def pack(t):
                        saved[slot] += t.numel() * t.element_size()
                        return t

                    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                        out = fn(g, u)
                    out.backward(dy)
                    return out.detach(), g.grad, u.grad

                ref, rg, ru = run(swiglu, 0)
                actual, ag, au = run(CompressedSwiGLU.apply, 1)
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                for a, r in ((ag, rg), (au, ru)):
                    self.assertTrue(torch.all(torch.isfinite(a)))
                    rel = torch.linalg.vector_norm(a.float() - r.float()) / torch.linalg.vector_norm(r.float())
                    self.assertLess(rel.item(), 0.02)
                self.assertLess(saved[1], saved[0])
                if base_g.numel() >= 1024:
                    self.assertLess(saved[1], saved[0] * 0.55)

    def test_zero_and_partial_group(self):
        import torch
        from solution.compressed_mlp import CompressedSwiGLU
        for n in (1, 63, 64, 65):
            g = torch.zeros(n, device="cuda", dtype=torch.float16, requires_grad=True)
            u = torch.zeros_like(g, requires_grad=True)
            y = CompressedSwiGLU.apply(g, u)
            y.sum().backward()
            self.assertTrue(torch.all(y == 0))
            self.assertTrue(torch.all(g.grad == 0))
            self.assertTrue(torch.all(u.grad == 0))


if __name__ == "__main__":
    unittest.main()
