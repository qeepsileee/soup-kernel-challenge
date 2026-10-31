import unittest
from solution.cuda_test_support import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "PyTorch + Triton + CUDA required")
class NF4SavedMLPTests(unittest.TestCase):
    def test_exact_forward_and_measured_backward_error_and_storage(self):
        import torch
        from solution.compressed_mlp import CompressedSwiGLU
        from solution.nf4_saved_mlp import NF4SavedSwiGLU
        from solution.swiglu import swiglu
        torch.manual_seed(940)
        for shape in ((65,), (2, 17, 8960), (4, 512, 8960)):
            for amplitude in (1., 4.):
                with self.subTest(shape=shape, amplitude=amplitude):
                    gate = torch.randn(shape, device="cuda", dtype=torch.float16) * amplitude
                    up = torch.randn_like(gate)
                    grad = torch.randn_like(gate) * 8

                    def run(fn):
                        g, u = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
                        saved = [0]
                        def pack(tensor):
                            saved[0] += tensor.numel() * tensor.element_size()
                            return tensor
                        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                            out = fn(g, u)
                        out.backward(grad)
                        return out.detach(), g.grad, u.grad, saved[0]

                    ref, rg, ru, _ = run(swiglu)
                    int8, _, _, bytes8 = run(CompressedSwiGLU.apply)
                    actual, ag, au, bytes4 = run(NF4SavedSwiGLU.apply)
                    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                    torch.testing.assert_close(actual, int8, rtol=0, atol=0)
                    for a, r in ((ag, rg), (au, ru)):
                        self.assertTrue(torch.isfinite(a).all())
                        relative = (a.float() - r.float()).norm() / r.float().norm().clamp_min(1e-20)
                        print(f"NF4 saved {shape} amplitude={amplitude}: gradient L2 error={relative.item():.4f}")
                        # Sanity bound ONLY, not a task-quality guarantee. The
                        # actual eval/curve tolerance remains the official one.
                        self.assertLess(relative.item(), .25)
                    self.assertLess(bytes4, bytes8)
                    if gate.numel() >= 1024:
                        self.assertLess(bytes4, bytes8 * .60)

    def test_zero_and_partial_groups(self):
        import torch
        from solution.nf4_saved_mlp import NF4SavedSwiGLU
        for n in (0, 1, 63, 64, 65):
            gate = torch.zeros(n, device="cuda", dtype=torch.float16, requires_grad=True)
            up = torch.zeros_like(gate, requires_grad=True)
            out = NF4SavedSwiGLU.apply(gate, up)
            out.sum().backward()
            self.assertTrue(torch.all(out == 0))
            self.assertTrue(torch.all(gate.grad == 0))
            self.assertTrue(torch.all(up.grad == 0))
