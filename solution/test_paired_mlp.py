import unittest
from solution.test_nf4 import GPU_AVAILABLE


def make_frozen_mlp(dim=1536, hidden=8960):
    import bitsandbytes as bnb
    import torch
    from types import SimpleNamespace
    module = torch.nn.Module()
    module.config = SimpleNamespace(hidden_act="silu")
    for name, ins, outs in (("gate_proj", dim, hidden), ("up_proj", dim, hidden),
                            ("down_proj", hidden, dim)):
        layer = bnb.nn.Linear4bit(ins, outs, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4")
        # compute_dtype controls GEMM, not the source dtype recorded in
        # quant_state. Quantize FP16 source weights like the actual T4 model.
        layer.weight = bnb.nn.Params4bit(torch.randn(outs, ins, dtype=torch.float16) * .02,
            requires_grad=False, compress_statistics=True, quant_type="nf4")
        layer = layer.cuda()
        # Simulate the already-initialized frozen projections of the live model.
        layer.compute_type_is_set = True
        setattr(module, name, layer)
    return module


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class PairedMLPTests(unittest.TestCase):
    def test_full_mlp_forward_backward_and_rng(self):
        import torch
        from solution.compressed_mlp import CompressedSwiGLU
        from solution.mlp_bridge import bridged_mlp
        from solution.paired_mlp import paired_mlp
        from solution.nf4_linear import patch_linear, unpatch_linear
        torch.manual_seed(930)
        patch_linear()
        try:
            for dim, hidden in ((64, 192), (1536, 8960)):
                module = make_frozen_mlp(dim, hidden)
                for name in ("gate_proj", "up_proj", "down_proj"):
                    self.assertEqual(getattr(module, name).weight.quant_state.dtype, torch.float16)
                for length in (256, 257):
                    with self.subTest(dim=dim, hidden=hidden, length=length):
                        # Noncontiguous input + partial Triton tile are intentional.
                        values = torch.randn(2, dim, length, device="cuda", dtype=torch.float16).transpose(1, 2)
                        a = values.detach().clone().requires_grad_(True)
                        b = values.detach().clone().requires_grad_(True)
                        grad = torch.randn_like(values) * 8
                        ref = bridged_mlp(module, a, activation=CompressedSwiGLU.apply)
                        rng = torch.cuda.get_rng_state().clone()
                        actual = paired_mlp(module, b)
                        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
                        ref.backward(grad)
                        actual.backward(grad)
                        self.assertEqual(actual.dtype, torch.float16)
                        torch.testing.assert_close(actual, ref, rtol=3e-3, atol=3e-3)
                        self.assertTrue(torch.isfinite(b.grad).all())
                        relative = (b.grad.float() - a.grad.float()).norm() / a.grad.float().norm().clamp_min(1e-20)
                        self.assertLess(relative.item(), .01)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        finally:
            unpatch_linear()

    def test_unsupported_input_rejected(self):
        import torch
        from solution.paired_mlp import paired_mlp
        module = make_frozen_mlp(64, 192)
        for dtype, rows in ((torch.float32, 512), (torch.float16, 17)):
            with self.subTest(dtype=dtype, rows=rows):
                x = torch.randn(rows, 64, device="cuda", dtype=dtype)
                with self.assertRaises(ValueError):
                    paired_mlp(module, x)
