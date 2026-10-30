import unittest
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class MLPBridgeTests(unittest.TestCase):
    def test_forward_backward_and_smaller_saved_activations(self):
        import bitsandbytes as bnb
        import torch
        from types import SimpleNamespace
        from solution.mlp_bridge import bridged_mlp, can_bridge
        from solution.nf4_linear import patch_linear, unpatch_linear
        from solution.swiglu import swiglu

        torch.manual_seed(2026)
        module = torch.nn.Module()
        module.config = SimpleNamespace(hidden_act="silu")
        for name, ins, outs in (("gate_proj", 1536, 8960),
                                ("up_proj", 1536, 8960),
                                ("down_proj", 8960, 1536)):
            layer = bnb.nn.Linear4bit(ins, outs, bias=False, compute_dtype=torch.float16,
                                     compress_statistics=True, quant_type="nf4")
            layer.weight = bnb.nn.Params4bit(torch.randn(outs, ins) * 0.02,
                requires_grad=False, compress_statistics=True, quant_type="nf4")
            setattr(module, name, layer.cuda())
        patch_linear()
        try:
            for length in (17, 129, 512):
                with self.subTest(length=length):
                    a = torch.randn((2, length, 1536), device="cuda", dtype=torch.float32,
                                    requires_grad=True)
                    b = a.detach().clone().requires_grad_(True)
                    grad = torch.randn_like(a) * 8
                    self.assertTrue(can_bridge(module, a))
                    sizes = [0, 0]

                    def pack(value, slot):
                        # Count only large intermediate activations, not parameters.
                        if value.ndim == 3 and value.shape[-1] == 8960:
                            sizes[slot] += value.numel() * value.element_size()
                        return value

                    with torch.autograd.graph.saved_tensors_hooks(lambda v: pack(v, 0), lambda v: v):
                        ref = module.down_proj(swiglu(module.gate_proj(a), module.up_proj(a)))
                    ref.backward(grad)
                    with torch.autograd.graph.saved_tensors_hooks(lambda v: pack(v, 1), lambda v: v):
                        actual = bridged_mlp(module, b)
                    actual.backward(grad)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                    torch.testing.assert_close(b.grad, a.grad, rtol=2e-5, atol=2e-5)
                    self.assertLess(sizes[1], sizes[0])
        finally:
            unpatch_linear()


if __name__ == "__main__":
    unittest.main()
