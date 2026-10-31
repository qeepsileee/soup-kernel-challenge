import os
import unittest
from unittest.mock import patch
from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA + bitsandbytes required")
class RetainedNF4Tests(unittest.TestCase):
    def test_two_live_forwards_have_exact_outputs_and_gradients(self):
        import torch
        import bitsandbytes.functional as functional
        from solution import nf4_linear
        torch.manual_seed(914)
        for shape in ((256, 1536), (1536, 1536), (8960, 1536), (1536, 8960)):
            with self.subTest(shape=shape):
                w = torch.randn(shape, device="cuda", dtype=torch.float16) * .02
                packed, state = functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
                x = torch.randn(2, 1024, shape[1], device="cuda", dtype=torch.float16)
                g = torch.randn(2, 1024, shape[0], device="cuda", dtype=torch.float16) * 8
                self.assertFalse(nf4_linear.uses_custom_gemm(x, state))

                def run(retain):
                    a = x.clone().requires_grad_(True)
                    b = (x * .5).requires_grad_(True)
                    with patch.dict(os.environ, {"SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT": str(retain)}):
                        with patch.object(nf4_linear, "dequantize_nf4",
                                          wraps=nf4_linear.dequantize_nf4) as decoder:
                            ya = nf4_linear.NF4Linear.apply(a, packed, None, state)
                            yb = nf4_linear.NF4Linear.apply(b, packed, None, state)
                            # Reverse consumption order catches unsafe buffer reuse.
                            yb.backward(g)
                            ya.backward(g)
                    return ya, yb, a.grad, b.grad, decoder.call_count

                base, candidate = run(0), run(1)
                for a, b in zip(base[:4], candidate[:4]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(base[4], 4)
                self.assertEqual(candidate[4], 2)
