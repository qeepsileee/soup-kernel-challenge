"""CUDA validation for fused backward; compilation and runtime require T4."""

import unittest

from solution.test_nf4 import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "CUDA + torch + triton + bitsandbytes required")
class FusedInputGradTests(unittest.TestCase):
    def setUp(self):
        import torch
        from solution.nf4 import unpatch_dequantize
        from solution.nf4_matmul import unpatch_input_grad

        unpatch_input_grad()
        unpatch_dequantize()
        torch.manual_seed(47)

    def tearDown(self):
        from solution.nf4_matmul import unpatch_input_grad

        unpatch_input_grad()

    def test_fused_dx_all_shapes_and_tails(self):
        import bitsandbytes.functional as functional
        import torch
        from solution.nf4_format import qwen_projection_shapes
        from solution.nf4_matmul import input_grad_nf4

        for shape in sorted(set(qwen_projection_shapes().values()) | {(33, 65)}):
            for nested in (False, True):
                for dtype in (torch.float16, torch.float32):
                    with self.subTest(shape=shape, nested=nested, dtype=dtype):
                        w = torch.randn(shape, device="cuda", dtype=dtype)
                        packed, state = functional.quantize_4bit(
                            w, quant_type="nf4", compress_statistics=nested)
                        # Strided dY with a tail in M; loss scale already applied.
                        g = torch.randn((2, shape[0], 17), device="cuda",
                                        dtype=torch.float16).transpose(1, 2) * 2
                        dense = functional.dequantize_4bit(packed, state).to(g.dtype)
                        expected = torch.matmul(g, dense)
                        actual = input_grad_nf4(g, packed, state)
                        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-2)

    def test_dense_dx_all_shapes_and_tails(self):
        import bitsandbytes.functional as functional
        import torch
        from solution.nf4_format import qwen_projection_shapes
        from solution.nf4_matmul import input_grad_dense

        for shape in sorted(set(qwen_projection_shapes().values()) | {(33, 65)}):
            for nested in (False, True):
                for dtype in (torch.float16, torch.float32):
                    with self.subTest(shape=shape, nested=nested, dtype=dtype):
                        w = torch.randn(shape, device="cuda", dtype=dtype)
                        packed, state = functional.quantize_4bit(
                            w, quant_type="nf4", compress_statistics=nested)
                        g = torch.randn((2, shape[0], 17), device="cuda",
                                        dtype=torch.float16).transpose(1, 2) * 2
                        expected = g @ functional.dequantize_4bit(packed, state).to(g.dtype)
                        torch.testing.assert_close(input_grad_dense(g, packed, state), expected,
                                                   rtol=3e-3, atol=1e-2)

    def test_bnb_autograd_patch_with_lora_and_dropout(self):
        import bitsandbytes as bnb
        import bitsandbytes.functional as functional
        import torch
        from bitsandbytes.autograd._functions import MatMul4Bit
        from solution.nf4_matmul import patch_input_grad, unpatch_input_grad

        w = torch.randn((256, 128), device="cuda", dtype=torch.float16)
        packed, state = functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
        x = torch.randn((2, 17, 128), device="cuda", dtype=torch.float16)
        a = torch.randn((16, 128), device="cuda", dtype=torch.float16) * 0.01
        b = torch.randn((256, 16), device="cuda", dtype=torch.float16) * 0.01
        upstream = torch.randn((2, 17, 256), device="cuda", dtype=torch.float16) * 16

        def run(patched):
            unpatch_input_grad()
            if patched:
                patch_input_grad()
            xi, ai, bi = [v.detach().clone().requires_grad_(True) for v in (x, a, b)]
            torch.manual_seed(123)
            base = bnb.matmul_4bit(xi, packed, state)
            low = torch.nn.functional.dropout(xi, p=0.05, training=True)
            low = torch.nn.functional.linear(low, ai)
            y = base + torch.nn.functional.linear(low, bi) * 2
            y.backward(upstream)
            return y.detach(), xi.grad, ai.grad, bi.grad

        reference = run(False)
        candidate = run(True)
        self.assertTrue(getattr(MatMul4Bit.backward, "_soup_fused_nf4_dx", False))
        self.assertEqual(MatMul4Bit.backward._nf4_dx_mode, "decode_then_gemm")
        # Forward and LoRA parameter gradients are unchanged; only dX uses GEMM candidate.
        for i in (0, 2, 3):
            torch.testing.assert_close(candidate[i], reference[i], rtol=0, atol=0)
        torch.testing.assert_close(candidate[1], reference[1], rtol=3e-3, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
