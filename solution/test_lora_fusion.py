import unittest
from solution.test_nf4 import GPU_AVAILABLE


def make_lora_layer(out_features):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit
    base = bnb.nn.Linear4bit(1536, out_features, bias=True,
                            compute_dtype=torch.float16, compress_statistics=True,
                            quant_type="nf4")
    base.weight = bnb.nn.Params4bit(torch.randn(out_features, 1536) * 0.02,
                                  requires_grad=False, compress_statistics=True,
                                  quant_type="nf4")
    base = base.cuda()
    base.bias.requires_grad_(False)
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)
    layer = Linear4bit(base, "default", config=config, r=16, lora_alpha=32).cuda()
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.02)
        layer.lora_B["default"].weight.normal_(std=0.02)
    return layer


@unittest.skipUnless(GPU_AVAILABLE, "CUDA dependencies required")
class LoRAFusionTests(unittest.TestCase):
    def test_epilogue_rounding_and_scaled_backward(self):
        import torch
        from solution.lora_fusion import LoRAAdd
        for bdtype in (torch.float16, torch.float32):
            for adtype in (torch.float16, torch.float32):
                for convert in (False, True):
                    with self.subTest(base=bdtype, adapter=adtype, convert=convert):
                        base = torch.randn((2, 17, 256), device="cuda", dtype=bdtype)
                        adapter = torch.randn_like(base, dtype=adtype)
                        a, b = [t.detach().clone().requires_grad_(True) for t in (base, adapter)]
                        c, d = [t.detach().clone().requires_grad_(True) for t in (base, adapter)]
                        scaled = b * 2.0
                        if convert:
                            scaled = scaled.to(bdtype)
                        ref = a.clone() + scaled
                        actual = LoRAAdd.apply(c, d, 2.0, convert)
                        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                        upstream = torch.randn_like(ref) * 16
                        ref.backward(upstream)
                        actual.backward(upstream)
                        torch.testing.assert_close(c.grad, a.grad, rtol=0, atol=0)
                        torch.testing.assert_close(d.grad, b.grad, rtol=0, atol=0)

    def test_real_peft_x_A_B_gradients_and_dropout_rng(self):
        import torch
        from peft.tuners.lora.bnb import Linear4bit
        from solution.lora_fusion import patch_lora, unpatch_lora
        unpatch_lora()
        original = Linear4bit.forward
        patch_lora()
        patched = Linear4bit.forward
        patch_lora()
        self.assertIs(patched, Linear4bit.forward)
        try:
            for out in (256, 1536):
                layer = make_lora_layer(out)
                before = {n: (id(p), p.shape) for n, p in layer.named_parameters()}
                for amp in (False, True):
                    for dtype in (torch.float16, torch.float32):
                        with self.subTest(out=out, amp=amp, dtype=dtype):
                            x = torch.randn((2, 17, 1536), device="cuda", dtype=dtype)
                            upstream = torch.randn((2, 17, out), device="cuda", dtype=torch.float32) * 8

                            def run(forward):
                                layer.zero_grad(set_to_none=True)
                                value = x.detach().clone().requires_grad_(True)
                                torch.cuda.manual_seed_all(333)
                                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                                    y = forward(layer, value)
                                rng = torch.cuda.get_rng_state().clone()
                                y.backward(upstream.to(y.dtype))
                                return (y.detach(), value.grad,
                                        layer.lora_A["default"].weight.grad.clone(),
                                        layer.lora_B["default"].weight.grad.clone(), rng)

                            ref = run(original)
                            actual = run(patched)
                            torch.testing.assert_close(actual[0], ref[0], rtol=0, atol=0)
                            for i in (1, 2, 3):
                                torch.testing.assert_close(actual[i], ref[i], rtol=2e-5, atol=2e-5)
                            self.assertTrue(torch.equal(actual[4], ref[4]))
                self.assertEqual(before, {n: (id(p), p.shape) for n, p in layer.named_parameters()})
                # Disabled adapters must retain the original fallback semantics.
                layer.enable_adapters(False)
                x = torch.randn((2, 17, 1536), device="cuda")
                torch.testing.assert_close(patched(layer, x), original(layer, x), rtol=0, atol=0)
        finally:
            unpatch_lora()


if __name__ == "__main__":
    unittest.main()
