import unittest
from solution.cuda_test_support import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "PyTorch + Triton + CUDA required")
class LosslessSavedTests(unittest.TestCase):
    def test_mask_bits_round_trip_and_tail(self):
        import torch
        from solution.lossless_saved import pack_mask, unpack_saved, PackedMask
        torch.manual_seed(950)
        for n in (0, 7, 4095, 4096, 4097, 1000001):
            for mode in ("zero", "one", "random"):
                with self.subTest(n=n, mode=mode):
                    mask = (torch.zeros(n, device="cuda", dtype=torch.bool) if mode == "zero" else
                            torch.ones(n, device="cuda", dtype=torch.bool) if mode == "one" else
                            torch.rand(n, device="cuda") > .05)
                    saved = pack_mask(mask)
                    torch.testing.assert_close(unpack_saved(saved), mask, rtol=0, atol=0)
                    if n >= 4096:
                        self.assertIsInstance(saved, PackedMask)
                        self.assertEqual(saved.packed.numel(), (n+7)//8)

    def test_dropout_forward_gradient_and_rng_exact(self):
        import torch
        from solution.lossless_saved import patch_lora_dropouts, unpatch_lora_dropouts
        model = torch.nn.Module()
        model.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
        dropout = model.lora_dropout["default"]
        x = torch.randn(2, 129, 1536, device="cuda", dtype=torch.float16)
        grad = torch.randn_like(x) * 8
        def run():
            value = x.clone().requires_grad_(True)
            torch.cuda.manual_seed_all(951)
            y = dropout(value)
            rng = torch.cuda.get_rng_state().clone()
            y.backward(grad)
            return y.detach(), value.grad, rng
        ref = run()
        self.assertEqual(patch_lora_dropouts(model), 1)
        try:
            actual = run()
            for a, b in zip(actual, ref):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            dropout.eval()
            torch.testing.assert_close(dropout(input=x), x, rtol=0, atol=0)
        finally:
            unpatch_lora_dropouts(model)

    def test_kv_snapshot_views_and_source_mutation(self):
        import torch
        from solution.lossless_saved import KVContext, SavedKV, unpack_saved
        source = torch.randn(2, 33, 2, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
        expanded = source.repeat_interleave(6, dim=1)
        context = KVContext()
        context.register(source, expanded, 6)
        view = expanded.transpose(1, 2)
        saved = context.pack(view)
        self.assertIsInstance(saved, SavedKV)
        self.assertEqual(saved.compact.numel()*6, expanded.numel())
        context.sources.clear()
        source.add_(10)
        restored = unpack_saved(saved)
        self.assertEqual(restored.stride(), view.stride())
        torch.testing.assert_close(restored, view, rtol=0, atol=0)

    def test_real_peft_adapter_gradients_and_parameter_identity(self):
        import torch
        from solution.test_lora_fusion import make_lora_layer
        from solution.lossless_saved import patch_lora_dropouts, unpatch_lora_dropouts
        for out in (256, 1536):
            with self.subTest(out=out):
                layer = make_lora_layer(out)
                identity = {n: id(p) for n, p in layer.named_parameters()}
                x = torch.randn(2, 129, 1536, device="cuda", dtype=torch.float16)
                grad = torch.randn(2, 129, out, device="cuda", dtype=torch.float16) * 8
                def run():
                    layer.zero_grad(set_to_none=True)
                    value = x.clone().requires_grad_(True)
                    torch.cuda.manual_seed_all(953)
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = layer(value)
                    rng = torch.cuda.get_rng_state().clone()
                    y.backward(grad)
                    return (y.detach(), value.grad, layer.lora_A["default"].weight.grad.clone(),
                            layer.lora_B["default"].weight.grad.clone(), rng)
                ref = run()
                self.assertEqual(patch_lora_dropouts(layer), 1)
                try:
                    actual = run()
                    torch.testing.assert_close(actual[0], ref[0], rtol=0, atol=0)
                    for i in (1, 2, 3):
                        torch.testing.assert_close(actual[i], ref[i], rtol=2e-5, atol=2e-5)
                    self.assertTrue(torch.equal(actual[4], ref[4]))
                    self.assertEqual(identity, {n: id(p) for n, p in layer.named_parameters()})
                finally:
                    unpatch_lora_dropouts(layer)

    def test_native_sdpa_forward_backward_and_real_codec_use(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from solution.lossless_saved import (patch_lossless_attention, unpatch_lossless_attention,
                                             KVContext, SavedKV)
        unpatch_lossless_attention()
        original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        module = torch.nn.Module()
        module.config = SimpleNamespace(model_type="qwen2")
        module.num_key_value_groups, module.is_causal = 6, True
        patch_lossless_attention()
        candidate = ALL_ATTENTION_FUNCTIONS["sdpa"]
        torch.manual_seed(952)
        try:
            # Aligned bias rows keep these cases on the T4 memory-efficient
            # route. Tail/view restoration is tested separately above.
            for length in (32, 128, 512):
                for dtype in (torch.bool, torch.float16):
                    with self.subTest(length=length, mask_dtype=dtype):
                        values = [torch.randn(2, h, length, 128, device="cuda", dtype=torch.float16)
                                  for h in (12, 2, 2)]
                        allowed = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
                        allowed[:, -3:] = False
                        mask = allowed[None, None]
                        if dtype != torch.bool:
                            mask = torch.where(mask, 0., -float("inf")).to(dtype)
                        grad = torch.randn(2, length, 12, 128, device="cuda", dtype=torch.float16)
                        count = [0]
                        real_pack = KVContext.pack
                        def counted(ctx, tensor):
                            result = real_pack(ctx, tensor)
                            count[0] += isinstance(result, SavedKV)
                            return result
                        def run(fn):
                            inputs = [x.clone().requires_grad_(True) for x in values]
                            y, _ = fn(module, *inputs, mask, dropout=0.)
                            y.backward(grad)
                            return y.detach(), [v.grad for v in inputs]
                        ref, rg = run(original)
                        with patch.object(KVContext, "pack", counted):
                            actual, ag = run(candidate)
                        self.assertGreaterEqual(count[0], 2, "must compress actual saved K/V, not silently fall back")
                        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                        for a, b in zip(ag, rg):
                            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
        finally:
            unpatch_lossless_attention()
