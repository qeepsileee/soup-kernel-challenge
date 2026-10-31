import unittest
from solution.cuda_test_support import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "PyTorch + Triton + CUDA required")
class RuntimeTrialTests(unittest.TestCase):
    def test_real_qwen_model_default_positions_dispatch_rotary(self):
        import torch
        from unittest.mock import patch
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        from solution.rotary import Rotary, patch_rotary, unpatch_rotary
        unpatch_rotary()
        config = Qwen2Config(vocab_size=64, hidden_size=256, intermediate_size=512,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
        config._attn_implementation = "sdpa"
        model = Qwen2Model(config).cuda().half().train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.float16) * .2
        grad = torch.randn_like(x) * 8

        def run():
            value = x.clone().requires_grad_(True)
            y = model(inputs_embeds=value, use_cache=False).last_hidden_state
            y.backward(grad)
            return y.detach(), value.grad

        ref = run()
        try:
            patch_rotary()
            with patch.object(Rotary, "apply", wraps=Rotary.apply) as spy:
                actual = run()
            self.assertEqual(spy.call_count, 1, "the normal model must launch the own RoPE path")
            self.assertEqual(tuple(spy.call_args.args[2].shape), (1, 32, 128))
            torch.testing.assert_close(actual[0], ref[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], ref[1], rtol=1e-3, atol=1e-3)
        finally:
            unpatch_rotary()

    def test_rotary_batch_broadcast_forward_and_gradient_exact(self):
        import torch
        from unittest.mock import patch
        from transformers.models.qwen2 import modeling_qwen2
        from solution.rotary import Rotary, patch_rotary, unpatch_rotary, supported
        unpatch_rotary()
        original = modeling_qwen2.apply_rotary_pos_emb
        patch_rotary()
        candidate = modeling_qwen2.apply_rotary_pos_emb
        try:
            for dtype in (torch.float16, torch.float32):
                for cb, sb in ((1, 1), (1, 2), (2, 1)):
                    for length in (17, 512):
                        with self.subTest(dtype=dtype, cb=cb, sb=sb, length=length):
                            q, k = [torch.randn(2, length, h, 128, device="cuda", dtype=dtype)
                                    .transpose(1, 2) for h in (12, 2)]
                            cos = torch.randn(cb, length, 128, device="cuda", dtype=dtype)
                            sin = torch.randn(sb, length, 128, device="cuda", dtype=dtype)
                            grads = [torch.randn_like(value) * 8 for value in (q, k)]

                            def run(fn):
                                inputs = [value.clone().requires_grad_(True) for value in (q, k)]
                                outputs = fn(*inputs, cos, sin)
                                torch.autograd.backward(outputs, grads)
                                return outputs, [value.grad for value in inputs]

                            self.assertTrue(supported(q, k, cos, sin))
                            ref, rg = run(original)
                            with patch.object(Rotary, "apply", wraps=Rotary.apply) as spy:
                                actual, ag = run(candidate)
                            self.assertEqual(spy.call_count, 1)
                            for a, b in zip((*actual, *ag), (*ref, *rg)):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            unpatch_rotary()

    def test_half_storage_comparison_preserves_values_and_lowers_setup_peak(self):
        import torch
        from solution.half_storage import convert_frozen_exact
        for values in ([0., -0., 1., 65504., 2**-24, float("inf")],
                       [0.12345679], [65520.], [2**-25], [1e-40], [float("nan")]):
            source = torch.tensor(values, device="cuda", dtype=torch.float32)
            half = source.half()
            self.assertEqual(torch.equal(source, half), torch.equal(source, half.float()))

        source = torch.full((4 * 1024 * 1024,), 1.5, device="cuda", dtype=torch.float32)

        def measure(candidate):
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(source.clone(), requires_grad=False)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            if candidate:
                self.assertEqual(convert_frozen_exact(model)["parameters"], 1)
            else:
                half = model.weight.detach().half()
                self.assertTrue(torch.equal(model.weight.detach(), half.float()))
                model.weight.data = half
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            torch.testing.assert_close(model.weight, source.half(), rtol=0, atol=0)
            return peak

        old_peak, new_peak = measure(False), measure(True)
        self.assertLess(new_peak, old_peak - source.numel() * source.element_size() * .75)

    def test_actual_qwen_attention_composes_all_saved_hooks(self):
        import torch
        from unittest.mock import patch
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        from solution.test_lora_fusion import make_lora_layer
        from solution.lossless_saved import patch_lossless_attention, unpatch_lossless_attention
        from solution.fast_dropout import (DropoutInputContext, SavedDropoutInput,
                                           patch_fast_lora_dropouts, unpatch_fast_lora_dropouts,
                                           patch_shared_lora_inputs, unpatch_shared_lora_inputs)
        from solution.rotary import patch_rotary, unpatch_rotary
        unpatch_rotary()
        unpatch_lossless_attention()
        unpatch_shared_lora_inputs()
        config = Qwen2Config(hidden_size=1536, intermediate_size=8960, num_hidden_layers=1,
                             num_attention_heads=12, num_key_value_heads=2)
        config._attn_implementation = "sdpa"
        model = Qwen2Attention(config, layer_idx=0).cuda().half()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.q_proj, model.v_proj = make_lora_layer(1536), make_lora_layer(256)
        x = torch.randn(2, 32, 1536, device="cuda", dtype=torch.float16) * .2
        cos = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16)
        sin = torch.randn_like(cos)
        allowed = torch.ones(32, 32, device="cuda", dtype=torch.bool).tril()[None, None]
        allowed[..., -3:] = False
        mask = torch.where(allowed, 0., -float("inf")).to(torch.float16)
        grad = torch.randn_like(x) * 8
        identity = {name: id(p) for name, p in model.named_parameters()}

        def run():
            model.zero_grad(set_to_none=True)
            value = x.clone().requires_grad_(True)
            torch.cuda.manual_seed_all(965)
            with torch.autocast("cuda", dtype=torch.float16):
                y, _ = model(value, position_embeddings=(cos, sin), attention_mask=mask)
            rng = torch.cuda.get_rng_state().clone()
            y.backward(grad)
            params = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            return y.detach(), value.grad, params, rng

        ref = run()
        count = [0]
        real_pack = DropoutInputContext.pack

        def counted(context, tensor):
            result = real_pack(context, tensor)
            count[0] += isinstance(result, SavedDropoutInput)
            return result

        try:
            patch_lossless_attention()
            patch_rotary()
            patch_shared_lora_inputs()
            self.assertEqual(patch_fast_lora_dropouts(model), 2)
            with patch.object(DropoutInputContext, "pack", counted):
                actual = run()
            self.assertGreaterEqual(count[0], 2)
            torch.testing.assert_close(actual[0], ref[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], ref[1], rtol=1e-3, atol=1e-3)
            for name in ref[2]:
                torch.testing.assert_close(actual[2][name], ref[2][name], rtol=1e-3, atol=1e-3)
            self.assertTrue(torch.equal(actual[3], ref[3]))
            self.assertEqual(identity, {name: id(p) for name, p in model.named_parameters()})
        finally:
            unpatch_fast_lora_dropouts(model)
            unpatch_shared_lora_inputs()
            unpatch_rotary()
            unpatch_lossless_attention()

    def test_rotary_strides_tails_rounding_and_scaled_backward(self):
        import torch
        from transformers.models.qwen2 import modeling_qwen2
        from solution.rotary import Rotary, supported, unpatch_rotary
        unpatch_rotary()
        original = modeling_qwen2.apply_rotary_pos_emb
        torch.manual_seed(960)
        for dtype in (torch.float16, torch.float32):
            for length in (17, 128, 379, 512):
                for transposed in (False, True):
                    with self.subTest(dtype=dtype, length=length, transposed=transposed):
                        q, k = [torch.randn(2, length, h, 128, device="cuda", dtype=dtype).transpose(1, 2)
                                for h in (12, 2)]
                        if not transposed:
                            q, k = q.contiguous(), k.contiguous()
                        # Non-symmetric halves catch incorrect reuse of sin/cos
                        # in backward; real Qwen position embeddings are a subset.
                        cos = torch.randn(2, length, 128, device="cuda", dtype=dtype)
                        sin = torch.randn_like(cos)
                        grads = [torch.randn_like(x) * 8 for x in (q, k)]

                        def run(fn):
                            inputs = [x.detach().clone().requires_grad_(True) for x in (q, k)]
                            outputs = fn(*inputs, cos, sin)
                            torch.autograd.backward(outputs, grads)
                            return outputs, [x.grad for x in inputs]

                        self.assertTrue(supported(q, k, cos, sin))
                        ref, rg = run(original)
                        actual, ag = run(Rotary.apply)
                        for a, b in zip((*actual, *ag), (*ref, *rg)):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_rotary_patch_fallback_and_unused_key_gradient(self):
        import torch
        from unittest.mock import patch
        from transformers.models.qwen2 import modeling_qwen2
        from solution.rotary import Rotary, patch_rotary, unpatch_rotary
        unpatch_rotary()
        original = modeling_qwen2.apply_rotary_pos_emb
        patch_rotary()
        patched = modeling_qwen2.apply_rotary_pos_emb
        patch_rotary()
        self.assertIs(patched, modeling_qwen2.apply_rotary_pos_emb)
        try:
            for d in (64, 128):
                q = torch.randn(2, 12, 17, d, device="cuda", dtype=torch.float16, requires_grad=True)
                k = torch.randn(2, 2, 17, d, device="cuda", dtype=torch.float16)
                c = torch.randn(2, 17, d, device="cuda", dtype=torch.float16)
                s = torch.randn_like(c)
                with patch.object(Rotary, "apply", wraps=Rotary.apply) as spy:
                    a, b = patched(q, k, c, s)
                self.assertEqual(spy.call_count, int(d == 128))
                self.assertFalse(b.requires_grad)
                ref, _ = original(q, k, c, s)
                torch.testing.assert_close(a, ref, rtol=0, atol=0)
                a.sum().backward()
                self.assertTrue(torch.isfinite(q.grad).all())
        finally:
            unpatch_rotary()

    def test_native_dropout_rng_forward_and_gradient_exact(self):
        import torch
        from solution.fast_dropout import patch_fast_lora_dropouts, unpatch_fast_lora_dropouts
        torch.manual_seed(961)
        model = torch.nn.Module()
        model.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
        dropout = model.lora_dropout["default"]
        for dtype in (torch.float16, torch.float32):
            for n in (4096, 4097, 4 * 512 * 1536):
                for probability in (.05, .17, .5):
                    with self.subTest(dtype=dtype, n=n, p=probability):
                        dropout.p = probability
                        x = torch.randn(n, device="cuda", dtype=dtype)
                        grad = torch.randn(n * 2, device="cuda", dtype=dtype)[::2] * 8

                        def run():
                            value = x.detach().clone().requires_grad_(True)
                            torch.cuda.manual_seed_all(962)
                            y = dropout(value)
                            rng = torch.cuda.get_rng_state().clone()
                            y.backward(grad)
                            return y.detach(), value.grad, rng

                        ref = run()
                        self.assertEqual(patch_fast_lora_dropouts(model), 1)
                        try:
                            for actual, expected in zip(run(), ref):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        finally:
                            unpatch_fast_lora_dropouts(model)

    def test_shared_input_reconstruction_exact_snapshot_and_views(self):
        import torch
        from solution.fast_dropout import (DropoutInputContext, SavedDropoutInput,
                                           unpack_dropout_input)
        from solution.lossless_saved import pack_mask
        x = torch.randn(2, 17, 1536, device="cuda", dtype=torch.float16)
        context = DropoutInputContext()
        saves = []
        outputs = []
        for probability in (.05, .17):
            out, mask = torch.ops.aten.native_dropout.default(x, probability, True)
            outputs.append(out)
            context.register(x, out, pack_mask(mask).packed, probability)
            view = out.view(-1, 1536).transpose(0, 1)
            saved = context.pack(view)
            self.assertIsInstance(saved, SavedDropoutInput)
            saves.append((saved, view.detach().clone()))
        self.assertEqual(context.source_count, 1, "q/v must share a single exact source snapshot")
        context.clear()
        x.add_(10)  # Reconstruct from the snapshot, not a mutable live source.
        for saved, expected in saves:
            actual = unpack_dropout_input(saved)
            self.assertEqual(actual.stride(), expected.stride())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs[0].add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified in place"):
            unpack_dropout_input(saves[0][0])

    def test_fused_mask_pack_snapshot_tail(self):
        import torch
        from solution.fast_dropout import (patch_fast_lora_dropouts, unpatch_fast_lora_dropouts,
                                           shared_dropout_inputs, unpack_dropout_input, SavedDropoutInput)
        model = torch.nn.Module()
        model.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.17)})
        dropout = model.lora_dropout["default"]
        self.assertEqual(patch_fast_lora_dropouts(model), 1)
        try:
            for n in (4097, 2 * 17 * 1536):
                x = torch.randn(n, device="cuda", dtype=torch.float16, requires_grad=True)
                with shared_dropout_inputs() as context:
                    out = dropout(x)
                    saved = context.pack(out)
                    self.assertIsInstance(saved, SavedDropoutInput)
                    self.assertEqual(context.source_count, 1)
                torch.testing.assert_close(saved.source, x, rtol=0, atol=0)
                torch.testing.assert_close(unpack_dropout_input(saved), out, rtol=0, atol=0)
                out.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
        finally:
            unpatch_fast_lora_dropouts(model)

    def test_real_peft_shared_qv_input_gradients_and_rng(self):
        import torch
        from solution.test_lora_fusion import make_lora_layer
        from solution.fast_dropout import (patch_fast_lora_dropouts, unpatch_fast_lora_dropouts,
                                           shared_dropout_inputs)
        model = torch.nn.ModuleDict({"q": make_lora_layer(1536), "v": make_lora_layer(256)})
        identity = {n: id(p) for n, p in model.named_parameters()}
        x = torch.randn(2, 129, 1536, device="cuda", dtype=torch.float16)
        grads = [torch.randn(2, 129, d, device="cuda", dtype=torch.float16) * 8 for d in (1536, 256)]

        def run(shared, needs_x_grad):
            model.zero_grad(set_to_none=True)
            value = x.clone().requires_grad_(needs_x_grad)
            torch.cuda.manual_seed_all(963)
            with torch.autocast("cuda", dtype=torch.float16):
                if shared:
                    with shared_dropout_inputs() as context:
                        outputs = [model[key](value) for key in ("q", "v")]
                    self.assertEqual(context.source_count, 1)
                    self.assertGreaterEqual(context.packed_count, 2, "must replace both real A inputs")
                else:
                    outputs = [model[key](value) for key in ("q", "v")]
            rng = torch.cuda.get_rng_state().clone()
            torch.autograd.backward(outputs, grads)
            params = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            return [o.detach() for o in outputs], value.grad, params, rng

        for needs_x_grad in (False, True):
            ref = run(False, needs_x_grad)
            self.assertEqual(patch_fast_lora_dropouts(model), 2)
            try:
                actual = run(True, needs_x_grad)
                for a, b in zip(actual[0], ref[0]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                if needs_x_grad:
                    torch.testing.assert_close(actual[1], ref[1], rtol=0, atol=0)
                for name in ref[2]:
                    torch.testing.assert_close(actual[2][name], ref[2][name], rtol=0, atol=0)
                self.assertTrue(torch.equal(actual[3], ref[3]))
                self.assertEqual(identity, {n: id(p) for n, p in model.named_parameters()})
            finally:
                unpatch_fast_lora_dropouts(model)


if __name__ == "__main__":
    unittest.main()
