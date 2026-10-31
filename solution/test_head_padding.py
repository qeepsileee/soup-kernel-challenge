"""Padding-head optimization: actual masked metrics and scaled gradients."""
import unittest
from solution.cuda_test_support import GPU_AVAILABLE


@unittest.skipUnless(GPU_AVAILABLE, "PyTorch + Triton + CUDA required")
class HeadPaddingTests(unittest.TestCase):
    def test_padding_patterns_metrics_gradient_and_actual_work(self):
        import torch
        from types import SimpleNamespace
        from collections import defaultdict
        from unittest.mock import patch
        from solution import chunked_loss as module
        from solution.patch_chunked_loss import _log_trl_metrics
        torch.manual_seed(970)
        head = torch.nn.Linear(64, 2048, bias=False, device="cuda", dtype=torch.float16)
        head.weight.requires_grad_(False)
        values = torch.randn(2, 37, 64, device="cuda", dtype=torch.float16)
        for pattern in ("right", "left", "holes"):
            mask = torch.ones(2, 37, device="cuda", dtype=torch.long)
            if pattern == "right":
                mask[0, 19:], mask[1, 29:] = 0, 0
            elif pattern == "left":
                mask[0, :13], mask[1, :9] = 0, 0
            else:
                mask[:, 9:17], mask[0, 25:31] = 0, 0
            labels = torch.randint(2048, (2, 37), device="cuda")
            labels[:, :18] = -100  # Ignored prompt still needs entropy.
            labels[mask == 0] = -100
            target = module.shift_for_next_token(labels)
            kept = (mask != 0) | (target != -100)
            for scale in (1., 1024., 65536.):
                with self.subTest(pattern=pattern, scale=scale):
                    a = values.clone().requires_grad_(True)
                    b = values.clone().requires_grad_(True)
                    ref, rs = module.chunked_head_loss(a, head, labels, chunk_size=8, precompute_grad=True)
                    projected = [0]
                    original = module.fused_cross_entropy

                    def counted(logits, *args, **kwargs):
                        projected[0] += logits.shape[0]
                        return original(logits, *args, **kwargs)

                    with patch.object(module, "fused_cross_entropy", counted):
                        out, step = module.chunked_head_loss(b, head, labels, chunk_size=8,
                            precompute_grad=True, attention_mask=mask)
                    self.assertEqual(projected[0], kept.sum().item())
                    self.assertLess(projected[0], mask.numel())
                    ga, = torch.autograd.grad(ref * scale, a)
                    gb, = torch.autograd.grad(out * scale, b)
                    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(step.token_loss, rs.token_loss, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(step.token_entropy[kept], rs.token_entropy[kept], rtol=2e-4, atol=2e-4)
                    self.assertTrue(torch.all(step.token_entropy[~kept] == 0))
                    self.assertEqual(step.correct.item(), rs.correct.item())
                    self.assertEqual(step.labelled.item(), rs.labelled.item())
                    self.assertTrue(torch.isfinite(gb).all())
                    relative = (ga.float()-gb.float()).norm() / ga.float().norm().clamp_min(1e-20)
                    self.assertLess(relative.item(), .005)
                    self.assertTrue(torch.all(gb[target == -100] == 0))

                    def metrics(value):
                        trainer = SimpleNamespace(_metrics={"train": defaultdict(list)},
                            _total_train_tokens=0,
                            accelerator=SimpleNamespace(gather_for_metrics=lambda tensor: tensor))
                        _log_trl_metrics(trainer, "train", {"attention_mask": mask}, value)
                        return trainer._metrics["train"]

                    actual, expected = metrics(step), metrics(rs)
                    self.assertAlmostEqual(actual["entropy"][0], expected["entropy"][0], delta=2e-4)
                    self.assertEqual(actual["num_tokens"], expected["num_tokens"])
                    self.assertEqual(actual["mean_token_accuracy"], expected["mean_token_accuracy"])

    def test_masked_labelled_positions_and_all_padding_are_preserved(self):
        import torch
        from unittest.mock import patch
        from solution import chunked_loss as module
        head = torch.nn.Linear(16, 64, bias=False, device="cuda", dtype=torch.float16)
        head.weight.requires_grad_(False)
        mask = torch.zeros(1, 8, device="cuda", dtype=torch.bool)
        labels = torch.full((1, 8), -100, device="cuda")
        for count in (0, 1):
            with self.subTest(labelled=count):
                target = labels.clone()
                if count:
                    target[0, 2] = 4  # Required even though attention_mask is zero.
                x = torch.randn(1, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
                projected = [0]
                original = module.fused_cross_entropy

                def counted(logits, *args, **kwargs):
                    projected[0] += logits.shape[0]
                    return original(logits, *args, **kwargs)

                with patch.object(module, "fused_cross_entropy", counted):
                    loss, step = module.chunked_head_loss(x, head, labels, shift_labels=target,
                        chunk_size=3, precompute_grad=True, attention_mask=mask)
                self.assertEqual(projected[0], count)
                self.assertEqual(step.labelled.item(), count)
                (loss * 1024.).backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                if not count:
                    self.assertEqual(loss.item(), 0.)
                    self.assertTrue(torch.all(x.grad == 0))
                    self.assertTrue(torch.all(step.token_entropy == 0))
                else:
                    self.assertGreater(loss.item(), 0.)
                    self.assertGreater(step.token_entropy[0, 2].item(), 0.)

    def test_real_vocab_amp_and_no_grad_fallback(self):
        import torch
        from solution.chunked_loss import chunked_head_loss
        head = torch.nn.Linear(64, 151936, bias=False, device="cuda", dtype=torch.float32)
        head.weight.requires_grad_(False)
        mask = torch.ones(2, 17, device="cuda", dtype=torch.long)
        mask[0, 8:], mask[1, 13:] = 0, 0
        labels = torch.randint(151936, (2, 17), device="cuda")
        labels[:, :4], labels[mask == 0] = -100, -100
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                a = torch.randn(2, 17, 64, device="cuda", dtype=dtype, requires_grad=True)
                b = a.detach().clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.float16):
                    ref, rs = chunked_head_loss(a, head, labels, chunk_size=3, precompute_grad=True)
                    loss, step = chunked_head_loss(b, head, labels, chunk_size=3,
                                                  precompute_grad=True, attention_mask=mask)
                ga, = torch.autograd.grad(ref * 65536., a)
                gb, = torch.autograd.grad(loss * 65536., b)
                torch.testing.assert_close(loss, ref, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(step.token_loss, rs.token_loss, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(step.token_entropy[mask != 0], rs.token_entropy[mask != 0],
                                           rtol=2e-4, atol=2e-4)
                self.assertLess(((ga.float()-gb.float()).norm()/ga.float().norm()).item(), .005)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ref, rs = chunked_head_loss(a, head, labels, precompute_grad=True)
            loss, step = chunked_head_loss(a, head, labels, precompute_grad=True, attention_mask=mask)
        torch.testing.assert_close(loss, ref, rtol=0, atol=0)
        torch.testing.assert_close(step.token_entropy, rs.token_entropy, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            chunked_head_loss(a, head, labels, attention_mask=mask[:, :-1])

    def test_real_model_hook_forwards_attention_mask_and_rng(self):
        import torch
        from unittest.mock import patch
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from solution.patch_chunked_loss import _OwnChunkedHeadLossHook
        config = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1)
        config._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(config).cuda().half().train()
        model.get_output_embeddings().weight.requires_grad_(False)
        ids = torch.randint(512, (2, 32), device="cuda")
        mask = torch.ones_like(ids)
        mask[0, 19:], mask[1, 25:] = 0, 0
        labels = ids.clone()
        labels[:, :8], labels[mask == 0] = -100, -100
        torch.cuda.manual_seed_all(971)
        with torch.autocast("cuda", dtype=torch.float16):
            ref = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        expected_rng = torch.cuda.get_rng_state().clone()
        hook = _OwnChunkedHeadLossHook(model, chunk=8)
        try:
            torch.cuda.manual_seed_all(971)
            with patch.object(hook, "_chunked_head_loss", wraps=hook._chunked_head_loss) as spy:
                with torch.autocast("cuda", dtype=torch.float16):
                    actual = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
            self.assertIs(spy.call_args.kwargs["attention_mask"], mask)
            torch.testing.assert_close(actual.loss, ref.loss, rtol=1e-3, atol=1e-3)
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(), expected_rng))
            step = hook.take_step()
            logp = ref.logits.detach().float().log_softmax(-1)
            entropy = -(logp.exp() * logp).sum(-1)
            torch.testing.assert_close((step.token_entropy * mask).sum()/mask.sum(),
                                       (entropy * mask).sum()/mask.sum(), rtol=2e-4, atol=2e-4)
            self.assertIsNone(actual.logits)
            (actual.loss * 1024.).backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        finally:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
