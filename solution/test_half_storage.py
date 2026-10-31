import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class HalfStorageTests(unittest.TestCase):
    def test_mixed_dtype_comparison_matches_explicit_round_trip(self):
        import torch
        from solution.half_storage import convert_frozen_exact

        cases = ([0., -0., 1., -1., 65504., 2**-24, float("inf"), -float("inf")],
                 [0.12345679], [65520.], [2**-25], [1e-40], [float("nan")])
        for values in cases:
            with self.subTest(values=values):
                source = torch.tensor(values, dtype=torch.float32)
                half = source.half()
                expected = torch.equal(source, half.float())
                self.assertEqual(torch.equal(source, half), expected)
                model = torch.nn.Module()
                model.weight = torch.nn.Parameter(source.clone(), requires_grad=False)
                stats = convert_frozen_exact(model)
                self.assertEqual(stats["parameters"], int(expected))
                self.assertEqual(model.weight.dtype, torch.float16 if expected else torch.float32)

    def test_exact_values_aliases_and_exclusions(self):
        import torch
        from solution.half_storage import convert_frozen_exact

        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(32, 16)
        model.embedding.weight = torch.nn.Parameter(
            torch.randn(32, 16).half().float(), requires_grad=False)
        model.head = torch.nn.Linear(16, 32, bias=False)
        model.head.weight = model.embedding.weight
        model.adapter = torch.nn.Parameter(torch.randn(16, 4))
        model.inexact = torch.nn.Parameter(torch.tensor([0.12345679]), requires_grad=False)
        model.register_buffer("quant_scale", torch.tensor([0.12345679]))
        identity = id(model.embedding.weight)
        before = model.embedding.weight.detach().clone()
        adapter = model.adapter.detach().clone()
        stats = convert_frozen_exact(model)
        self.assertEqual(stats, {"parameters": 1, "saved_bytes": 1024})
        self.assertEqual(id(model.embedding.weight), identity)
        self.assertIs(model.head.weight, model.embedding.weight)
        torch.testing.assert_close(model.embedding.weight.float(), before, rtol=0, atol=0)
        torch.testing.assert_close(model.adapter, adapter, rtol=0, atol=0)
        self.assertEqual(model.adapter.dtype, torch.float32)
        self.assertEqual(model.inexact.dtype, torch.float32)
        self.assertEqual(model.quant_scale.dtype, torch.float32)
        self.assertEqual(convert_frozen_exact(model)["saved_bytes"], 0)
