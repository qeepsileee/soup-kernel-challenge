import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class HalfStorageTests(unittest.TestCase):
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
