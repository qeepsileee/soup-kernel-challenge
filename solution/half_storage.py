"""Opt-in experiment: exact-value FP16 storage of frozen FP32 parameters.

Parameter values round-trip exactly, but activation/gradient rounding changes.
Full official quality validation is still required. Never convert trainable
adapters, Params4bit subclasses, or buffers (including quantization metadata).
"""
import os


def convert_frozen_exact(model):
    import torch

    count = saved = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if (type(parameter) is not torch.nn.Parameter or parameter.requires_grad
                    or parameter.dtype != torch.float32):
                continue
            half = parameter.detach().to(torch.float16)
            # Reject overflow, underflow and any change of parameter value.
            if not torch.equal(parameter.detach(), half.float()):
                continue
            saved += parameter.numel() * 2
            count += 1
            # Keep Parameter identity, tied weights and all optimizer references.
            parameter.data = half
    return {"parameters": count, "saved_bytes": saved}


def patch_half_storage():
    if os.environ.get("SOUP_EXPERIMENT_HALF_STORAGE") != "1":
        return
    from soup_cli.trainer.sft import SFTTrainerWrapper
    original = SFTTrainerWrapper._setup_transformers
    if getattr(original, "_own_half_storage", False):
        return

    def setup(self, *args, **kwargs):
        import torch

        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if (not isinstance(model, torch.nn.Module)
                or getattr(getattr(model, "config", None), "model_type", None) != "qwen2"
                or not any(p.is_cuda for p in model.parameters())):
            return result
        stats = convert_frozen_exact(model)
        model._own_half_storage_stats = stats
        print("[half_storage]", stats, flush=True)
        return result

    setup._own_half_storage = True
    setup._original = original
    SFTTrainerWrapper._setup_transformers = setup
