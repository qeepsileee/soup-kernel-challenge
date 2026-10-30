"""Remove lossless FP16->FP32->FP16 MLP intermediates, retain FP32 residuals.

Reuses the shipped example SwiGLU unchanged; its kernels do NOT count as ours.
Only frozen NF4 MLP projections with FP16 compute and no bias/LoRA qualify.
"""
import torch


def can_bridge(module, x):
    import bitsandbytes as bnb
    from .nf4 import supported

    if not x.is_cuda or x.dtype != torch.float32 or x.numel() == 0:
        return False
    if getattr(getattr(module, "config", None), "hidden_act", "silu") != "silu":
        return False
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(module, name, None)
        if not isinstance(layer, bnb.nn.Linear4bit):
            return False
        if (layer.compute_dtype != torch.float16 or not layer.compute_type_is_set
                or layer.bias is not None or layer.weight.requires_grad
                or layer.weight.device != x.device
                or not supported(layer.weight, getattr(layer.weight, "quant_state", None))):
            return False
    return True


def bridged_mlp(module, x, activation=None):
    import bitsandbytes as bnb
    from .swiglu import swiglu

    def project(layer, value):
        return bnb.matmul_4bit(value, layer.weight, layer.weight.quant_state)

    # Keep SEPARATE casts: backward must accumulate gate/up contributions in
    # FP32 at x, not sum them at a shared FP16 intermediate and round again.
    gate = project(module.gate_proj, x.to(torch.float16))
    up = project(module.up_proj, x.to(torch.float16))
    # Example kernel evaluates activation in FP32, rounds only its output. The
    # baseline down_proj immediately rounds that same output to FP16 anyway.
    hidden = (swiglu if activation is None else activation)(gate, up)
    output = project(module.down_proj, hidden)
    return output.to(x.dtype)


def patch_mlp_bridge():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    original = Qwen2MLP.forward
    if getattr(original, "_nf4_mlp_bridge", False):
        return

    def forward(self, x):
        if can_bridge(self, x):
            return bridged_mlp(self, x)
        return original(self, x)

    forward._nf4_mlp_bridge = True
    forward._original = original
    Qwen2MLP.forward = forward
