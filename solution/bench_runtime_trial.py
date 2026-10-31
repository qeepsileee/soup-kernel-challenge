"""ABBA component comparisons against the validated memory branch, not scores."""
import argparse
from statistics import median
import time


def main():
    import torch
    from transformers.models.qwen2 import modeling_qwen2
    from .rotary import Rotary, unpatch_rotary
    from .lossless_saved import patch_lora_dropouts, unpatch_lora_dropouts
    from .fast_dropout import (patch_fast_lora_dropouts, unpatch_fast_lora_dropouts,
                               shared_dropout_inputs)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if not 16 <= args.tokens <= 512 or args.iterations < 1:
        parser.error("tokens must be 16..512; iterations must be positive")
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(964)
    length = args.tokens
    unpatch_rotary()
    reference_rotary = modeling_qwen2.apply_rotary_pos_emb
    q, k = [torch.randn(4, length, h, 128, device="cuda", dtype=torch.float16)
            .transpose(1, 2).detach().requires_grad_(True) for h in (12, 2)]
    cos = torch.randn(4, length, 128, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)
    rotary_grads = [torch.randn_like(t) for t in (q, k)]
    x = torch.randn(4, length, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    dropout_grad = torch.randn_like(x)
    dropout_model = torch.nn.Module()
    dropout_model.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})

    class Adapter(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
            self.a = torch.nn.Linear(1536, 16, bias=False)
            self.b = torch.nn.Linear(16, out, bias=False)

        def forward(self, value):
            return self.b(self.a(self.lora_dropout["default"](value)))

    adapters = torch.nn.ModuleList([Adapter(1536), Adapter(256)]).cuda()
    adapter_grads = [torch.randn(4, length, d, device="cuda", dtype=torch.float16) for d in (1536, 256)]

    def clear():
        for value in (q, k, x):
            value.grad = None
        adapters.zero_grad(set_to_none=True)

    def configure(kind, candidate):
        model = dropout_model if kind == "dropout-adjoint" else adapters
        unpatch_fast_lora_dropouts(model)
        unpatch_lora_dropouts(model)
        if kind != "rotary":
            (patch_fast_lora_dropouts if candidate else patch_lora_dropouts)(model)

    def forward(kind, candidate):
        if kind == "rotary":
            fn = Rotary.apply if candidate else reference_rotary
            return fn(q, k, cos, sin), rotary_grads
        if kind == "dropout-adjoint":
            return (dropout_model.lora_dropout["default"](x),), (dropout_grad,)
        with torch.autocast("cuda", dtype=torch.float16):
            if candidate:
                with shared_dropout_inputs():
                    outputs = [adapter(x) for adapter in adapters]
            else:
                outputs = [adapter(x) for adapter in adapters]
        return outputs, adapter_grads

    def step(kind, candidate):
        clear()
        outputs, grads = forward(kind, candidate)
        torch.autograd.backward(outputs, grads)

    def measure(kind, candidate):
        configure(kind, candidate)
        for _ in range(10):
            step(kind, candidate)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            step(kind, candidate)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / args.iterations
        clear()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(kind, candidate)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - before) / 2**20
        clear()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        outputs, grads = forward(kind, candidate)
        torch.cuda.synchronize()
        resident = (torch.cuda.memory_allocated() - before) / 2**20
        torch.autograd.backward(outputs, grads)
        return elapsed, peak, resident

    print(torch.cuda.get_device_name(), f"B=4 T={length}; baseline=validated memory codecs", flush=True)
    try:
        for kind in ("rotary", "dropout-adjoint", "shared-qv-adapters"):
            bases, candidates = [], []
            for trial in range(4):
                if trial % 2:
                    c, b = measure(kind, True), measure(kind, False)
                else:
                    b, c = measure(kind, False), measure(kind, True)
                bases.append(b)
                candidates.append(c)
            base, candidate = median(row[0] for row in bases), median(row[0] for row in candidates)
            print(f"{kind}: base={base:.3f}ms candidate={candidate:.3f}ms speedup={base/candidate:.3f}x "
                  f"pairs={[round(b[0]/c[0],3) for b,c in zip(bases,candidates)]}", flush=True)
            print(f"  incremental peak: {median(row[1] for row in bases):.2f} -> "
                  f"{median(row[1] for row in candidates):.2f}MiB; live after forward: "
                  f"{median(row[2] for row in bases):.2f} -> {median(row[2] for row in candidates):.2f}MiB",
                  flush=True)
    finally:
        for model in (dropout_model, adapters):
            unpatch_fast_lora_dropouts(model)
            unpatch_lora_dropouts(model)
    print("Diagnostic local allocated memory/time only. Official whole-task checks decide admission.")


if __name__ == "__main__":
    main()
