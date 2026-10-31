"""ABBA exact saved-data codecs: native attention and native LoRA dropout."""
import argparse
from statistics import median
import time


def main():
    import torch
    from types import SimpleNamespace
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .lossless_saved import (patch_lossless_attention, unpatch_lossless_attention,
                                patch_lora_dropouts, unpatch_lora_dropouts)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not 16 <= args.tokens <= 512 or args.iterations < 1:
        parser.error("tokens must be 16..512; iterations must be positive")
    torch.manual_seed(954)
    length = args.tokens
    unpatch_lossless_attention()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    patch_lossless_attention()
    compact = ALL_ATTENTION_FUNCTIONS["sdpa"]
    module = torch.nn.Module()
    module.config = SimpleNamespace(model_type="qwen2")
    module.num_key_value_groups, module.is_causal = 6, True
    q, k, v = [torch.randn(4, h, length, 128, device="cuda", dtype=torch.float16, requires_grad=True)
               for h in (12, 2, 2)]
    mask = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()[None, None]
    mask[..., -3:] = False
    attention_grad = torch.randn(4, length, 12, 128, device="cuda", dtype=torch.float16)
    dropout_model = torch.nn.Module()
    dropout_model.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
    x = torch.randn(4, length, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    dropout_grad = torch.randn_like(x)

    def clear():
        for tensor in (q, k, v, x):
            tensor.grad = None

    def forward(kind, optimized):
        if kind == "attention":
            y, _ = (compact if optimized else original)(module, q, k, v, mask, dropout=0.)
            return y, attention_grad
        return dropout_model.lora_dropout["default"](x), dropout_grad

    def step(kind, optimized):
        clear()
        y, grad = forward(kind, optimized)
        y.backward(grad)

    def measure(kind, optimized):
        if kind == "dropout":
            (patch_lora_dropouts if optimized else unpatch_lora_dropouts)(dropout_model)
        for _ in range(5):
            step(kind, optimized)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            step(kind, optimized)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / args.iterations
        clear()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(kind, optimized)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - before) / 2**20
        clear()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        y, grad = forward(kind, optimized)
        torch.cuda.synchronize()
        resident = (torch.cuda.memory_allocated() - before) / 2**20
        y.backward(grad)
        return elapsed, peak, resident

    try:
        print(torch.cuda.get_device_name(), "B=4 T=", length, "Q=12 KV=2 D=128; native backend unchanged")
        for kind in ("attention", "dropout"):
            bases, candidates = [], []
            for trial in range(4):
                if trial % 2:
                    c, b = measure(kind, True), measure(kind, False)
                else:
                    b, c = measure(kind, False), measure(kind, True)
                bases.append(b)
                candidates.append(c)
            base, candidate = median(row[0] for row in bases), median(row[0] for row in candidates)
            print(f"{kind}: base={base:.3f}ms lossless={candidate:.3f}ms speedup={base/candidate:.3f}x "
                  f"pairs={[round(b[0]/c[0],3) for b,c in zip(bases,candidates)]}", flush=True)
            print(f"  incremental peak: base={median(row[1] for row in bases):.2f}MiB "
                  f"lossless={median(row[1] for row in candidates):.2f}MiB; "
                  f"live after forward: base={median(row[2] for row in bases):.2f}MiB "
                  f"lossless={median(row[2] for row in candidates):.2f}MiB", flush=True)
    finally:
        unpatch_lossless_attention()
        unpatch_lora_dropouts(dropout_model)
    print("Local allocated memory only; official whole-model time/memory and admission still need the full harness.")


if __name__ == "__main__":
    main()
