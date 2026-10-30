"""Compare full frozen MLP with normal vs INT8 saved activations; no patching."""
import argparse
import time
from statistics import median


def main():
    import bitsandbytes as bnb
    import torch
    from types import SimpleNamespace
    from .compressed_mlp import CompressedSwiGLU
    from .mlp_bridge import bridged_mlp
    from .nf4 import patch_dequantize
    from .nf4_linear import patch_linear
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.tokens <= 0 or args.iterations <= 0:
        parser.error("tokens and iterations must be positive")
    torch.manual_seed(802)
    patch_dequantize()
    patch_linear()
    module = torch.nn.Module()
    module.config = SimpleNamespace(hidden_act="silu")
    for name, ins, outs in (("gate_proj", 1536, 8960), ("up_proj", 1536, 8960),
                            ("down_proj", 8960, 1536)):
        layer = bnb.nn.Linear4bit(ins, outs, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4")
        layer.weight = bnb.nn.Params4bit(torch.randn(outs, ins) * .02,
            requires_grad=False, compress_statistics=True, quant_type="nf4")
        setattr(module, name, layer.cuda())
    x = torch.randn((1, args.tokens, 1536), device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn_like(x) * 8

    def step(compressed):
        x.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            y = bridged_mlp(module, x, activation=CompressedSwiGLU.apply if compressed else None)
        y.backward(grad)

    def measure(compressed):
        for _ in range(5):
            step(compressed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            step(compressed)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000 / args.iterations
        x.grad = None
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(compressed)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - before) / 2**20
        x.grad = None
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            y = bridged_mlp(module, x, activation=CompressedSwiGLU.apply if compressed else None)
        torch.cuda.synchronize()
        resident = (torch.cuda.memory_allocated() - before) / 2**20
        y.backward(grad)
        return elapsed, peak, resident

    baseline, candidate = [], []
    for trial in range(4):
        if trial % 2:
            c, b = measure(True), measure(False)
        else:
            b, c = measure(False), measure(True)
        baseline.append(b)
        candidate.append(c)
    a, b = median(x[0] for x in baseline), median(x[0] for x in candidate)
    print(f"MLP tokens={args.tokens}: baseline={a:.3f}ms compressed={b:.3f}ms speedup={a/b:.3f}x")
    print("paired speedups:", [round(x[0]/y[0], 3) for x, y in zip(baseline, candidate)])
    print(f"Diagnostic incremental allocated peaks: baseline={median(x[1] for x in baseline):.1f}MiB "
          f"compressed={median(x[1] for x in candidate):.1f}MiB")
    print(f"Live allocated after forward: baseline={median(x[2] for x in baseline):.1f}MiB "
          f"compressed={median(x[2] for x in candidate):.1f}MiB")
    print("Peaks are per-layer PyTorch allocated memory, not the official whole-run nvidia-smi score.")


if __name__ == "__main__":
    main()
