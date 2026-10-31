"""ABBA full frozen MLP, including all dequantization and INT8 work."""
import argparse
from statistics import median
import time


def main():
    import torch
    from .test_paired_mlp import make_frozen_mlp
    from .compressed_mlp import CompressedSwiGLU
    from .mlp_bridge import bridged_mlp
    from .paired_mlp import paired_mlp
    from .nf4_linear import patch_linear, unpatch_linear
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--candidate", choices=("paired", "nf4-saved"), default="paired")
    args = parser.parse_args()
    if args.tokens < 512 or args.iterations < 1:
        parser.error("tokens must be >=512; iterations must be positive")
    torch.manual_seed(930)
    if args.candidate == "nf4-saved":
        from .nf4_saved_mlp import NF4SavedSwiGLU
        candidate_fn = lambda module, x: bridged_mlp(module, x, activation=NF4SavedSwiGLU.apply)
    else:
        candidate_fn = paired_mlp
    patch_linear()
    try:
        module = make_frozen_mlp()
        x = torch.randn(1, args.tokens, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
        grad = torch.randn_like(x) * 8

        def step(paired):
            x.grad = None
            with torch.autocast("cuda", dtype=torch.float16):
                y = candidate_fn(module, x) if paired else bridged_mlp(module, x, activation=CompressedSwiGLU.apply)
            y.backward(grad)

        def measure(paired):
            for _ in range(5):
                step(paired)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                step(paired)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000 / args.iterations
            x.grad = None
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            step(paired)
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - before) / 2**20
            x.grad = None
            before = torch.cuda.memory_allocated()
            with torch.autocast("cuda", dtype=torch.float16):
                y = candidate_fn(module, x) if paired else bridged_mlp(module, x, activation=CompressedSwiGLU.apply)
            torch.cuda.synchronize()
            resident = (torch.cuda.memory_allocated() - before) / 2**20
            y.backward(grad)
            return elapsed, peak, resident

        bases, candidates = [], []
        for trial in range(4):
            if trial % 2:
                c, b = measure(True), measure(False)
            else:
                b, c = measure(False), measure(True)
            bases.append(b)
            candidates.append(c)
        base, candidate = median(v[0] for v in bases), median(v[0] for v in candidates)
        print(torch.cuda.get_device_name(), "FP16 frozen MLP tokens=", args.tokens, "candidate=", args.candidate)
        print(f"base={base:.3f}ms candidate={candidate:.3f}ms speedup={base/candidate:.3f}x "
              f"pairs={[round(b[0]/c[0],3) for b,c in zip(bases,candidates)]}")
        print(f"Incremental allocated peak: base={median(v[1] for v in bases):.1f}MiB "
              f"paired={median(v[1] for v in candidates):.1f}MiB")
        print(f"Live allocated after forward: base={median(v[2] for v in bases):.1f}MiB "
              f"paired={median(v[2] for v in candidates):.1f}MiB")
        print("Diagnostic only. Not the official whole-model memory/time measurement.")
    finally:
        unpatch_linear()


if __name__ == "__main__":
    main()
