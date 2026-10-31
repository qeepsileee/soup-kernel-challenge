"""Paired exact-compute NF4 forward/backward retention diagnostic."""
import argparse
import os
from statistics import median
import time


def main():
    import torch
    import bitsandbytes.functional as functional
    from .nf4_experiments import ExperimentalNF4Linear as NF4Linear
    from .nf4_format import qwen_projection_shapes
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--retain", choices=("0", "1", "kv"), default="1")
    parser.add_argument("--dx-layout", choices=("original", "transposed"), default="original")
    parser.add_argument("--blas", choices=("original", "cublaslt"), default="original",
                        help="diagnostic preference; precision settings remain unchanged")
    args = parser.parse_args()
    if min(args.tokens, args.iterations) < 1:
        parser.error("tokens and iterations must be positive")
    torch.manual_seed(914)
    print(torch.cuda.get_device_name(), "tokens=", args.tokens,
          "retain=", args.retain, "dx-layout=", args.dx_layout, "requested-blas=", args.blas)
    previous = os.environ.get("SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT")
    previous_layout = os.environ.get("SOUP_EXPERIMENT_NF4_DX_LAYOUT")
    previous_blas = torch.backends.cuda.preferred_blas_library()
    try:
        for name, shape in qwen_projection_shapes().items():
            w = torch.randn(shape, device="cuda", dtype=torch.float16) * .02
            packed, state = functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
            del w
            x = torch.randn(args.tokens, shape[1], device="cuda", dtype=torch.float16, requires_grad=True)
            grad = torch.randn(args.tokens, shape[0], device="cuda", dtype=torch.float16)

            def step(collect=False):
                x.grad = None
                y = NF4Linear.apply(x, packed, None, state)
                y.backward(grad)
                if collect:
                    return y.detach().clone(), x.grad.detach().clone()

            def configure(candidate):
                os.environ["SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT"] = args.retain if candidate else "0"
                os.environ["SOUP_EXPERIMENT_NF4_DX_LAYOUT"] = args.dx_layout if candidate else "original"
                torch.backends.cuda.preferred_blas_library(
                    "cublaslt" if candidate and args.blas == "cublaslt" else previous_blas)

            if args.blas != "original":
                configure(False)
                reference = step(collect=True)
                configure(True)
                actual = step(collect=True)
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, rtol=3e-3, atol=1e-2)
                print(f"{name}: requested {args.blas}, numerical preflight passed; "
                      f"max forward delta={(actual[0].float()-reference[0].float()).abs().max().item():.6g}",
                      flush=True)
                del reference, actual

            def measure(candidate):
                configure(candidate)
                for _ in range(5):
                    step()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    step()
                torch.cuda.synchronize()
                return (time.perf_counter() - start) * 1000 / args.iterations

            bases, candidates = [], []
            for trial in range(4):
                if trial % 2:
                    c, b = measure(1), measure(0)
                else:
                    b, c = measure(0), measure(1)
                bases.append(b)
                candidates.append(c)
            print(f"{name:12s} base={median(bases):.3f}ms candidate={median(candidates):.3f}ms "
                  f"speedup={median(bases)/median(candidates):.3f}x "
                  f"pairs={[round(b/c, 3) for b,c in zip(bases,candidates)]}", flush=True)
    finally:
        if previous is None:
            os.environ.pop("SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT", None)
        else:
            os.environ["SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT"] = previous
        if previous_layout is None:
            os.environ.pop("SOUP_EXPERIMENT_NF4_DX_LAYOUT", None)
        else:
            os.environ["SOUP_EXPERIMENT_NF4_DX_LAYOUT"] = previous_layout
        torch.backends.cuda.preferred_blas_library(previous_blas)
    print("Layer timings omit simultaneous retention across the full model; use the official harness for peak memory.")


if __name__ == "__main__":
    main()
