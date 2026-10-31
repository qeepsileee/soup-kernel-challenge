"""Diagnostic only: count actual compiled variants for multiple tensor extents.

Does not alter production code, compiler cache settings or official harness.
Cold compilation must still be included in official whole-command measurements.
"""
from contextlib import contextmanager
import argparse
import time


@contextmanager
def observe(kernels):
    seen = {kernel.fn.__name__: set() for kernel in kernels}
    saved = []
    missing = object()
    for kernel in kernels:
        previous = kernel.__dict__.get("run", missing)
        original = kernel.run
        name = kernel.fn.__name__

        def run(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            if result is not None:
                seen[_name].add(getattr(result, "hash", id(result)))
            return result

        kernel.run = run
        saved.append((kernel, previous))
    try:
        yield seen
    finally:
        for kernel, previous in saved:
            if previous is missing:
                del kernel.run
            else:
                kernel.run = previous


def benchmark_rope():
    import torch
    from .rotary import Rotary, rotary_forward_kernel, rotary_backward_kernel, supported
    if not torch.cuda.is_available():
        raise RuntimeError("T4 required")
    print(torch.cuda.get_device_name(), "RoPE compilation diagnostic; not a training score", flush=True)
    print("Use a fresh standalone TRITON_CACHE_DIR for cold timings; official harness caches are untouched.",
          flush=True)
    kernels = [rotary_forward_kernel, rotary_backward_kernel]
    lengths = (512, 430, 406, 382, 379, 371, 317, 316, 312, 295, 292, 279, 235, 512)
    with observe(kernels) as variants:
        for length in lengths:
            q, k = [torch.randn(4, length, heads, 128, device="cuda", dtype=torch.float16)
                    .transpose(1, 2).detach().requires_grad_(True) for heads in (12, 2)]
            cos = torch.randn(1, length, 128, device="cuda", dtype=torch.float16)
            sin = torch.randn_like(cos)
            grads = [torch.randn_like(x) for x in (q, k)]
            assert supported(q, k, cos, sin)
            for layout in ("input-strides", "contiguous-grad"):
                upstream = grads if layout == "input-strides" else [value.contiguous() for value in grads]
                q.grad = k.grad = None
                before = {name: len(keys) for name, keys in variants.items()}
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = Rotary.apply(q, k, cos, sin)
                torch.autograd.backward(outputs, upstream)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                counts = {name: len(keys) for name, keys in variants.items()}
                added = {name: counts[name] - before[name] for name in counts}
                print(f"RoPE B=4 T={length} cos_batch=1 {layout}: {elapsed:.4f}s "
                      f"compiled_variants={counts} new_variants={added}", flush=True)
                del outputs, upstream
            del q, k, cos, sin, grads
    print("Diagnostic only. Cold startup and compilation still count in official measurements.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rope-only", action="store_true",
                        help="inspect RoPE variants across real variable sequence lengths")
    args = parser.parse_args()
    if args.rope_only:
        benchmark_rope()
        return
    import torch
    import bitsandbytes.functional as functional
    from .nf4 import dequantize_nf4, nf4_decode_kernel
    from .compressed_mlp import (CompressedSwiGLU, example_compressed_swiglu_fwd,
                                 example_compressed_swiglu_bwd)
    if not torch.cuda.is_available():
        raise RuntimeError("T4 required")
    torch.empty(1, device="cuda")
    with observe([nf4_decode_kernel]) as variants:
        for shape in ((256, 1536), (1536, 1536), (8960, 1536)):
            w = torch.randn(shape, device="cuda", dtype=torch.float16)
            packed, state = functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
            start = time.perf_counter()
            actual = dequantize_nf4(packed, state)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ref = functional.dequantize_4bit(packed, state)
            torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-7)
            print(f"NF4 shape={shape}: {elapsed:.3f}s, compiled variants={len(variants[nf4_decode_kernel.fn.__name__])}", flush=True)
            del w, packed, state, actual, ref
    with observe([example_compressed_swiglu_fwd, example_compressed_swiglu_bwd]) as variants:
        for length in (64, 256, 512):
            g = torch.randn((4, length, 8960), device="cuda", dtype=torch.float16, requires_grad=True)
            u = torch.randn_like(g, requires_grad=True)
            start = time.perf_counter()
            y = CompressedSwiGLU.apply(g, u)
            y.sum().backward()
            torch.cuda.synchronize()
            counts = {name: len(keys) for name, keys in variants.items()}
            print(f"MLP seq={length}: {time.perf_counter()-start:.3f}s, compiled variants={counts}", flush=True)
            del g, u, y
    print("Variant counts are diagnostic; actual full-run speed/quality must use the official harness.")


if __name__ == "__main__":
    main()
