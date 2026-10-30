"""Diagnostic only: count actual compiled variants for multiple tensor extents.

Does not alter production code, compiler cache settings or official harness.
Cold compilation must still be included in official whole-command measurements.
"""
from contextlib import contextmanager
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


def main():
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
