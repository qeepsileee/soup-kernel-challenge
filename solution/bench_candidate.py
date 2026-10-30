"""Short component benchmark; actual speed/coverage decided by unmodified harness."""
import argparse
import time


def main():
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    from .nf4_linear import NF4Linear, unpatch_linear
    from .nf4 import unpatch_dequantize
    from .nf4_format import qwen_projection_shapes
    from .rmsnorm import FrozenRMSNorm
    from .entropy import entropy_from_logits
    from trl.trainer.utils import entropy_from_logits as reference_entropy

    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    args = p.parse_args()
    if args.tokens <= 0:
        p.error("tokens must be positive")
    unpatch_linear()
    unpatch_dequantize()
    torch.manual_seed(55)

    def measure(fn, repeats=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / repeats

    def report(name, base, candidate):
        print(f"{name:15s} base={base:.3f}ms candidate={candidate:.3f}ms speedup={base/candidate:.3f}x", flush=True)

    print(torch.cuda.get_device_name(), "tokens", args.tokens, flush=True)
    for name, shape in qwen_projection_shapes().items():
        w = torch.randn(shape, device="cuda", dtype=torch.float16)
        packed, state = functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
        del w
        x = torch.randn((args.tokens, shape[1]), device="cuda", dtype=torch.float16, requires_grad=True)
        grad = torch.randn((args.tokens, shape[0]), device="cuda", dtype=torch.float16)

        def run(candidate):
            x.grad = None
            y = NF4Linear.apply(x, packed, None, state) if candidate else bnb.matmul_4bit(x, packed, state)
            y.backward(grad)

        base = measure(lambda: run(False))
        cand = measure(lambda: run(True))
        report(name, base, cand)
        del x, grad, packed, state

    x = torch.randn((args.tokens, 1536), device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.ones(1536, device="cuda", dtype=torch.float32)
    grad = torch.randn_like(x)

    def rms(candidate):
        x.grad = None
        if candidate:
            y = FrozenRMSNorm.apply(x, w, 1e-6)
        else:
            y = w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6))
        y.backward(grad)

    report("RMSNorm fwd+bwd", measure(lambda: rms(False)), measure(lambda: rms(True)))
    del x, w, grad

    # Benchmark Fused Cross-Entropy vs PyTorch CE + autograd backward + entropy
    from .fused_ce import fused_cross_entropy
    import torch.nn.functional as F

    chunk_tokens = min(args.tokens, 128)
    logits_bench = torch.randn((chunk_tokens, 151936), device="cuda", dtype=torch.float16)
    targets_bench = torch.randint(0, 151936, (chunk_tokens,), device="cuda", dtype=torch.long)
    targets_bench[targets_bench % 3 == 0] = -100  # mix in prompt/pad tokens

    def bench_pyt():
        l = logits_bench.clone().requires_grad_(True)
        wide = l.float()
        loss = F.cross_entropy(wide, targets_bench, ignore_index=-100, reduction="sum") / chunk_tokens
        loss.backward()
        with torch.no_grad():
            reference_entropy(l.float())

    def bench_fused():
        l = logits_bench.clone()
        fused_cross_entropy(l, targets_bench, denom=chunk_tokens, needs_grad=True, inplace_grad=True)

    report(f"Fused CE ({chunk_tokens}x152k)", measure(bench_pyt, 10), measure(bench_fused, 10))
    del logits_bench, targets_bench

    logits = torch.randn((args.tokens, 151936), device="cuda", dtype=torch.float32)
    with torch.no_grad():
        expected = reference_entropy(logits)
        actual = entropy_from_logits(logits)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        report("entropy", measure(lambda: reference_entropy(logits), 10),
               measure(lambda: entropy_from_logits(logits), 10))


if __name__ == "__main__":
    main()
