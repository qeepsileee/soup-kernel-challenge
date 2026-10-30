"""Compare actual PEFT baseline and LoRA epilogue candidate on q/v geometries."""
import argparse
import time
from statistics import median


def main():
    import torch
    from peft.tuners.lora.bnb import Linear4bit
    from .test_lora_fusion import make_lora_layer
    from .lora_fusion import patch_lora, unpatch_lora
    from .nf4 import patch_dequantize
    from .nf4_linear import patch_linear
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--policy", choices=("all", "selective"), default="selective")
    parser.add_argument("--out-features", type=int, choices=(256, 1536))
    args = parser.parse_args()
    if args.tokens <= 0 or args.iterations <= 0 or args.rounds < 2:
        parser.error("tokens/iterations must be positive; rounds must be >=2")
    # Same validated NF4 base path on BOTH sides; isolate LoRA's contribution.
    patch_dequantize()
    patch_linear()
    unpatch_lora()
    original = Linear4bit.forward
    patch_lora(policy=args.policy)
    candidate = Linear4bit.forward
    try:
        for out in ((args.out_features,) if args.out_features else (256, 1536)):
            layer = make_lora_layer(out)
            x = torch.randn((args.tokens, 1536), device="cuda", requires_grad=True)
            grad = torch.randn((args.tokens, out), device="cuda") * 8
            for amp in (False, True):
                def measure(fn):
                    def step():
                        layer.zero_grad(set_to_none=True)
                        x.grad = None
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            result = fn(layer, x)
                        result.backward(grad.to(result.dtype))
                    for _ in range(5):
                        step()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        step()
                    torch.cuda.synchronize()
                    return (time.perf_counter() - start) * 1000 / args.iterations
                baselines, candidates = [], []
                for trial in range(args.rounds):
                    if trial % 2 == 0:
                        baseline, optimized = measure(original), measure(candidate)
                    else:
                        optimized, baseline = measure(candidate), measure(original)
                    baselines.append(baseline)
                    candidates.append(optimized)
                baseline, optimized = median(baselines), median(candidates)
                ratios = [round(b/c, 3) for b, c in zip(baselines, candidates)]
                print(f"LoRA out={out} tokens={args.tokens} amp={amp}: "
                      f"base={baseline:.3f}ms candidate={optimized:.3f}ms "
                      f"speedup={baseline/optimized:.3f}x policy={args.policy} "
                      f"paired_speedups={ratios}", flush=True)
    finally:
        unpatch_lora()


if __name__ == "__main__":
    main()
