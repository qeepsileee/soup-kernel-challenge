"""Isolated candidate benchmark; doesn't patch the shared chunked-loss files."""
import argparse
import time
from statistics import median


def main():
    import torch
    from .parallel_ce import parallel_cross_entropy
    from .fused_ce import fused_cross_entropy
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    torch.manual_seed(609)
    for n in (64, 256):
        for dtype in (torch.float16, torch.float32):
            z = torch.randn((n, 151936), device="cuda", dtype=dtype)
            target = torch.randint(151936, (n,), device="cuda")
            target[::3] = -100
            # Statistics-only mode used by chunked forward. Identical work on both.
            def run(fn):
                return fn(z, target, denom=n, needs_grad=False)

            def torch_reference(logits, targets, denom, needs_grad):
                logp = logits.float().log_softmax(-1)
                valid = targets != -100
                loss = -logp.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)
                loss = torch.where(valid, loss, 0.)
                entropy = -(logp.exp() * logp).sum(-1)
                hits = ((logits.argmax(-1) == targets) & valid).int()
                return loss, entropy, hits, loss.sum() / denom, None

            ref = run(torch_reference)
            actual = run(parallel_cross_entropy)
            for i in (0, 1, 2, 3):
                torch.testing.assert_close(actual[i], ref[i], rtol=3e-5, atol=3e-5)
            serial = run(fused_cross_entropy)
            baseline_fn = fused_cross_entropy
            label = "serial"
            if not torch.all(torch.isfinite(serial[1])):
                print("Serial CE entropy is nonfinite; using PyTorch reference for timing.", flush=True)
                baseline_fn, label = torch_reference, "PyTorch"

            def measure(fn):
                for _ in range(5):
                    run(fn)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    run(fn)
                torch.cuda.synchronize()
                return (time.perf_counter() - start) * 1000 / args.iterations

            pairs = []
            for trial in range(4):
                if trial % 2:
                    candidate, baseline = measure(parallel_cross_entropy), measure(baseline_fn)
                else:
                    baseline, candidate = measure(baseline_fn), measure(parallel_cross_entropy)
                pairs.append((baseline, candidate))
            a, b = median(x[0] for x in pairs), median(x[1] for x in pairs)
            print(f"CE stats N={n} dtype={dtype}: {label}={a:.3f}ms tiled={b:.3f}ms "
                  f"speedup={a/b:.3f}x pairs={[round(x/y,3) for x,y in pairs]}", flush=True)


if __name__ == "__main__":
    main()
