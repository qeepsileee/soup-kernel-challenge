"""ABBA: skip only unconsumed padding in the frozen head; not an official score."""
import argparse
from statistics import median
import time


def main():
    import torch
    from unittest.mock import patch
    from . import chunked_loss as module
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    assert torch.cuda.is_available()
    torch.manual_seed(972)
    head = torch.nn.Linear(1536, 151936, bias=False, device="cuda", dtype=torch.float16)
    head.weight.requires_grad_(False)
    x = torch.randn(4, 512, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    labels = torch.randint(151936, (4, 512), device="cuda")
    print(torch.cuda.get_device_name(), "B=4 T=512 D=1536 V=151936 chunk=128; validated precomputed head", flush=True)

    def step(candidate, mask):
        x.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = module.chunked_head_loss(x, head, labels, chunk_size=128,
                precompute_grad=True, attention_mask=mask if candidate else None)
        (loss * 65536.).backward()

    def measure(candidate, mask):
        for _ in range(2):
            step(candidate, mask)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            step(candidate, mask)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start) * 1000 / args.iterations
        x.grad = None
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(candidate, mask)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated()-before) / 2**20
        return elapsed, peak

    for lengths in ((512, 512, 512, 512), (512, 384, 256, 128), (320, 256, 192, 128)):
        mask = torch.zeros(4, 512, device="cuda", dtype=torch.long)
        labels.random_(0, 151936)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
            labels[row, :length//2] = -100
            labels[row, length:] = -100
        target = module.shift_for_next_token(labels)
        projected = int(((mask != 0) | (target != -100)).sum().item())
        actual_rows = [0]
        original = module.fused_cross_entropy

        def counted(logits, *a, **kw):
            actual_rows[0] += logits.shape[0]
            return original(logits, *a, **kw)

        with patch.object(module, "fused_cross_entropy", counted):
            step(True, mask)
        assert actual_rows[0] == projected, "benchmark must actually skip only unconsumed rows"
        bases, candidates = [], []
        for trial in range(4):
            if trial % 2:
                c, b = measure(True, mask), measure(False, mask)
            else:
                b, c = measure(False, mask), measure(True, mask)
            bases.append(b)
            candidates.append(c)
        base, candidate = median(v[0] for v in bases), median(v[0] for v in candidates)
        print(f"lengths={lengths} projected={projected}/2048: base={base:.2f}ms "
              f"candidate={candidate:.2f}ms speedup={base/candidate:.3f}x "
              f"pairs={[round(b[0]/c[0],3) for b,c in zip(bases,candidates)]}", flush=True)
        print(f"  incremental allocated peak: {median(v[1] for v in bases):.1f} -> "
              f"{median(v[1] for v in candidates):.1f}MiB", flush=True)
    print("Diagnostic component latency/local allocation only; full-task checks decide quality/time/memory.")


if __name__ == "__main__":
    main()
