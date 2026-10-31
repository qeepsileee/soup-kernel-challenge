"""ABBA frozen-head diagnostic; never changes the official training task."""
import argparse
import time
from statistics import median


def main():
    import torch
    from .chunked_loss import chunked_head_loss
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256])
    args = parser.parse_args()
    if args.iterations < 1 or any(c < 1 for c in args.chunks):
        parser.error("iterations and chunks must be positive")
    torch.manual_seed(910)
    head = torch.nn.Linear(1536, 151936, bias=False, device="cuda", dtype=torch.float16)
    head.weight.requires_grad_(False)
    # Model after prepare_model_for_kbit_training: FP32 storage, FP16 compute.
    head.float()
    x = torch.randn(4, 512, 1536, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(151936, (4, 512), device="cuda")
    labels[:, :256] = -100

    def step(chunk, eager):
        x.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = chunked_head_loss(x, head, labels, chunk_size=chunk,
                                       precompute_grad=eager)
        (loss * 65536.).backward()

    def measure(chunk, eager):
        for _ in range(2):
            step(chunk, eager)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.iterations):
            step(chunk, eager)
        torch.cuda.synchronize()
        return ((time.perf_counter() - started) * 1000 / args.iterations,
                torch.cuda.max_memory_allocated() / 2**20)

    print(torch.cuda.get_device_name(), "B=4 T=512 D=1536 V=151936; 50% prompt")
    for chunk in args.chunks:
        for eager in (False, True):
            baseline, candidate = [], []
            for trial in range(4):
                if trial % 2:
                    c, b = measure(chunk, eager), measure(64, False)
                else:
                    b, c = measure(64, False), measure(chunk, eager)
                baseline.append(b)
                candidate.append(c)
            base, cand = median(v[0] for v in baseline), median(v[0] for v in candidate)
            print(f"chunk={chunk} precompute={eager}: base={base:.2f}ms candidate={cand:.2f}ms "
                  f"speedup={base/cand:.3f}x peak_allocated={median(v[1] for v in candidate):.0f}MiB "
                  f"pairs={[round(b[0]/c[0], 3) for b,c in zip(baseline,candidate)]}", flush=True)
    print("Diagnostic only; full speed/quality/memory must use the official harness.")


if __name__ == "__main__":
    main()
