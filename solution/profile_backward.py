"""Explicit diagnostic only: attribute frozen-projection backward CUDA work.

Runs the harness's short profiler unchanged, with additional record_function
ranges. Never imported by apply(), and never used as a scored measurement.
"""
import argparse
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_backward_profile")
    parser.add_argument("--head-chunk", type=int, default=128)
    parser.add_argument("--recompute-head", action="store_true")
    parser.add_argument("--half-storage", action="store_true")
    args = parser.parse_args()
    if args.head_chunk < 1:
        parser.error("head-chunk must be positive")
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    # Must precede importing torch. Diagnostics use exactly one GPU.
    sys.path.insert(0, str(repo / "harness"))
    from common import ENV
    os.environ.update(ENV)
    settings = {"SOUP_EXPERIMENT_HEAD_CHUNK": str(args.head_chunk),
                "SOUP_EXPERIMENT_PRECOMPUTE_HEAD": "0" if args.recompute_head else "1",
                "SOUP_EXPERIMENT_HALF_STORAGE": "1" if args.half_storage else "0"}
    os.environ.update(settings)
    print("Diagnostic settings:", settings, flush=True)
    import torch
    import yaml
    from torch.autograd import DeviceType
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from harness import _train
    from . import nf4_linear, chunked_loss

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((repo / "task/task.yaml").read_text())
    config["data"]["train"] = str(repo / "data/train.jsonl")
    config["output"] = str(out / "adapter")
    config_path = out / "diagnostic.yaml"
    config_path.write_text(yaml.safe_dump(config))

    phase = ContextVar("nf4_diagnostic_phase", default="unlabelled")
    weights, captures, restores = {}, [], []

    def replace(obj, name, value):
        # Preserve the staticmethod descriptor, not only its bound function.
        restores.append((obj, name, vars(obj)[name]))
        setattr(obj, name, value)

    original_setup = SFTTrainerWrapper._setup_transformers

    def setup(self, *a, **kw):
        result = original_setup(self, *a, **kw)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            for name, module in model.named_modules():
                w = getattr(module, "weight", None)
                state = getattr(w, "quant_state", None)
                if state is not None:
                    projection = name.removesuffix(".base_layer").rsplit(".", 1)[-1]
                    weights[w.data_ptr()] = f"{projection}[{','.join(map(str, state.shape))}]"
        return result

    original_forward = nf4_linear.NF4Linear.forward
    original_backward = nf4_linear.NF4Linear.backward
    original_decode = nf4_linear.dequantize_nf4
    original_matmul = torch.matmul

    def forward(ctx, x, packed, bias, state):
        label = weights.get(packed.data_ptr(), f"unknown[{','.join(map(str, state.shape))}]")
        ctx._diagnostic_projection = label
        token = phase.set("forward:" + label)
        try:
            with torch.profiler.record_function("DIAG:nf4.forward:" + label):
                return original_forward(ctx, x, packed, bias, state)
        finally:
            phase.reset(token)

    def backward(ctx, grad):
        label = getattr(ctx, "_diagnostic_projection", "unknown")
        token = phase.set("backward:" + label)
        try:
            with torch.profiler.record_function("DIAG:nf4.backward:" + label):
                return original_backward(ctx, grad)
        finally:
            phase.reset(token)

    def decode(*a, **kw):
        with torch.profiler.record_function("DIAG:nf4.decode:" + phase.get()):
            return original_decode(*a, **kw)

    def matmul(*a, **kw):
        current = phase.get()
        if current.startswith("backward:"):
            with torch.profiler.record_function("DIAG:nf4.dx_gemm:" + current):
                return original_matmul(*a, **kw)
        return original_matmul(*a, **kw)

    def wrap_head(cls, mode):
        for method in ("forward", "backward"):
            original = getattr(cls, method)

            def wrapper(*a, _fn=original, _tag=f"DIAG:head.{mode}.{method}", **kw):
                with torch.profiler.record_function(_tag):
                    return _fn(*a, **kw)

            replace(cls, method, staticmethod(wrapper))

    original_profile = torch.profiler.profile

    def profile(*a, **kw):
        kw["record_shapes"] = True
        prof = original_profile(*a, **kw)
        captures.append(prof)
        return prof

    argv = sys.argv[:]
    try:
        replace(SFTTrainerWrapper, "_setup_transformers", setup)
        replace(nf4_linear.NF4Linear, "forward", staticmethod(forward))
        replace(nf4_linear.NF4Linear, "backward", staticmethod(backward))
        replace(nf4_linear, "dequantize_nf4", decode)
        replace(torch, "matmul", matmul)
        replace(torch.profiler, "profile", profile)
        wrap_head(chunked_loss._ChunkedHeadLossFunction, "recompute")
        wrap_head(chunked_loss._PrecomputedHeadLossFunction, "precompute")
        sys.argv = ["profile_backward", "--config", str(config_path),
                    "--metrics", str(out / "metrics.json"), "--solution",
                    "--profile", str(out / "profile.json")]
        try:
            _train.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv = argv
        for obj, name, value in reversed(restores):
            setattr(obj, name, value)

    ranges, copies = {}, {}
    kernels = []
    for prof in captures:
        kernels.extend(_train.kernel_table(prof))
        for event in prof.events():
            if event.device_type != DeviceType.CPU:
                continue
            if event.name.startswith("DIAG:"):
                row = ranges.setdefault(event.name, {"range": event.name, "calls": 0,
                                                     "gpu_us_inclusive": 0., "cpu_us_inclusive": 0.})
                row["calls"] += 1
                row["gpu_us_inclusive"] += float(event.device_time_total)
                row["cpu_us_inclusive"] += float(event.cpu_time_total)
            if event.name not in ("aten::_to_copy", "aten::copy_", "aten::clone", "aten::contiguous"):
                continue
            parent, location = event.cpu_parent, "outside_instrumented_ranges"
            while parent is not None:
                if parent.name.startswith("DIAG:"):
                    location = parent.name
                    break
                parent = parent.cpu_parent
            key = (event.name, location, str(event.input_shapes))
            row = copies.setdefault(key, {"op": event.name, "location": location,
                                         "shapes": event.input_shapes, "calls": 0, "gpu_us_self": 0.})
            row["calls"] += 1
            row["gpu_us_self"] += float(event.self_device_time_total)
    total_us = sum(k["time_us"] for k in kernels)
    decode_kernel_us = sum(k["time_us"] for k in kernels if k["name"] == "nf4_decode_kernel")
    ranges = sorted(ranges.values(), key=lambda r: -r["gpu_us_inclusive"])
    copies = sorted(copies.values(), key=lambda r: -r["gpu_us_self"])
    categories = {}
    for category in ("nf4.backward:", "nf4.decode:backward:", "nf4.dx_gemm:",
                     "nf4.forward:", "nf4.decode:forward:", "head."):
        us = sum(r["gpu_us_inclusive"] for r in ranges if r["range"].startswith("DIAG:" + category))
        categories[category] = {"gpu_ms": us / 1000,
                                "percent_of_kernel_time": 100 * us / total_us if total_us else None}
    report = {"diagnostic_only": True, "settings": settings, "active_steps": _train.PROFILE_ACTIVE,
              "total_kernel_ms": total_us / 1000, "categories": categories,
              "nf4_decode_kernel_ms_unpartitioned": decode_kernel_us / 1000,
              "ranges_inclusive": ranges, "copy_ops_self": copies,
              "kernels": sorted(kernels, key=lambda r: -r["time_us"])}
    path = out / "backward_breakdown.json"
    path.write_text(json.dumps(report, indent=2))
    print("\n=== DIAGNOSTIC GPU WORK (not a score) ===")
    for name, row in categories.items():
        percent = row["percent_of_kernel_time"]
        if "decode:" in name and row["gpu_ms"] == 0 and decode_kernel_us:
            print(f"{name:28} CUDA attribution unavailable (see raw decode kernel total)")
            continue
        share = f"{percent:.1f}%" if percent is not None else "unavailable"
        print(f"{name:28} {row['gpu_ms']:9.2f} ms | {share}")
    print("Ranges overlap: backward includes decode/dX GEMM. Do not sum all categories.")
    print(f"Raw nf4_decode_kernel: {decode_kernel_us/1000:.2f} ms; not partitioned into forward/backward.")
    print("=== TOP COPY/CAST SOURCES ===")
    for row in copies[:15]:
        print(f"{row['gpu_us_self']/1000:8.2f}ms | {row['calls']:5d} | "
              f"{row['op']} | {row['location']} | {row['shapes']}")
    print("=== TOP CUDA KERNELS (includes attention if significant) ===")
    for row in report["kernels"][:15]:
        print(f"{row['time_us']/1000:8.2f}ms | {row['count']:5d} | {row['name']}")
    if not any(r["gpu_us_inclusive"] > 0 for r in ranges):
        print("WARNING: no attributed CUDA time; inspect profiler linkage before drawing conclusions.")
    print("Report:", path)


if __name__ == "__main__":
    main()
