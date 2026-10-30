"""Explicit diagnostic CLI, never imported by apply() or used for scoring.

Wrap the existing short profiled run with module labels and record_shapes.
Do not edit harness/task/tests or report these instrumented times as a score.
"""
import json
import os
from pathlib import Path
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    # Same single-GPU/allocator environment as the official harness; set before torch.
    sys.path.insert(0, str(repo / "harness"))
    from common import ENV
    os.environ.update(ENV)
    import torch
    import yaml
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from harness import _train

    out = repo / "runs_cast_trace"
    out.mkdir(exist_ok=True)
    config = yaml.safe_load((repo / "task/task.yaml").read_text())
    config["data"]["train"] = str(repo / "data/train.jsonl")
    config["output"] = str(out / "adapter")
    config_path = out / "diagnostic.yaml"
    config_path.write_text(yaml.safe_dump(config))

    captures, modules, handles = [], {}, []
    original_profile = torch.profiler.profile

    def profile(*args, **kwargs):
        kwargs["record_shapes"] = True
        p = original_profile(*args, **kwargs)
        captures.append(p)
        return p

    original_setup = SFTTrainerWrapper._setup_transformers

    def setup(self, *args, **kwargs):
        result = original_setup(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if not isinstance(model, torch.nn.Module):
            return result
        for name, module in model.named_modules():
            if type(module).__name__ not in ("Linear", "Linear4bit", "Qwen2RMSNorm", "Embedding"):
                continue
            stack = []

            def before(mod, inputs, label=name, contexts=stack):
                x = next((v for v in inputs if isinstance(v, torch.Tensor)), None)
                weight = getattr(mod, "weight", None)
                modules[label] = {"class": type(mod).__name__,
                    "input_dtype": str(x.dtype) if x is not None else None,
                    "input_shape": list(x.shape) if x is not None else None,
                    "weight_dtype": str(weight.dtype) if weight is not None else None,
                    "compute_dtype": str(getattr(mod, "compute_dtype", None))}
                ctx = torch.profiler.record_function("SOUP_MODULE:" + label)
                ctx.__enter__()
                contexts.append(ctx)

            def after(mod, inputs, output, contexts=stack):
                if contexts:
                    contexts.pop().__exit__(None, None, None)

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
        return result

    torch.profiler.profile = profile
    SFTTrainerWrapper._setup_transformers = setup
    argv = sys.argv[:]
    sys.argv = ["trace_casts", "--config", str(config_path), "--metrics", str(out / "metrics.json"),
                "--solution", "--profile", str(out / "profile.json")]
    try:
        try:
            _train.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv = argv
        torch.profiler.profile = original_profile
        SFTTrainerWrapper._setup_transformers = original_setup
        for handle in handles:
            handle.remove()

    groups = {}
    for prof in captures:
        for e in prof.events():
            if e.name not in ("aten::copy_", "aten::_to_copy", "aten::clone", "aten::contiguous", "aten::add", "aten::mul"):
                continue
            parent = e.cpu_parent
            location = "backward_or_unlabelled"
            parents = []
            while parent is not None:
                parents.append(parent.name)
                if parent.name.startswith("SOUP_MODULE:"):
                    location = parent.name.removeprefix("SOUP_MODULE:")
                    break
                parent = parent.cpu_parent
            key = (e.name, location, str(e.input_shapes))
            group = groups.setdefault(key, {"op": e.name, "module": location,
                "shapes": e.input_shapes, "count": 0, "gpu_us": 0., "cpu_us": 0.,
                "parents": parents[:5]})
            group["count"] += 1
            group["gpu_us"] += float(getattr(e, "self_device_time_total", 0.))
            group["cpu_us"] += float(e.self_cpu_time_total)
    rows = sorted(groups.values(), key=lambda r: r["gpu_us"], reverse=True)
    path = out / "cast_sources.json"
    path.write_text(json.dumps({"operations": rows, "modules": modules}, indent=2))
    print("\n=== DIAGNOSTIC: top copy/cast/add/mul sources (not a scored run) ===")
    for row in rows[:30]:
        print(f"{row['gpu_us']/1000:8.2f}ms | {row['count']:5d} | {row['op']} | {row['module']} | {row['shapes']}")
    print("Detailed module dtype information:", path)


if __name__ == "__main__":
    main()
