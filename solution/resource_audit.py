"""Four-step diagnostic: unique saved activation storage + startup phases.

Never imported by apply(). Does not change task files. Instrumented runtime and
memory are NOT official scores. Python import timings are inclusive/overlapping.
"""
import argparse
import builtins
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid


class _Saved:
    __slots__ = ("tensor", "key", "owner")

    def __init__(self, tensor, key, owner):
        self.tensor, self.key, self.owner = tensor, key, owner

    def __del__(self):
        self.owner.release(self.key)


class SavedInventory:
    def __init__(self, torch):
        self.torch = torch
        self.scope = ContextVar("saved_inventory_scope", default="unlabelled")
        self.parameters = set()
        self.live, self.peak = {}, []
        self.live_bytes = self.peak_bytes = 0
        self.forwards = []

    def key(self, tensor):
        return tensor.device.index, tensor.untyped_storage().data_ptr()

    def refresh_parameters(self, model):
        # Runs on first real forward, AFTER half-storage changes parameter data.
        self.parameters = {self.key(t) for t in list(model.parameters()) + list(model.buffers())
                           if t.is_cuda and t.numel()}

    def pack(self, tensor):
        if not tensor.is_cuda or not tensor.numel():
            return tensor
        key = self.key(tensor)
        if key in self.parameters:
            return tensor
        if key not in self.live:
            self.live[key] = {"references": 0, "bytes": tensor.untyped_storage().nbytes(),
                              "scope": self.scope.get(), "dtype": str(tensor.dtype),
                              "shape": list(tensor.shape)}
            self.live_bytes += self.live[key]["bytes"]
        self.live[key]["references"] += 1
        if self.live_bytes > self.peak_bytes:
            self.peak_bytes = self.live_bytes
            self.peak = [dict(row) for row in self.live.values()]
        return _Saved(tensor, key, self)

    def unpack(self, saved):
        return saved.tensor if isinstance(saved, _Saved) else saved

    def release(self, key):
        row = self.live.get(key)
        if row is not None:
            row["references"] -= 1
            if not row["references"]:
                self.live_bytes -= row["bytes"]
                del self.live[key]

    def groups(self):
        groups = {}
        for row in self.peak:
            key = row["scope"], row["dtype"], tuple(row["shape"])
            group = groups.setdefault(key, {"scope": row["scope"], "dtype": row["dtype"],
                                             "shape": row["shape"], "storages": 0, "bytes": 0})
            group["storages"] += 1
            group["bytes"] += row["bytes"]
        return sorted(groups.values(), key=lambda row: -row["bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_resource_audit")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    print(f"[audit] starting standalone diagnostic, stopping after {args.steps} steps; not a score", flush=True)
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    sys.path.insert(0, str(repo / "harness"))
    from common import OFFLINE_ENV
    os.environ.update(OFFLINE_ENV)
    # YAML parsing is outside the diagnostic child-equivalent training phases.
    import yaml
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_tag = uuid.uuid4().hex[:12]
    cache = out / ("triton_cache_" + cache_tag)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / ("inductor_cache_" + cache_tag))
    config = yaml.safe_load((repo / "task/task.yaml").read_text())
    config["data"]["train"] = str(repo / "data/train.jsonl")
    config["output"] = str(out / "adapter")
    config_path = out / "diagnostic.yaml"
    config_path.write_text(yaml.safe_dump(config))
    from harness import _train

    original_import = builtins.__import__
    imports, depth, handles, restores = {}, {}, [], []
    tracked = {"torch", "transformers", "triton", "peft", "trl", "soup_cli.cli", "soup_cli.trainer.sft",
               "transformers.models.qwen2.modeling_qwen2", "transformers.modeling_utils",
               "transformers.trainer", "trl.trainer", "trl.trainer.sft_trainer"}
    state = {"inventory": None, "hook_context": None, "setup_installed": False,
             "trainer_installed": False, "installing": False, "phases": {}}
    started = time.perf_counter()

    def install_ready_hooks():
        if state["installing"]:
            return
        state["installing"] = True
        try:
            torch = sys.modules.get("torch")
            if torch is None or not hasattr(torch, "autograd"):
                return
            if state["inventory"] is None and not depth.get("torch", 0):
                inventory = SavedInventory(torch)
                hooks = torch.autograd.graph.saved_tensors_hooks(inventory.pack, inventory.unpack)
                hooks.__enter__()
                state["inventory"], state["hook_context"] = inventory, hooks
                original_empty = torch.empty
                def empty(*a, **kw):
                    if str(kw.get("device", "")).startswith("cuda"):
                        torch.empty = original_empty
                        begin = time.perf_counter()
                        result = original_empty(*a, **kw)
                        state["phases"]["first_cuda_allocation_wall_s"] = time.perf_counter() - begin
                        print("[audit] first CUDA allocation finished", state["phases"]["first_cuda_allocation_wall_s"], flush=True)
                        return result
                    return original_empty(*a, **kw)
                restores.append((torch, "empty", original_empty))
                torch.empty = empty
            inventory = state["inventory"]
            sft = sys.modules.get("soup_cli.trainer.sft")
            if inventory is not None and sft is not None and hasattr(sft, "SFTTrainerWrapper") and not state["setup_installed"]:
                cls = sft.SFTTrainerWrapper
                original_setup = cls._setup_transformers

                def setup(self, *a, **kw):
                    print("[audit] model setup starting", flush=True)
                    begin = time.perf_counter()
                    result = original_setup(self, *a, **kw)
                    state["phases"]["model_setup_wall_s"] = time.perf_counter() - begin
                    print("[audit] model setup finished", state["phases"]["model_setup_wall_s"], flush=True)
                    model = getattr(self, "model", None)
                    if not isinstance(model, torch.nn.Module):
                        return result
                    refreshed = [False]
                    def root_pre(module, inputs):
                        if not refreshed[0]:
                            inventory.refresh_parameters(module)
                            refreshed[0] = True
                    def root_post(module, inputs, output):
                        inventory.forwards.append({"saved_mib": inventory.live_bytes / 2**20,
                            "allocated_mib": torch.cuda.memory_allocated() / 2**20})
                    handles.append(model.register_forward_pre_hook(root_pre, prepend=True))
                    handles.append(model.register_forward_hook(root_post, always_call=True))
                    for name, module in model.named_modules():
                        if type(module).__name__ not in ("Linear", "Linear4bit", "Dropout", "Embedding",
                                                        "Qwen2RMSNorm", "Qwen2MLP", "Qwen2Attention"):
                            continue
                        suffix = re.sub(r"^\d+\.", "", name.rsplit("layers.", 1)[-1])
                        label, stack = suffix + "|" + type(module).__name__, []
                        def before(mod, inputs, label=label, stack=stack):
                            stack.append(inventory.scope.set(label))
                        def after(mod, inputs, output, stack=stack):
                            if stack:
                                inventory.scope.reset(stack.pop())
                        handles.append(module.register_forward_pre_hook(before))
                        handles.append(module.register_forward_hook(after, always_call=True))
                    return result

                restores.append((cls, "_setup_transformers", original_setup))
                cls._setup_transformers = setup
                state["setup_installed"] = True
            trainer = sys.modules.get("transformers.trainer")
            callback = sys.modules.get("transformers.trainer_callback")
            if trainer is not None and callback is not None and hasattr(trainer, "Trainer") and not state["trainer_installed"]:
                original_init = trainer.Trainer.__init__
                class Stop(callback.TrainerCallback):
                    def on_step_end(self, arguments, trainer_state, control, **kwargs):
                        if trainer_state.global_step >= args.steps:
                            control.should_training_stop = True
                def init(self, *a, **kw):
                    begin = time.perf_counter()
                    original_init(self, *a, **kw)
                    state["phases"]["trainer_init_wall_s"] = time.perf_counter() - begin
                    self.add_callback(Stop())
                restores.append((trainer.Trainer, "__init__", original_init))
                trainer.Trainer.__init__ = init
                state["trainer_installed"] = True
        finally:
            state["installing"] = False

    def observed_import(name, *a, **kw):
        outer = depth.get(name, 0) == 0
        depth[name] = depth.get(name, 0) + 1
        begin = time.perf_counter()
        succeeded = False
        try:
            result = original_import(name, *a, **kw)
            succeeded = True
            return result
        finally:
            depth[name] -= 1
            if succeeded and outer and name in tracked and name not in imports:
                imports[name] = time.perf_counter() - begin
                print(f"[audit] first import {name}: {imports[name]:.3f}s (inclusive)", flush=True)
            if succeeded:
                install_ready_hooks()

    argv = sys.argv[:]
    try:
        builtins.__import__ = observed_import
        sys.argv = ["resource_audit", "--config", str(config_path),
                    "--metrics", str(out / "metrics.json"), "--solution"]
        try:
            _train.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        builtins.__import__ = original_import
        sys.argv = argv
        for handle in handles:
            handle.remove()
        for obj, name, value in reversed(restores):
            setattr(obj, name, value)
        if state["hook_context"] is not None:
            state["hook_context"].__exit__(None, None, None)
    inventory = state["inventory"]
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    report = {"diagnostic_only": True, "requested_steps": args.steps,
              "actual_steps": metrics.get("steps"), "step_s": metrics.get("step_s"),
              "apply_s": metrics.get("apply_s"), "fresh_triton_cache": str(cache),
              "elapsed_instrumented_s": time.perf_counter() - started,
              "imports_inclusive_s": imports, "phases_wall_s": state["phases"],
              "saved_storage_peak_mib": inventory.peak_bytes / 2**20 if inventory else None,
              "forwards": inventory.forwards if inventory else [],
              "saved_storage_groups": inventory.groups() if inventory else []}
    (out / "resource_audit.json").write_text(json.dumps(report, indent=2))
    print("\n=== RESOURCE AUDIT: NOT A SCORE ===")
    print("Import wall times (inclusive, overlapping):", json.dumps(imports))
    print("Setup wall times (may overlap imports):", json.dumps(state["phases"]))
    print("Diagnostic step times:", report["step_s"], "apply_s:", report["apply_s"])
    print("Unique non-parameter saved storage peak MiB:", report["saved_storage_peak_mib"])
    for row in report["saved_storage_groups"][:25]:
        print(f"{row['bytes']/2**20:8.1f}MiB | {row['storages']:4d} storages | "
              f"{row['scope']} | {row['dtype']} | {row['shape']}")
    if not report["forwards"]:
        print("WARNING: no model forward snapshots; do not infer a memory breakdown.")
    print("Raw snapshot report:", out / "resource_audit.json")


if __name__ == "__main__":
    main()
