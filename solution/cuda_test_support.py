"""Availability for standalone Triton loss tests; bitsandbytes is not needed."""
import importlib.util

GPU_AVAILABLE = False
if all(importlib.util.find_spec(name) is not None for name in ("torch", "triton")):
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
