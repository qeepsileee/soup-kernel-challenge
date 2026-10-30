"""Kaggle: python -m solution.bench_nf4 --iterations 100 > runs/nf4_micro.json

Microbenchmark only, NOT a substitute for the unchanged official harness.
"""

import argparse
import json
from importlib.metadata import version


def main():
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    from .nf4 import dequantize_nf4, patch_dequantize, unpatch_dequantize
    from .nf4_matmul import patch_input_grad, unpatch_input_grad
    from .nf4_format import qwen_projection_shapes

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--fused-dx", action="store_true",
                        help="also benchmark candidate fused NF4 input-gradient kernel")
    args = parser.parse_args()
    if args.iterations < 1 or args.tokens < 1:
        parser.error("iterations and tokens must be positive")
    torch.manual_seed(31)
    unpatch_dequantize()
    unpatch_input_grad()
    reference = functional.dequantize_4bit

    def time_ms(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            fn()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) / args.iterations

    result = {"gpu": torch.cuda.get_device_name(),
              "versions": {p: version(p) for p in ("torch", "triton", "bitsandbytes")},
              "tokens": args.tokens, "fused_dx": args.fused_dx, "layers": []}
    for name, shape in qwen_projection_shapes().items():
        weight = torch.randn(shape, device="cuda", dtype=torch.float16)
        packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
        del weight
        expected, actual = reference(packed, state), dequantize_nf4(packed, state)
        max_error = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        del expected, actual
        destination = torch.empty(shape, device="cuda", dtype=torch.float16)
        base_ms = time_ms(lambda: reference(packed, state, out=destination))
        ours_ms = time_ms(lambda: dequantize_nf4(packed, state, out=destination))
        del destination
        x = torch.randn((args.tokens, shape[1]), device="cuda", dtype=torch.float16,
                        requires_grad=True)
        grad = torch.randn((args.tokens, shape[0]), device="cuda", dtype=torch.float16)

        def step():
            x.grad = None
            bnb.matmul_4bit(x, packed, state).backward(grad)

        base_step_ms = time_ms(step)
        patch_dequantize()
        if args.fused_dx:
            patch_input_grad()
        try:
            ours_step_ms = time_ms(step)
        finally:
            unpatch_input_grad()
            unpatch_dequantize()
        result["layers"].append({"name": name, "shape_out_in": shape,
            "max_abs_error": max_error, "decode_base_ms": base_ms,
            "decode_candidate_ms": ours_ms, "decode_speedup": base_ms / ours_ms,
            "forward_backward_base_ms": base_step_ms,
            "forward_backward_candidate_ms": ours_step_ms,
            "forward_backward_speedup": base_step_ms / ours_step_ms,
            "quant_state": {"blocksize": state.blocksize, "dtype": str(state.dtype),
                "nested": state.nested, "scale_blocksize": state.state2.blocksize,
                "packed_shape": list(packed.shape), "scales_shape": list(state.absmax.shape)}})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
