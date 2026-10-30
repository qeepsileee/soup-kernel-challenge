# Candidate: remove MLP activation dtype round trips

Previous validated training code: d863fed (all 10 checks, own share 5.6%).
This new MLP bridge awaits GPU validation and a fresh official comparison.

User-provided diagnostic: [4,512,8960] copy events were 117.77ms over 252 calls;
84 calls at each other sequence length strongly suggest three copies per MLP
layer (28 layers), though backward/unlabelled events alone do not prove the
direction of each copy. The source path provides the mechanism: Linear4bit
computes in FP16, returns input dtype FP32; the example SwiGLU stores in the gate
dtype, then down_proj casts its FP32 input back to FP16.

mlp_bridge.py directly calls the SAME bnb.matmul_4bit entry point for each frozen
projection, preserves its custom/cuBLAS dispatch, and reuses the unchanged
example SwiGLU. Intermediate values originating in FP16 GEMM are kept in FP16;
the activation itself is evaluated by the existing example kernel in FP32 and
its output is rounded where down_proj previously rounded it. Final residual
output is FP32. No LoRA parameters, dropout, optimizer, attention, data, steps
or metrics are changed. Unsupported MLP layouts/dtypes fall back to the prior
forward. No global scratch or per-layer persistent dense weights are introduced.

Keep separate x.to(fp16) calls for gate and up: sharing a cast changes backward
accumulation from FP32 to FP16. The saved gate/up activations can use half the
storage because their values originally came from FP16, not arbitrary FP32.
End-to-end peak savings depend on live tensors and require measurement.

The reused/modified calling convention of example SwiGLU does NOT turn its
kernels into original team kernels. The official own-kernel share must remain
>=5% through NF4/RMSNorm/entropy, measured in every run. Do not claim this bridge
itself solves own-kernel coverage.

GPU test: compare forward bitwise against the previous example path, compare
scaled backward gradients, and count saved large activations for seq 17/129/512.
Local syntax checks/CPU skips cannot validate CUDA behavior.

```bash
python -m pytest solution -q
python harness/run_task.py --repeats 2 --out runs_mlp_bridge
python harness/check_rules.py --result runs_mlp_bridge/result.json
```

Use a fresh idle GPU and leave previous runs/ artifacts intact. The diagnostic
profiling table is not a speed benchmark. Next targets are the repeated lm_head
weight cast and the full logits copies; coordinate with the loss workstream.
