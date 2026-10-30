import pytest
import torch

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from solution.rmsnorm import FrozenRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs CUDA GPU"
)

TOLERANCE = {
    torch.float16: 2e-3,
    torch.float32: 1e-4
}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [
    (4, 512, 8960),
    (3, 77, 1001)
])
def test_rmsnorm_forward_backward(dtype, shape):
    torch.manual_seed(42)

    d = shape[-1]
    eps = 1e-6

    module = Qwen2RMSNorm(d, eps=eps).cuda().to(dtype)
    module.weight.requires_grad_(False)

    x_ref = torch.randn(
        shape, device="cuda", dtype=dtype, requires_grad=True
    )
    x_triton = x_ref.detach().clone().requires_grad_(True)

    grad = torch.randn_like(x_ref)

    y_ref = module(x_ref)
    y_ref.backward(grad)

    y_triton = FrozenRMSNorm.apply(
        x_triton, module.weight, eps
    )
    y_triton.backward(grad)

    tol = TOLERANCE[dtype]

    torch.testing.assert_close(
        y_triton.float(),
        y_ref.float(),
        rtol=tol,
        atol=tol
    )

    torch.testing.assert_close(
        x_triton.grad.float(),
        x_ref.grad.float(),
        rtol=tol,
        atol=tol
    )