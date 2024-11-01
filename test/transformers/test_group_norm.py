import pytest
import torch

from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.transformers.functional import liger_group_norm
from liger_kernel.transformers.group_norm import LigerGroupNorm


@pytest.mark.parametrize(
    "batch_size, num_channels, num_groups, hidden_size",
    [
        (2, 1, 1, 4),
        # (2, 4, 2, 128),
        # (16, 12, 3, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        # (torch.float16, 1e-3, 1e-3),
    ],
)
def test_liger_group_norm(batch_size, num_channels, num_groups, hidden_size, dtype, atol, rtol):
    torch.manual_seed(0)

    _tensor = torch.randn(
        batch_size, num_channels, hidden_size, dtype=dtype, device="cuda", requires_grad=True
    )

    liger_x = _tensor.clone().detach().requires_grad_(True)

    liger_ln = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to(dtype).cuda()
    torch_ln = torch.nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, eps=1e-6).to(dtype).cuda()
    
    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(liger_x,)
    torch_output = torch_ln(_tensor)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(_tensor)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)
    # print(liger_x.grad)
    # print(_tensor.grad)
    # assert torch.allclose(liger_x.grad, _tensor.grad, atol=atol, rtol=rtol)
    # # assert torch.allclose(
    # #     liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol
    # # )
    print(grad_output)
    print(grad_output.sum())
    print(liger_ln.bias.grad)
    print(torch_ln.bias.grad)
    assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol)