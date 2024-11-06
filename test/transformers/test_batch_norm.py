import random

import pytest
import torch

from liger_kernel.transformers.batch_norm import LigerBatchNorm

random_batch_size = random.randint(1, 64)
random_num_features = random.randint(1, 8193)
random_seq_length = random.randint(1, 512)


@pytest.mark.parametrize(
    "batch_size, num_features, seq_len",
    [   
        (2, 3, None),
        (64, 1024, None),
        (2, 3, 1),
        (8, 7, 64),
        (random_batch_size, random_num_features, random_seq_length),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_liger_batch_norm(
    batch_size, num_features, seq_len,  dtype, atol, rtol
):
    torch.manual_seed(0)

    if seq_len:
        # Create a 3D tensor
        _tensor = torch.randn(
            batch_size, num_features, seq_len, dtype=dtype, device="cuda"
        )
    else:
        # Create a 2D tensor
        _tensor = torch.randn(
            batch_size, num_features, dtype=dtype, device="cuda"
        )

    liger_x = _tensor.clone().detach().requires_grad_(True)
    torch_x = _tensor.clone().detach().requires_grad_(True)

    liger_bn = LigerBatchNorm(num_features, eps=1e-6).to(dtype).cuda()
    
    torch_bn = (
        torch.nn.BatchNorm1d(num_features=num_features, eps=1e-6)
        .to(dtype)
        .cuda()
    )

    with torch.no_grad():
        liger_bn.weight.copy_(liger_bn.weight)
        liger_bn.bias.copy_(liger_bn.bias)

    liger_output = liger_bn(
        liger_x,
    )
    torch_output = torch_bn(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)
    
    assert torch.allclose(liger_bn.running_mean, torch_bn.running_mean, atol=atol, rtol=rtol)
    assert torch.allclose(liger_bn.running_var, torch_bn.running_var, atol=atol, rtol=rtol)
    grad_output = torch.randn_like(torch_x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)
    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(
        liger_bn.bias.grad, torch_bn.bias.grad, atol=atol, rtol=rtol
    ), "Bias grads different"
    assert torch.allclose(
        liger_bn.weight.grad, torch_bn.weight.grad, atol=atol, rtol=rtol
    ), "Weight grads different"
