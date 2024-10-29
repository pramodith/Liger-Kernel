from test.utils import assert_verbose_allclose, set_seed, supports_bfloat16
from typing import Optional

import pytest
import torch

from liger_kernel.transformers.functional import liger_tvd
from liger_kernel.transformers.tvd import LigerTVD, LigerTVDFunction

set_seed(42)


class TVD(torch.nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
        reduction_mode: str = "batchmean",
    ):
        super(TVD, self).__init__()
        self.ignore_index = ignore_index
        self.dtype = dtype
        self.reduction_mode = reduction_mode

    def forward(
        self,
        log_student: torch.Tensor,  # input
        log_teacher: torch.Tensor,  # target
        label: Optional[torch.Tensor] = None,
    ):
        
        # student_probs = torch.exp(log_student.to(torch.float))
        # teacher_probs = torch.exp(log_teacher.to(torch.float))
        token_wise_loss = 0.5 * torch.abs((log_student - log_teacher))

        if self.reduction_mode == "sum":
            loss = torch.sum(token_wise_loss)
        elif self.reduction_mode == "mean":
            loss = torch.sum(token_wise_loss)/(log_student.size(0) * log_student.size(1))
        elif self.reduction_mode == "batchmean":
            loss = torch.sum(token_wise_loss)/(log_student.size(0))
        elif self.reduction_mode == "none":
            loss = token_wise_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction_mode}")

        if label:
            ignore_mask = label != self.ignore_index
            loss = loss * ignore_mask

        return loss.to(self.dtype)


_SHAPE_PARAMS = (
    "B, T, V",
    [
        (2, 1, 32000),  # llama2, mistral
        (2, 4096, 32000),  # llama2, mistral
        # weird shape
        (41, 401, 1271),
        pytest.param(
            1,
            4096,
            128256,
            marks=pytest.mark.skipif(
                torch.cuda.get_device_properties(0).total_memory
                < 36 * 1000 * 1000 * 1000,
                reason="This test requires a GPU with at least 36GB of memory",
            ),
        ),
        (3, 423, 32000),
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        # pytest.param(
        #     torch.bfloat16,
        #     1e-8,
        #     5e-2,
        #     marks=pytest.mark.skipif(
        #         not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
        #     ),
        # ),
        (torch.float32, 1e-8, 1e-6),
        (torch.float16, 1e-3, 1e-3),
    ],
)


def _test_correctness_once(
    target_tvd,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    reduction_mode,
    is_last_layer=True,
    device="cuda",
):
    torch_tvd = TVD(dtype=dtype, reduction_mode=reduction_mode)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)
    x3 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).softmax(dim=-1)

    output = torch_tvd(x1, target)
    output2 = target_tvd(x2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)
    # symmetry
    output3 = target_tvd(target, x3)
    assert torch.allclose(output3, output2, atol=atol, rtol=rtol)
    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0

    if reduction_mode == "none":
        return
    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)

def _test_correctness_with_ignore_index_once(
    target_tvd,
    ignore_index,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    reduction_mode,
    device="cuda",
):
    torch_tvd = TVD(ignore_index=ignore_index, dtype=dtype, reduction_mode=reduction_mode)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).softmax(dim=-1)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output = torch_tvd(x1, target, label)
    output2 = target_tvd(x2, target, label)
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()

    if reduction_mode == "none":
        return
    # assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


# def _test_correctness_functional(
#     B, T, V, ignore_index, is_last_layer, dtype, atol, rtol, device="cuda"
# ):
#     input = torch.randn(
#         B * T, V, device=device, dtype=dtype, requires_grad=True
#     ).log_softmax(dim=-1)

#     x1 = input.detach().clone().requires_grad_(True)
#     x2 = input.detach().clone().requires_grad_(True)

#     with torch.no_grad():
#         target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)

#     label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

#     # Assign some random number of elements as ignore_index
#     num_elements_to_assign = torch.randint(
#         1, B * T // 2, (1,)
#     ).item()  # Random number of elements to set to ignore_index
#     indices_to_assign = torch.randperm(B * T)[
#         :num_elements_to_assign
#     ]  # Randomly select indices
#     label[indices_to_assign] = ignore_index

#     output = LigerTVDFunction.apply(x1, target, label, ignore_index)
#     output2 = liger_tvd(x2, target, label, ignore_index)
#     assert torch.allclose(output, output2, atol=atol, rtol=rtol)
#     if (
#         not is_last_layer
#     ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
#         output = output * 2.0
#         output2 = output2 * 2.0
#     output.backward()
#     output2.backward()
#     assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize("reduction_mode", ["batchmean", "sum", "mean", "none"])
def test_correctness(B, T, V, dtype, atol, rtol, reduction_mode):
    liger_tvd = LigerTVD(reduction_mode=reduction_mode)
    _test_correctness_once(liger_tvd, B, T, V, dtype, atol, rtol, reduction_mode)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize("reduction_mode", ["batchmean", "sum", "mean", "none"])
def test_correctness_not_last(B, T, V, dtype, atol, rtol, reduction_mode):
    liger_tvd = LigerTVD(reduction_mode=reduction_mode)

    _test_correctness_once(liger_tvd, B, T, V, dtype, atol, rtol, reduction_mode, is_last_layer=False)


# @pytest.mark.parametrize(*_SHAPE_PARAMS)
# @pytest.mark.parametrize(*_DTYPE_PARAMS)
# @pytest.mark.parametrize("ignore_index", [2, 42])
# def test_correctness_with_ignore_index(B, T, V, ignore_index, dtype, atol, rtol):
#     liger_tvd = LigerTVD(ignore_index=ignore_index)
#     _test_correctness_with_ignore_index_once(
#         liger_tvd, ignore_index, B, T, V, dtype, atol, rtol
#     )


# @pytest.mark.parametrize(*_SHAPE_PARAMS)
# @pytest.mark.parametrize(*_DTYPE_PARAMS)
# @pytest.mark.parametrize(
#     "ignore_index, is_last_layer",
#     [
#         (2, False),
#         (42, True),
#     ],
# )
# def test_correctness_functional(
#     B, T, V, ignore_index, is_last_layer, dtype, atol, rtol
# ):
#     _test_correctness_functional(
#         B, T, V, ignore_index, is_last_layer, dtype, atol, rtol
#     )
