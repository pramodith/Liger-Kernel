import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt

MAX_FUSED_SIZE = 65536

@triton.jit
def _batch_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (num_features, seq_len, batch_size)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (num_features, seq_len, batch_size)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (num_features,)
    B_ptr,  # pointer to bias, shape (num_features,)
    Mean_ptr,  # pointer to mean, shape (num_features,)
    RSTD_ptr,  # pointer to rstd, shape (num_features,)
    Running_mean_ptr,  # pointer to running mean, shape (num_features,)
    Running_var_ptr,  # pointer to running var, shape (num_features,)
    n_cols,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
    momentum: tl.constexpr, # momentum for running mean and var
):
    """
    References:
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html    
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    feature_idx = tl.program_id(0)

    W_row = tl.load(W_ptr + feature_idx)
    B_row = tl.load(B_ptr + feature_idx)
    running_mean = tl.load(Running_mean_ptr + feature_idx)
    running_var = tl.load(Running_var_ptr + feature_idx)
    
    offsets = tl.arange(0, BLOCK_SIZE)

    # Compute mean and variance
    s = 0.0
    square_sum = 0.0
    for batch_idx in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offsets + batch_idx
        mask = col_offsets < n_cols
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        
        s += tl.sum(X_row, axis=0)
        square_sum += tl.sum(X_row * X_row, axis=0)
    
    mean = s / n_cols
    # variance = E[X^2] - E[X]^2
    var = (square_sum / n_cols) - (mean * mean)
    rstd = rsqrt(var + eps)
    tl.store(Mean_ptr + feature_idx, mean)
    tl.store(RSTD_ptr + feature_idx, rstd)

    # Update running mean and var that'll be used in inference
    running_mean = running_mean * (1 - momentum) + mean * momentum
    running_var = running_var * (1 - momentum) + var * momentum

    tl.store(Running_mean_ptr + feature_idx, running_mean)
    tl.store(Running_var_ptr + feature_idx, running_var)

    # Compute output
    for batch_idx in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offsets + batch_idx
        mask = col_offsets < n_cols
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=mean)
        Y_row = (X_row - mean) * rstd * W_row + B_row
        tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _batch_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    stride_dx,  # stride of each row in input grad
    stride_dw,  # stride of each row in weights grad
    stride_db,  # stride of each row in bias grad
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/batch_norm.py
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    for _ in range(row_start, row_end):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)


def batch_norm_forward(X, W, B, RM, RV, eps, momentum):
    """
    The forward pass of the batch normalization layer that calls the corresponding Triton kernels.
    :param X: input tensor
    :param W: weight tensor
    :param B: bias tensor
    :param RM: running mean tensor
    :param RV: running variance tensor
    :param eps: epsilon for numerical stability
    :param momentum: momentum for running mean and var
    """
    shape = X.shape

    if len(shape)==2:
        batch_size, num_features = shape
        seq_length = 1
    if len(shape)==3:
        batch_size, num_features, seq_length = shape
    X = X.view(-1, num_features)
    
    # We need to compute the mean and variance across elements in the same feature dimension across all samples in the batch
    X_T = X.T.contiguous()
    
    n_cols = batch_size * seq_length
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(batch_size*seq_length))
    Y = torch.empty((num_features, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(num_features, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(num_features, dtype=X.dtype, device=X.device)
    
    assert (
        X.shape[1] == W.shape[0]
    ), f"Incompatible hidden size dimension between input tensor with shape[1] = {X.shape[1]} and weight tensor with shape[0] = {W.shape[0]}"

    _batch_norm_forward_kernel[(num_features,)](
        Y,
        Y.stride(0),
        X_T,
        X_T.stride(0),
        W,
        B,
        Mean,
        RSTD,
        RM,
        RV,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        momentum=momentum,
    )
    return Y.T.view(*shape), X, Mean, RSTD, RM, RV


def batch_norm_backward(dY, X, W, B, Mean, RSTD):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    _DW = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)
    _DB = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)
    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16
    _batch_norm_backward_kernel[grid](
        X,
        W,
        Mean,
        RSTD,
        DX,
        _DW,
        _DB,
        dY,
        X.stride(0),
        DX.stride(0),
        _DW.stride(0),
        _DB.stride(0),
        dY.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
    )

    DW = _DW.sum(dim=0).to(W.dtype)
    DB = _DB.sum(dim=0).to(W.dtype)

    DX = DX.view(*shape)
    return DX, DW, DB


class LigerBatchNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, running_mean, running_var, eps, momentum):
        Y, X, mean, rstd, running_mean, running_var = batch_norm_forward(X, W, B, running_mean, running_var, eps, momentum)
        ctx.save_for_backward(X, W, B, mean, rstd, running_mean, running_var)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD, RM, RV = ctx.saved_tensors
        DX, DW, DB = batch_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None, None, None, None
