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
    Y_ptr,  # pointer to output, shape (num_features, batch_size*seq_len)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (num_features, batch_size*seq_len)
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
    momentum: tl.constexpr  # momentum for running mean and var
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
    
    X_ptr += X_row_stride * feature_idx
    Y_ptr += Y_row_stride * feature_idx

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
    running_mean = running_mean * (1-momentum) + mean * momentum
    # Need to use the bias corrected variance for running variance
    running_var = running_var * (1-momentum) + (var*n_cols/(n_cols-1)) * momentum

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
    X_ptr,  # pointer to input, shape (num_features, batch_size*seq_len)
    W_ptr,  # pointer to weights, shape (num_features,)
    Mean_ptr,  # pointer to mean, shape (num_features,)
    RSTD_ptr,  # pointer to rstd, shape (num_features,)
    DX_ptr,  # pointer to input grad, shape (num_features, batch_size*seq_len)
    DW_ptr,  # pointer to weights grad, shape (num_features,)
    DB_ptr,  # pointer to bias grad, shape (num_features,)
    Upstream_ptr,  # pointer to output grad, shape (num_features, batch_size*seq_len)
    stride_x,  # stride of each row in input
    n_cols,
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
    feature_idx = tl.program_id(0)

    # DY, DX and X are the same shape so will have the same stride
    X_ptr += stride_x * feature_idx
    Upstream_ptr += stride_x * feature_idx
    DX_ptr += stride_x * feature_idx
    offsets = tl.arange(0, BLOCK_SIZE)

    W = tl.load(W_ptr + feature_idx)
    mean = tl.load(Mean_ptr + feature_idx)
    rstd = tl.load(RSTD_ptr + feature_idx)

    c1 = 0.0
    c2 = 0.0

    dw = 0.0
    db = 0.0 

    # Compute the sum terms for the gradients
    for i in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = offsets + i
        mask = col_offsets < n_cols
        X = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        Upstream_grad = tl.load(Upstream_ptr + col_offsets, mask=mask, other=0.0)

        # Equations for the gradients are the same as layer norm, in fact batch norm can be thought of as a transpose of layer norm
        x_hat = (X - mean) * rstd
        wdy = W * Upstream_grad
        c1 += tl.sum(x_hat * wdy)
        c2 += tl.sum(wdy)
        
        dw += tl.sum(x_hat * Upstream_grad)
        db += tl.sum(Upstream_grad)
    
    c1 = c1/n_cols
    c2 = c2/n_cols

    # Compute the gradients for the input
    for i in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = offsets + i
        mask = col_offsets < n_cols
        X = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        Upstream_grad = tl.load(Upstream_ptr + col_offsets, mask=mask, other=0.0)
        x_hat = (X - mean) * rstd
        dx = (W * Upstream_grad - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + col_offsets, dx.to(dtype), mask=mask)
    
    tl.store(DW_ptr + feature_idx, dw)
    tl.store(DB_ptr + feature_idx, db)
    


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

    assert 2 <= len(shape) <= 3, f"Wrong input shape : {shape}."
    
    if len(shape) == 2:
        batch_size, num_features = shape
        seq_length = 1
    elif len(shape) == 3:
        batch_size, num_features, seq_length = shape
        X = X.permute(0, 2, 1).contiguous()
    
    X = X.view(-1, num_features)
    # We need to compute the mean and variance across elements in the same feature dimension across all samples in the batch
    X = X.T.contiguous()
    
    n_cols = batch_size * seq_length
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    Y = torch.empty((num_features, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(num_features, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(num_features, dtype=X.dtype, device=X.device)
    # Move running mean and variance to gpu
    RM = RM.to(X.device)
    RV = RV.to(X.device)

    _batch_norm_forward_kernel[(num_features,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
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
    Y = Y.view(num_features, batch_size, seq_length).permute(1, 0, 2).view(*shape)
    return Y, Mean, RSTD, RM, RV


def batch_norm_backward(dY, X, W, B, Mean, RSTD):
    shape = dY.shape
    
    if len(shape) == 2:
        batch_size, num_features = shape
        seq_length = 1
    elif len(shape) == 3:
        batch_size, num_features, seq_length = shape
        dY = dY.permute(0, 2, 1).contiguous()
        X = X.permute(0, 2, 1).contiguous()

    dY = dY.view(-1, num_features).T.contiguous()
    X = X.view(-1, num_features).T.contiguous()
    n_cols = batch_size * seq_length
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

    DX = torch.empty((num_features, n_cols), dtype=X.dtype, device=X.device)
    DW = torch.empty((num_features,), dtype=W.dtype, device=W.device)
    DB = torch.empty((num_features, ), dtype=W.dtype, device=W.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16
    _batch_norm_backward_kernel[(num_features, )](
        X,
        W,
        Mean,
        RSTD,
        DX,
        DW,
        DB,
        dY,
        X.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
    )
    DX = DX.view(num_features, batch_size, seq_length).permute(1, 0, 2).view(*shape)
    return DX, DW, DB


class LigerBatchNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, running_mean, running_var, eps, momentum):
        Y, mean, rstd, running_mean, running_var = batch_norm_forward(X, W, B, running_mean, running_var, eps, momentum)
        ctx.save_for_backward(X, W, B, mean, rstd)
        return Y, running_mean, running_var

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, *args):
        X, W, B, mean, rstd = ctx.saved_tensors
        DX, DW, DB = batch_norm_backward(dY, X, W, B, mean, rstd)
        return DX, DW, DB, None, None, None, None
