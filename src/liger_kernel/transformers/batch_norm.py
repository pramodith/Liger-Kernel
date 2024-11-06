import torch
import torch.nn as nn

from liger_kernel.ops.batch_norm import LigerBatchNormFunction

class LigerBatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float=1e-6, momentum: float=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = num_features
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(num_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_features)
        )
        self.variance_epsilon = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features, requires_grad=False)
        self.running_var = torch.ones(num_features, requires_grad=False)

    def forward(self, hidden_states):
        output, running_mean, running_var = LigerBatchNormFunction.apply(
            hidden_states, self.weight, self.bias,  self.running_mean, self.running_var, self.variance_epsilon, self.momentum
        )
        self.running_mean = running_mean
        self.running_var = running_var
        return output

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"