import torch
import triton
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.transformers.batch_norm import LigerBatchNorm


def bench_speed_batch_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    num_features = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    batch_size = extra_benchmark_config["M"]
    seq_len = extra_benchmark_config["seq_length"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (batch_size, num_features)
    triton_bn = LigerBatchNorm(
        num_features=num_features, eps=eps
    ).to("cuda")
    torch_bn = torch.nn.BatchNorm1d(
        num_features=num_features, eps=eps
    ).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return triton_bn(x)
        if provider == "huggingface":
            return torch_bn(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd, quantiles=QUANTILES, grad_to_none=[x], rep=500
        )
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full, quantiles=QUANTILES, grad_to_none=[x], rep=500
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_batch_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    num_features = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    batch_size = extra_benchmark_config["M"]
    seq_len = extra_benchmark_config["seq_length"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (batch_size, num_features)
    triton_bn = LigerBatchNorm(
        num_features=num_features, eps=eps
    ).to("cuda")
    torch_bn = torch.nn.BatchNorm1d(
        num_features=num_features, eps=eps
    ).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return triton_bn(x)
        if provider == "huggingface":
            return torch_bn(x)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "batch_norm",
        "x_name": "C",
        "x_label": "num_features",
        "x_values": [2**i for i in range(15, 24)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "M": 128,
                "seq_length": 128,
                "dtype": torch.float32,
                "eps": 1e-6,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_batch_norm,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="time",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_batch_norm,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )