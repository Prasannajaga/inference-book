# %% [markdown]
# ### Understanding basic of kernels
# 
# 
# we will explore and compare why we need GPUS and what scale we need CPUS and breakdown on kernels optimization as well here 
# 
# this blogs is going to be fun how each matrix elements compute with CPU , GPU , GPU + kernels
# 
# as we speak GPU process at parallely which is very fast and CPU does it in sequentially 
# 
# optimized kernel can improve latency on computing complex matrix multiplication here 

# %%
import gc
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl

architecture_names = {
    (7, 5): "Turing",
    (8, 0): "Ampere (A100-class)",
    (8, 6): "Ampere (consumer-class)",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (12, 0): "Blackwell",
}

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not visible. In VS Code, choose the project's .venv Python kernel, "
        "then verify that `uv run python -c \"import torch; print(torch.cuda.is_available())\"` prints True."
    )

DEVICE_INDEX = torch.cuda.current_device()
DEVICE = torch.device(f"cuda:{DEVICE_INDEX}")
props = torch.cuda.get_device_properties(DEVICE_INDEX)
torch.manual_seed(0)

print(f"PyTorch:             {torch.__version__}")
print(f"PyTorch CUDA build:  {torch.version.cuda}")
print(f"Triton:              {triton.__version__}")
print(f"Device:              {props.name}")
print(f"Compute capability:  {props.major}.{props.minor} (sm_{props.major}{props.minor})")
print(f"CUDA architectures compiled into PyTorch: {torch.cuda.get_arch_list()}")

# %% [markdown]
# as you have noticed here that Floast operation has differnet in sizes the more accurate we need 
# the more memory we need to store on HBM, and matrix multiplication compute gets heavy you easiy get 
# CUDA oom 

# %%
@triton.jit
def matrix_matmul_kernel(
    a,
    b,
    c,
    m,
    n,
    k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    programs_n = tl.cdiv(n, BLOCK_N)
    program_m = program_id // programs_n
    program_n = program_id % programs_n

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    a_offsets = offsets_m[:, None] * k + offsets_k[None, :]
    b_offsets = offsets_k[:, None] * n + offsets_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a_values = tl.load(
            a + a_offsets,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k),
            other=0.0,
        )
        b_values = tl.load(
            b + b_offsets,
            mask=(offsets_k[:, None] < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a_values, b_values, input_precision="ieee")
        a_offsets += BLOCK_K
        b_offsets += BLOCK_K * n
        offsets_k += BLOCK_K

    c_offsets = offsets_m[:, None] * n + offsets_n[None, :]
    c_mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
    tl.store(c + c_offsets, accumulator, mask=c_mask)


def launch_matrix_matmul(
    a,
    b,
    c,
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
):
    if not (a.is_cuda and b.is_cuda and c.is_cuda):
        raise ValueError("The Triton path requires CUDA tensors.")
    if not (a.ndim == b.ndim == c.ndim == 2):
        raise ValueError("A, B, and C must be two-dimensional matrices.")
    if a.shape[1] != b.shape[0] or c.shape != (a.shape[0], b.shape[1]):
        raise ValueError("Expected A[M, K], B[K, N], and C[M, N].")
    if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
        raise ValueError("This kernel expects contiguous row-major matrices.")

    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    matrix_matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


MATRIX_ROWS = 1096
MATRIX_INNER = 1096
MATRIX_COLS = 1096
MATRIX_BLOCK_M = 64
MATRIX_BLOCK_N = 64
MATRIX_BLOCK_K = 32
MATRIX_NUM_WARPS = 4
INPUT_SCALE = 0.1
CPU_WARMUP_RUNS = 1
GPU_WARMUP_RUNS = 2

DTYPE_CONFIGS = [
    {
        "label": "FP8 (E4M3)",
        "dtype": getattr(torch, "float8_e4m3fn", None),
        "bits": 8,
        "rtol": 0.25,
        "atol": 0.25,
    },
    {"label": "FP16", "dtype": torch.float16, "bits": 16, "rtol": 2e-2, "atol": 2e-2},
    {"label": "FP32", "dtype": torch.float32, "bits": 32, "rtol": 1e-4, "atol": 1e-4},
]

CPU_PATH = "CPU PyTorch (resident)"
TORCH_GPU_PATH = "PyTorch GPU (resident)"
TRITON_GPU_PATH = "Triton GPU (resident)"
TRITON_ROUND_TRIP_PATH = "Triton GPU + round trip"
BENCHMARK_PATHS = [CPU_PATH, TORCH_GPU_PATH, TRITON_GPU_PATH, TRITON_ROUND_TRIP_PATH]

print("Expression: C = A @ B")
print(f"A shape:    [{MATRIX_ROWS:,}, {MATRIX_INNER:,}]")
print(f"B shape:    [{MATRIX_INNER:,}, {MATRIX_COLS:,}]")
print(f"C shape:    [{MATRIX_ROWS:,}, {MATRIX_COLS:,}]")
for config in DTYPE_CONFIGS:
    matrix_mib = MATRIX_ROWS * MATRIX_COLS * config["bits"] / 8 / 2**20
    print(f"One {config['label']:<10} output matrix: {matrix_mib:7.1f} MiB")
print(f"CPU worker threads available to PyTorch: {torch.get_num_threads()}")



# %% [markdown]
# we will benchmark matrix multiplication across CPU, PyTorch GPU, Triton GPU, and the full CPU↔GPU round trip

# %%
def _wall_latency_ms(fn):
    start = time.perf_counter()
    fn()
    return float((time.perf_counter() - start) * 1e3)


def _cuda_latency_ms(fn):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    fn()
    end_event.record()
    end_event.synchronize()
    return float(start_event.elapsed_time(end_event))


def _empty_measurement(status, error=None):
    return {"latency_ms": np.nan, "status": status, "error": error}


def benchmark_matmul_dtype(
    source_a,
    source_b,
    dtype_config,
    *,
    block_m=MATRIX_BLOCK_M,
    block_n=MATRIX_BLOCK_N,
    block_k=MATRIX_BLOCK_K,
    num_warps=MATRIX_NUM_WARPS,
    cpu_warmup_runs=CPU_WARMUP_RUNS,
    gpu_warmup_runs=GPU_WARMUP_RUNS,
):
    """Benchmark one matmul dtype and return one row per execution path."""
    label = dtype_config["label"]
    dtype = dtype_config["dtype"]
    bits = dtype_config["bits"]
    m, k = source_a.shape
    _, n = source_b.shape
    output_numel = m * n
    output_mib = output_numel * bits / 8 / 2**20
    measurements = {}
    correctness = {
        "correct": False,
        "max_abs_error": np.nan,
        "mean_abs_error": np.nan,
        "peak_gpu_memory_mib": np.nan,
    }

    if dtype is None:
        message = f"{label} is not available in this PyTorch build"
        measurements = {
            path: _empty_measurement("unavailable", message)
            for path in BENCHMARK_PATHS
        }
    else:
        a_cpu = source_a.to(dtype)
        b_cpu = source_b.to(dtype)

        # CPU PyTorch matmul latency.
        c_cpu = None
        try:
            c_cpu = torch.empty((m, n), dtype=dtype)
            for _ in range(cpu_warmup_runs):
                torch.matmul(a_cpu, b_cpu, out=c_cpu)
            measurements[CPU_PATH] = {
                "latency_ms": _wall_latency_ms(
                    lambda: torch.matmul(a_cpu, b_cpu, out=c_cpu)
                ),
                "status": "ok",
                "error": None,
            }
        except Exception as exc:
            measurements[CPU_PATH] = _empty_measurement("unsupported", str(exc))
        finally:
            del c_cpu

        a_gpu = b_gpu = c_gpu = c_from_gpu = None
        try:
            torch.cuda.reset_peak_memory_stats(DEVICE)
            a_gpu = a_cpu.to(DEVICE)
            b_gpu = b_cpu.to(DEVICE)
            c_gpu = torch.empty((m, n), dtype=dtype, device=DEVICE)
            c_from_gpu = torch.empty((m, n), dtype=dtype)

            # PyTorch GPU matmul latency with resident tensors.
            try:
                for _ in range(gpu_warmup_runs):
                    torch.matmul(a_gpu, b_gpu, out=c_gpu)
                measurements[TORCH_GPU_PATH] = {
                    "latency_ms": _cuda_latency_ms(
                        lambda: torch.matmul(a_gpu, b_gpu, out=c_gpu)
                    ),
                    "status": "ok",
                    "error": None,
                }
            except Exception as exc:
                measurements[TORCH_GPU_PATH] = _empty_measurement(
                    "unsupported", str(exc)
                )

            # Triton GPU matmul latency with resident tensors.
            try:
                launch = lambda: launch_matrix_matmul(
                    a_gpu,
                    b_gpu,
                    c_gpu,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                )
                launch()  # Compile the dtype specialization outside the timing.
                for _ in range(gpu_warmup_runs):
                    launch()
                measurements[TRITON_GPU_PATH] = {
                    "latency_ms": _cuda_latency_ms(launch),
                    "status": "ok",
                    "error": None,
                }

                # Full CPU → GPU → Triton matmul → CPU latency.
                def triton_round_trip():
                    a_gpu.copy_(a_cpu)
                    b_gpu.copy_(b_cpu)
                    launch()
                    c_from_gpu.copy_(c_gpu)
                    torch.cuda.synchronize()

                measurements[TRITON_ROUND_TRIP_PATH] = {
                    "latency_ms": _wall_latency_ms(triton_round_trip),
                    "status": "ok",
                    "error": None,
                }

                # Validate Triton against FP32 PyTorch matmul on the same quantized inputs.
                reference_gpu = torch.matmul(a_gpu.float(), b_gpu.float())
                storage_reference = reference_gpu.to(dtype).float()
                actual = c_gpu.float()
                error = (actual - storage_reference).abs()
                torch.testing.assert_close(
                    actual,
                    storage_reference,
                    rtol=dtype_config["rtol"],
                    atol=dtype_config["atol"],
                )
                correctness = {
                    "correct": True,
                    "max_abs_error": float(error.max()),
                    "mean_abs_error": float(error.mean()),
                    "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(DEVICE) / 2**20,
                }
                del reference_gpu, storage_reference, actual, error
            except Exception as exc:
                measurements[TRITON_GPU_PATH] = _empty_measurement("failed", str(exc))
                measurements[TRITON_ROUND_TRIP_PATH] = _empty_measurement(
                    "failed", str(exc)
                )
        except Exception as exc:
            for path in (
                TORCH_GPU_PATH,
                TRITON_GPU_PATH,
                TRITON_ROUND_TRIP_PATH,
            ):
                measurements.setdefault(
                    path, _empty_measurement("failed", str(exc))
                )
        finally:
            del a_gpu, b_gpu, c_gpu, c_from_gpu
            del a_cpu, b_cpu
            gc.collect()
            torch.cuda.empty_cache()

    result_rows = []
    for path in BENCHMARK_PATHS:
        result_rows.append(
            {
                "dtype": label,
                "dtype_bits": bits,
                "path": path,
                "latency_ms": measurements[path]["latency_ms"],
                "status": measurements[path]["status"],
                "error": measurements[path]["error"],
                "rows": m,
                "inner_dim": k,
                "cols": n,
                "numel": output_numel,
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "matrix_mib": output_mib,
                **correctness,
            }
        )
    return result_rows


def run_matmul_benchmarks(
    rows=MATRIX_ROWS,
    inner_dim=MATRIX_INNER,
    cols=MATRIX_COLS,
    dtype_configs=DTYPE_CONFIGS,
    *,
    seed=0,
    input_scale=INPUT_SCALE,
    **benchmark_kwargs,
):
    """Run the same matrix multiplication for every configured dtype."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source_a = torch.randn(
        (rows, inner_dim), dtype=torch.float32, generator=generator
    ).mul_(input_scale)
    source_b = torch.randn(
        (inner_dim, cols), dtype=torch.float32, generator=generator
    ).mul_(input_scale)

    result_rows = []
    try:
        for dtype_config in dtype_configs:
            print(f"Benchmarking matmul {dtype_config['label']} ...")
            result_rows.extend(
                benchmark_matmul_dtype(
                    source_a,
                    source_b,
                    dtype_config,
                    **benchmark_kwargs,
                )
            )
    finally:
        del source_a, source_b
        gc.collect()

    return pd.DataFrame(result_rows)



# %%
benchmark_results = run_matmul_benchmarks()


# %%
display(
    benchmark_results[
        ["dtype", "path", "latency_ms", "status"]
    ].round({"latency_ms": 3})
)



# %%
DTYPE_ORDER = ["FP8 (E4M3)", "FP16", "FP32"]

PLOT_PATHS = [
    CPU_PATH,
    TORCH_GPU_PATH,
    TRITON_GPU_PATH,
    TRITON_ROUND_TRIP_PATH,
]

PATH_LABELS = {
    CPU_PATH: "CPU PyTorch",
    TORCH_GPU_PATH: "PyTorch GPU",
    TRITON_GPU_PATH: "Triton GPU",
    TRITON_ROUND_TRIP_PATH: "Triton + transfers",
}

PATH_COLORS = {
    CPU_PATH: "#6B7280",
    TORCH_GPU_PATH: "#4C78A8",
    TRITON_GPU_PATH: "#F58518",
    TRITON_ROUND_TRIP_PATH: "#B279A2",
}


def plot_matmul_latency_comparison(results, *, log_scale=True):
    """Plot one large latency comparison without modifying results."""
    dtype_order = [
        dtype
        for dtype in DTYPE_ORDER
        if dtype in set(results["dtype"].dropna())
    ]
    x = np.arange(len(dtype_order), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(18, 7))
    width = 0.82 / len(PLOT_PATHS)
    plotted_latencies = []

    for path_index, path in enumerate(PLOT_PATHS):
        path_rows = (
            results.loc[results["path"].eq(path)]
            .drop_duplicates("dtype", keep="last")
            .set_index("dtype")
        )
        latencies = np.full(len(dtype_order), np.nan, dtype=np.float64)

        for dtype_index, dtype in enumerate(dtype_order):
            if dtype not in path_rows.index:
                continue
            row = path_rows.loc[dtype]
            latency = float(row["latency_ms"])
            if str(row["status"]).lower() == "ok" and np.isfinite(latency):
                latencies[dtype_index] = latency

        valid = np.isfinite(latencies)
        if not valid.any():
            continue

        positions = x + (path_index - (len(PLOT_PATHS) - 1) / 2) * width
        bars = ax.bar(
            positions[valid],
            latencies[valid],
            width=width * 0.92,
            color=PATH_COLORS[path],
            label=PATH_LABELS[path],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(
            bars,
            labels=[f"{latency:.3f} ms" for latency in latencies[valid]],
            padding=5,
            fontsize=10,
            rotation=90,
        )
        plotted_latencies.extend(latencies[valid])

    if log_scale:
        ax.set_yscale("log")

    if plotted_latencies:
        positive = np.asarray(plotted_latencies, dtype=np.float64)
        positive = positive[positive > 0]
        if positive.size:
            if log_scale:
                ax.set_ylim(positive.min() * 0.60, positive.max() * 3.0)
            else:
                ax.set_ylim(0, positive.max() * 1.28)

    ax.set_title(
        "Matmul latency comparison — lower is better ↓",
        fontsize=18,
        fontweight="bold",
        pad=34,
    )
    ax.text(
        0.5,
        1.015,
        "C = A @ B | latency in milliseconds",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#444444",
    )
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_xlabel("Data type", fontsize=12)
    ax.set_xticks(x, dtype_order, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        title="Execution path",
        frameon=False,
        fontsize=11,
        title_fontsize=11,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
    )

    if "props" in globals() and not results.empty:
        config = results.iloc[0]
        fig.text(
            0.5,
            0.018,
            (
                f"GPU: {props.name} | sm_{props.major}{props.minor} | "
                f"A: {int(config.rows):,} × {int(config.inner_dim):,} | "
                f"B: {int(config.inner_dim):,} × {int(config.cols):,} | "
                f"block: {int(config.block_m)} × {int(config.block_n)} × {int(config.block_k)} | "
                f"warps: {int(config.num_warps)}"
            ),
            ha="center",
            fontsize=9,
            color="#555555",
        )

    plt.tight_layout(rect=(0.03, 0.12, 0.99, 0.94))
    return fig, ax



# %%
benchmark_figure, benchmark_axis = plot_matmul_latency_comparison(benchmark_results)
plt.show()

