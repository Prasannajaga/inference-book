# GEMV CUDA Benchmark: What We Tried, What Worked, and What Did Not

This note turns the experiment in [`kernels/gemv_benchmark.ipynb`](kernels/gemv_benchmark.ipynb) into a readable benchmark report.

The short version: we built a small CUDA extension for GEMV, compared it against PyTorch and cuBLAS, and learned the useful lesson that a simple cooperative row-wise CUDA kernel can beat a naive kernel by a lot, but it does not meaningfully beat cuBLAS/PyTorch on this shape. The benchmark is useful as a CUDA learning exercise and as a baseline-building tool, but it is not evidence of an end-to-end LLM serving improvement.

## What problem are we benchmarking?

The operation is GEMV:

```text
y = W @ x
```

with:

```text
W: [4096, 4096]
x: [4096]
y: [4096]
```

This is a decode-like operation: one vector multiplied by a large matrix. Each weight is read once, `x` is reused, and the output is small. That makes the benchmark heavily influenced by memory bandwidth and reduction overhead.

The notebook runs two separate precision cases:

| Precision | W dtype | x dtype | y dtype | Logical bytes per GEMV |
|---|---:|---:|---:|---:|
| FP16 | `torch.float16` | `torch.float16` | `torch.float16` | 33.571 MB |
| FP32 | `torch.float32` | `torch.float32` | `torch.float32` | 67.142 MB |

This is important. Earlier, the experiment used FP16 inputs with FP32 output. That was useful for correctness checking, but it was not a clean “FP16 versus FP32” benchmark. The current benchmark uses same-dtype input and output for each precision mode.

Internally, the custom CUDA kernels still accumulate in `float`. For FP16, that means:

```text
load fp16 -> convert to fp32 -> accumulate in fp32 -> cast output to fp16
```

So this is not “pure fp16 arithmetic all the way down.” It is a normal inference-style FP16 memory format with FP32 accumulation.

## Environment

Notebook output:

```text
GPU name:           NVIDIA GeForce RTX 4050 Laptop GPU
CUDA version:       13.0
PyTorch version:    2.12.1+cu130
Compute capability: 8.9
Total GPU memory:   5.64 GiB
TF32 matmul:        False
```

TF32 was disabled so the FP32 path is closer to real FP32 instead of silently using TF32 tensor cores through PyTorch.

The notebook compiles the native extension through `torch.utils.cpp_extension.load_inline`. This affects build/startup time, but not the measured kernel timings. The benchmark warms up each function and times only CUDA work using CUDA events.

## Implementations compared

The notebook compares four methods.

### 1. PyTorch `torch.matmul`

This is the regular PyTorch baseline:

```python
torch.matmul(W, x, out=y_torch)
```

For this operation, PyTorch will eventually dispatch to a vendor-tuned backend. This is the baseline we actually care about beating.

### 2. Naive CUDA

The naive kernel assigns one CUDA thread to one output row:

```text
thread i computes y[i]
```

Each thread loops over all 4096 columns serially:

```text
for col in 0..4095:
    acc += W[row, col] * x[col]
```

This is simple and correct, but it leaves a lot of GPU parallelism unused inside each row. The GPU has many threads, but each individual row reduction is still a long scalar loop inside one thread.

### 3. Optimized CUDA

The optimized kernel assigns one CUDA block to one output row:

```text
block i computes y[i]
```

Inside each block:

- 256 threads cooperate on the row.
- Threads walk through columns in a strided pattern.
- FP16 uses `half2` vectorized loads when alignment allows.
- FP32 uses `float2` vectorized loads when alignment allows.
- Each thread accumulates a partial sum in FP32.
- Warp shuffle reductions combine the partial sums.
- A small shared-memory handoff combines per-warp sums.

This is the main custom kernel experiment. It directly fixes the biggest flaw in the naive implementation: one thread no longer owns the whole dot product.

### 4. Direct cuBLAS

The extension also calls `cublasGemmEx` directly.

Even though this is GEMV, the code expresses it as a one-column GEMM:

```text
[M, N] @ [N, 1] -> [M, 1]
```

This makes cuBLAS a strong vendor baseline. The code also avoids copying/transposing `W` by using the usual row-major/column-major interpretation trick: PyTorch stores `W` row-major, while cuBLAS reads column-major, so the wrapper transposes cuBLAS's interpretation.

## Correctness

The notebook validates each method against a PyTorch FP32 reference, then casts that reference to the benchmark dtype.

Tolerance:

| Precision | rtol | atol |
|---|---:|---:|
| FP16 | `1e-2` | `1e-1` |
| FP32 | `1e-4` | `1e-3` |

Correctness output:

| Precision | Method | Max abs error | Mean abs error | Max rel error |
|---|---|---:|---:|---:|
| FP16 | PyTorch `torch.matmul` | 0.0000 | 0.0000 | 0.0000 |
| FP16 | Naive CUDA | 0.0625 | 0.000074 | 0.2885 |
| FP16 | Optimized CUDA | 0.03125 | 0.000019 | 0.0989 |
| FP16 | cuBLAS direct | 0.0000 | 0.0000 | 0.0000 |
| FP32 | PyTorch `torch.matmul` | 0.0000 | 0.0000 | 0.0000 |
| FP32 | Naive CUDA | 0.000626 | 0.000054 | 0.0021 |
| FP32 | Optimized CUDA | 0.000076 | 0.000016 | 0.0006 |
| FP32 | cuBLAS direct | 0.0000 | 0.0000 | 0.0000 |

The custom kernels are not bit-identical to the reference. That is expected because reduction order differs. Floating-point summation is not associative, so changing which partial sums are added first changes the final rounded value.

The optimized kernel is more accurate than the naive kernel in this run:

- FP16 max absolute error improves from `0.0625` to `0.03125`.
- FP32 max absolute error improves from `6.2561e-4` to `7.6294e-5`.

One caveat: max relative error can look large when the expected value is close to zero. The FP16 naive path has `0.2885` max relative error, but its mean absolute error is only `7.4179e-05`, and it passed the configured FP16 tolerance.

## Timing results

Benchmark settings:

```text
warmup iterations: 50
timed iterations:  500
timing method:     CUDA events
```

Raw timing output:

| Precision | Method | Mean latency | Min | Median | p95 |
|---|---|---:|---:|---:|---:|
| FP16 | PyTorch `torch.matmul` | 205.793 µs | 183.296 µs | 209.920 µs | 210.944 µs |
| FP16 | Naive CUDA | 639.586 µs | 618.496 µs | 623.616 µs | 680.960 µs |
| FP16 | Optimized CUDA | 185.232 µs | 183.296 µs | 185.344 µs | 186.368 µs |
| FP16 | cuBLAS direct | 185.148 µs | 183.296 µs | 185.344 µs | 186.368 µs |
| FP32 | PyTorch `torch.matmul` | 362.854 µs | 361.472 µs | 362.496 µs | 363.520 µs |
| FP32 | Naive CUDA | 712.241 µs | 669.696 µs | 715.776 µs | 730.112 µs |
| FP32 | Optimized CUDA | 362.195 µs | 361.472 µs | 362.496 µs | 363.520 µs |
| FP32 | cuBLAS direct | 362.868 µs | 361.472 µs | 362.496 µs | 363.520 µs |

The same data expressed as logical bandwidth:

| Precision | Method | Mean latency | Logical bandwidth | Speedup vs naive |
|---|---|---:|---:|---:|
| FP16 | PyTorch `torch.matmul` | 205.793 µs | 163.13 GB/s | 3.11x |
| FP16 | Naive CUDA | 639.586 µs | 52.49 GB/s | 1.00x |
| FP16 | Optimized CUDA | 185.232 µs | 181.24 GB/s | 3.45x |
| FP16 | cuBLAS direct | 185.148 µs | 181.32 GB/s | 3.45x |
| FP32 | PyTorch `torch.matmul` | 362.854 µs | 185.04 GB/s | 1.96x |
| FP32 | Naive CUDA | 712.241 µs | 94.27 GB/s | 1.00x |
| FP32 | Optimized CUDA | 362.195 µs | 185.37 GB/s | 1.97x |
| FP32 | cuBLAS direct | 362.868 µs | 185.03 GB/s | 1.96x |

The notebook also computes “percent of peak bandwidth” using:

```python
PEAK_BANDWIDTH_GBPS = 1008.0
```

That value is a placeholder from a different GPU class, not the RTX 4050 Laptop GPU printed by the notebook. So the reported `~18% of peak` should not be interpreted literally. The useful metric here is the logical bandwidth comparison between methods in the same run.

## What worked

### The optimized CUDA design fixed the naive kernel's core problem

The naive kernel is bad because each row is handled serially by one thread. For a 4096-wide dot product, that means one thread performs 4096 multiply-add operations before writing a single output.

The optimized kernel distributes each row across 256 threads. That gives the GPU much more parallel work per row, and the result is obvious:

| Precision | Naive CUDA | Optimized CUDA | Speedup |
|---|---:|---:|---:|
| FP16 | 639.586 µs | 185.232 µs | 3.45x |
| FP32 | 712.241 µs | 362.195 µs | 1.97x |

That is the strongest result in the notebook. The cooperative-row design is clearly better than the one-thread-per-row design.

### FP16 benefits more from the custom optimized path

In FP16, optimized CUDA is faster than PyTorch:

```text
PyTorch:        205.793 µs
Optimized CUDA: 185.232 µs
Speedup:        1.11x
```

This is a real improvement over this PyTorch call in the notebook. The likely reason is that this shape is a narrow one-column GEMM/GEMV case, and the custom kernel's simple row-wise strategy has low overhead.

But the important comparison is cuBLAS:

```text
Optimized CUDA: 185.232 µs
cuBLAS direct:  185.148 µs
```

Those are essentially the same. So the custom kernel did not beat the vendor baseline.

### FP32 converged almost exactly to PyTorch/cuBLAS

For FP32:

```text
PyTorch:        362.854 µs
Optimized CUDA: 362.195 µs
cuBLAS direct:  362.868 µs
```

This is basically a tie. The custom kernel is only about `1.002x` faster than PyTorch in this run, which is too small to treat as a meaningful win.

## What did not work

### The naive kernel did not work as a performance strategy

The naive kernel is useful as a teaching baseline, but not as an optimization.

It loses badly:

- FP16 naive is `3.45x` slower than optimized CUDA.
- FP32 naive is `1.97x` slower than optimized CUDA.

The reason is structural, not incidental. One thread per row does not expose enough parallelism inside each dot product. It also creates long per-thread dependency chains in the accumulator.

### The custom optimized kernel did not beat cuBLAS

This is the most important negative result.

The optimized kernel caught up to cuBLAS, but it did not outperform it:

| Precision | Optimized CUDA | cuBLAS direct | Difference |
|---|---:|---:|---:|
| FP16 | 185.232 µs | 185.148 µs | cuBLAS slightly faster |
| FP32 | 362.195 µs | 362.868 µs | optimized slightly faster |

Both differences are tiny. On a benchmark like this, they are not enough to claim a durable kernel win.

This means the optimized kernel is educationally successful but not practically compelling yet. If we already have cuBLAS available, replacing it with this kernel would add maintenance burden without a clear performance payoff.

### The “percent peak bandwidth” metric did not work as configured

The notebook uses:

```text
PEAK_BANDWIDTH_GBPS = 1008.0
```

But the benchmark ran on:

```text
NVIDIA GeForce RTX 4050 Laptop GPU
```

So the percent-of-peak plot is using the wrong peak-bandwidth denominator. The logical bandwidth values are still useful:

```text
FP16 optimized: ~181 GB/s
FP32 optimized: ~185 GB/s
```

But the plotted percentage of peak should be treated as a placeholder until `PEAK_BANDWIDTH_GBPS` is changed to match the actual GPU.

### This does not prove an LLM serving improvement

This notebook is a standalone microbenchmark. It does not measure:

- TTFT
- inter-token latency
- requests/sec
- tokens/sec
- p50/p95/p99 latency
- max stable concurrency
- GPU memory pressure inside a real serving engine

So this experiment cannot justify integrating a custom GEMV kernel into an LLM serving path.

That matters especially for this repository's broader goal: custom kernels should be kept only if they improve end-to-end serving performance for a specific model and serving engine. A GEMV microbenchmark win, even if real, is not enough.

## Why the optimized kernel landed near cuBLAS

For this shape, GEMV is dominated by reading the weight matrix:

```text
FP16 W read: 4096 × 4096 × 2 bytes ≈ 33.55 MB
FP32 W read: 4096 × 4096 × 4 bytes ≈ 67.11 MB
```

The vector `x` and output `y` are tiny by comparison. Once the kernel has:

- coalesced global loads,
- enough threads per row,
- vectorized loads where possible,
- low-overhead reductions,

there is not a huge amount of headroom left. That is why the optimized kernel jumps far ahead of the naive version but then lands right beside cuBLAS.

cuBLAS is already very good at this kind of operation. Matching it is nice. Beating it reliably is harder.

## What I would change next

If this notebook were going to become a stronger benchmark, I would change four things.

### 1. Sweep shapes

One shape is not enough. We should test decode-like and prefill-like cases:

```text
[1, hidden_dim]
[4, hidden_dim]
[8, hidden_dim]
[16, hidden_dim]
[32, hidden_dim]
[128, hidden_dim]
[512, hidden_dim]
[2048, hidden_dim]
[8192, hidden_dim]
```

The custom kernel might help small batches and lose for larger batches, or vice versa.

### 2. Use the actual GPU bandwidth

Replace:

```python
PEAK_BANDWIDTH_GBPS = 1008.0
```

with the correct sustained or specified bandwidth for the measured GPU. Otherwise the percent-of-peak plot is mostly decorative.

### 3. Add variance reporting

The notebook prints mean/min/median/p95, which is good. For a blog-quality benchmark, I would also report:

- standard deviation,
- multiple independent benchmark runs,
- maybe a confidence interval.

That would help decide whether the tiny optimized-vs-cuBLAS differences are noise.

### 4. Keep this separate from model-serving claims

This experiment should remain a standalone kernel study unless profiling shows GEMV is actually hot in a specific model-serving path. For the main project, RMSNorm/SwiGLU/RoPE are better first targets than custom GEMV.

## Update: larger shapes and new variants

After the first 4096×4096 benchmark, we added three more optimized variants:

| Variant | Idea |
|---|---|
| Optimized CUDA | one 256-thread block computes one row |
| Optimized CUDA v1 | one warp computes one row; no shared-memory row reduction |
| Optimized CUDA v2 | one warp per row, with tiled shared-vector staging |
| Optimized CUDA v3 | FP16-only full-vector shared staging when the vector fits; FP32 falls back to v1 |

The larger benchmark used 10K×10K and 20K×20K GEMV. The naive kernel was skipped for these large shapes because it is only a teaching baseline.

Latest large-shape results on the RTX 4050 Laptop GPU:

| Shape | Precision | Best custom method | Best custom latency | cuBLAS latency | Speedup vs cuBLAS |
|---|---|---|---:|---:|---:|
| 10K×10K | FP16 | Optimized CUDA v1 | 1071.187 µs | 1166.442 µs | 1.089× |
| 10K×10K | FP32 | Optimized CUDA v1 | 2136.961 µs | 2201.847 µs | 1.030× |
| 20K×20K | FP16 | Optimized CUDA v3 | 4293.525 µs | 5003.992 µs | 1.166× |
| 20K×20K | FP32 | Optimized CUDA | 8524.591 µs | 8547.413 µs | 1.003× |

The important new result is FP16 at 20K×20K. v3 finally gives a meaningful speedup over direct cuBLAS in this microbenchmark. The reason is simple: the FP16 vector is small enough to fit in default shared memory at this size:

```text
20,000 fp16 values × 2 bytes = 40 KB
```

So v3 loads the full vector once per block, then reuses it across the block's row warps. This avoids v2's repeated tile synchronizations while still reducing repeated vector traffic. The weight matrix still dominates traffic, but cutting the vector traffic and keeping the reduction simple is enough to beat cuBLAS for this specific GEMV shape.

v2 did not win. Its idea was reasonable — stage vector tiles in shared memory — but the repeated `__syncthreads()` overhead outweighed the reuse benefit.

FP32 is less exciting. The current optimized kernel slightly beats cuBLAS at 20K×20K, but only by about `1.003×`, which is too small to treat as a robust win without repeated runs.

## Final takeaway

The experiment succeeded as a learning benchmark and produced one real large-shape microbenchmark win:

- The naive CUDA kernel showed what not to do.
- The optimized CUDA kernel demonstrated cooperative row-level reduction.
- Separate FP16 and FP32 runs made the benchmark contract cleaner.
- v1 showed that one-warp-per-row is a strong GEMV strategy.
- v3 beat direct cuBLAS by about `1.166×` on 20K×20K FP16.

But it still does not prove an LLM serving optimization:

- the win is a standalone GEMV microbenchmark,
- the strongest win is shape- and dtype-specific,
- No end-to-end LLM serving metric was measured.

So the honest conclusion is:

```text
KEEP as a standalone CUDA learning benchmark.
KEEP v1/v3 as benchmark variants worth studying.
DO NOT integrate this as a custom model kernel without profiling evidence.
```

That is not a failure. It is exactly what a good benchmark is supposed to tell us before we spend more time on a kernel.
