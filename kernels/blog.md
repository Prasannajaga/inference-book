# The Micro-Level Mechanics of Matrix Multiplication: CPU vs. GPU vs. Triton

*A deep dive into kernel basics, hardware scaling, and custom Triton kernel benchmarking.*

---

## 1. Introduction: Why Do We Need GPUs anyway?

When you start looking at how AI models work under the hood, you quickly realize everything boils down to massive matrix multiplications. In this blog, we will explore and compare why we need GPUs, at what scale we need CPUs, and do a breakdown on kernel optimizations as well.

This blog is going to be fun! We'll look at how each matrix element computes with CPU vs. GPU vs. GPU + custom Triton kernels. As we speak, GPU processes parallelly, which is very fast, whereas CPU does it sequentially. A CPU is like a high-performance sports car—it can execute a single thread of instructions incredibly fast, one after the other. A GPU is like a massive bus system—it moves thousands of passenger-threads at the same time, albeit at a slightly lower individual speed. For deep learning, where we perform millions of independent multiply-accumulate operations, the GPU’s parallel nature wins hands down.

Optimized custom kernels can improve the latency of computing these complex matrix multiplications. But at what scale do we really need them? The takeaway of this blog is to understand how CPU and GPU parallelism works, and how to make the right decision when choosing your hardware. For example, if you are running a very tiny model (like `<1B` parameters), you don't even need a GPU to run it until you are serving real production use cases and your traffic grows. At that point, hardware choices and custom kernels become a matter of survival.

---

## 2. Setting Up the Lab

Before we look at the numbers, let's establish the environment used for these benchmarks:

* **PyTorch Version:** `2.12.1+cu130`
* **Triton Version:** `3.7.1`
* **GPU Device:** `NVIDIA GeForce RTX 4050 Laptop GPU`
* **Compute Capability:** `8.9` (`sm_89` - Ada Lovelace Architecture)
* **CPU Worker Threads available to PyTorch:** `8`

### A Note on Precision and Memory

Float operations have different sizes—the more accurate we need them to be, the more memory we need to store them on HBM. Since this PyTorch version doesn't support native FP8, I will skip the FP8 comparison here. So we go with FP16 and FP32.

For a single output matrix at `2096 × 2096`:

* **FP16:** `8.4 MiB`
* **FP32:** `16.8 MiB`

---

## 3. Dissecting the Triton Matrix Multiplication Kernel

To understand how GPU computation works at the block level, we benchmark a basic block-based matrix multiplication kernel written in Triton.

```python
@triton.jit
def matrix_matmul_kernel(
    a, b, c, m, n, k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
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
```

### How this Kernel Works

* **Blocks:** The output matrix $C$ is split into `BLOCK_M × BLOCK_N` blocks. Each program instance computes one block.
* **Accumulator:** It loops over the inner dimension $K$ in `BLOCK_K` steps, loading sub-blocks of $A$ and $B$ to multiply (`tl.dot`) and accumulate.
* **Guards:** Boolean masks prevent out-of-bounds memory reads and writes.

---

## 4. The 1K Matrix Benchmark: CPU vs. GPU

Let's look at the results of multiplying two $1096 \times 1096$ matrices. The theoretical computational cost of this operation is:

* **Total Output Elements:** `1,201,216`
* **Multiply-Accumulate Terms:** `1,316,532,736`
* **Estimated FLOPs:** `2,633,065,472` (~2.63 GFLOPs)

Here are the measured latencies for the execution paths from our updated benchmark run:

## 5. What We Learned from the 1K Benchmark

Here is the breakdown of what these numbers actually mean in plain terms:

### why CPU FP16 < FP32?

CPU ran very slower as expected for small flops. The CPU is insanely slow at FP16—taking `927.953 ms` compared to just `10.145 ms` for FP32. That's a 90x slowdown! Standard consumer CPUs don't have native hardware support for FP16, so they have to emulate it in software. If you're running on CPU, stick to FP32.

### 2. Don't Forget the Copy Bottleneck

If your data isn't already on the GPU, you have to pay the PCIe copy tax to move it over and move the results back. For FP16, the actual Triton math takes `0.234 ms`, but adding the round-trip copy pushes it to `1.233 ms` (a 5x slowdown). If you are constantly moving data back and forth, you lose all GPU speed advantages.

**Why do we even do CPU-GPU round trips?**
For example, in LLM servers, the CPU handles network requests and tokenizes text, so we copy those tokens to the GPU. After the GPU finishes computing, we copy the resulting token ID back to the CPU to turn it into text.

**Why is this tradeoff okay?**
For a single tiny matrix multiplication, a round trip is terrible because the PCIe copy overhead is much slower than the math. But for a real LLM run, we copy the input tokens *once* to the GPU, run *hundreds* of layers and massive matrix operations entirely on the GPU, and then copy the single final token *back once*. Because the GPU does so much math per copy, the copy overhead becomes tiny, making the GPU speedup totally worth it.

### 3. Triton vs. cuBLAS (PyTorch GPU)

Our custom Triton kernel (`0.234 ms`) didn't beat PyTorch GPU (`0.231 ms`). This is because PyTorch uses cuBLAS, which is NVIDIA's closed-source, highly tuned math library that is extremely hard to beat for a simple single matrix multiplication.

The real power of Triton isn't in writing a single matmul, but in **kernel fusion**—combining matmul, activation (like SwiGLU), and normalization (like RMSNorm) in one go so we don't keep reading and writing intermediate tensors to the slow GPU memory.

---

## 6. The 2K Matrix Benchmark and Scaling Behavior

When we scale the matrix size to $2096 \times 2096$, the theoretical computational cost is:

* **Total Output Elements:** `4,393,216`
* **Multiply-Accumulate Terms:** `9,208,180,736`
* **Estimated FLOPs:** `18,416,361,472` (~18.42 GFLOPs)

Here are the measured latencies for the 2K matrix pass:

| Execution Path | Data Type | Latency (ms) | Achieved Throughput |
| :--- | :--- | :--- | :--- |
| **CPU PyTorch (resident)** | FP16 | 7369.317 ms | 2.50 GFLOP/s |
| **PyTorch GPU (resident)** | FP16 | 1.033 ms | 17.82 TFLOP/s |
| **Triton GPU (resident)** | FP16 | 1.035 ms | 17.79 TFLOP/s |
| **CPU PyTorch (resident)** | FP32 | 67.067 ms | 274.60 GFLOP/s |
| **PyTorch GPU (resident)** | FP32 | 3.836 ms | 4.80 TFLOP/s |
| **Triton GPU (resident)** | FP32 | 3.449 ms | 5.34 TFLOP/s |

### What We Learned from the 2K Benchmark

Here is why the 2K matrix achieved higher TFLOP/s and scaled so well:

* **GPU Saturation (Why TFLOPs went up):** A GPU has thousands of cores. With a small 1K matrix, the GPU is under-utilized because many cores sit idle. When we scale to 2K (which has $8\times$ the math), we finally fill up the GPU cores. It's like filling a passenger bus to capacity instead of running it nearly empty—the overall efficiency (TFLOP/s) goes way up.
* **CPU Collapse on FP16:** The CPU completely choked on FP16, taking `7.37 seconds` compared to just `67.07 ms` on FP32. Emulating FP16 is just too heavy for a CPU.
* **Triton vs PyTorch at 2K:** PyTorch and Triton are neck-and-neck on FP16 (~`1.03 ms`), but Triton actually wins on FP32 (`3.45 ms` vs `3.84 ms`). Custom kernels start showing their strength as data sizes scale up.
* **Scaling is King:** This scaling efficiency is why frontier models focus on compute scaling first. Simply scaling the compute makes the hardware run more efficiently, and scaling datasets makes the models smarter.

---

## 7. Conclusions

That's the end of our benchmark analysis! You should have known by now how GPUs make intelligence faster. Since the day the industry found that scaling parameters and datasets can get you to AGI, NVIDIA (Jensen) started deploying 100s of GB of memory in single chips and printing money.

The benchmark war you see between frontier model companies today is just scaling. Companies like Anthropic simply don't want others to get that scaling capacity and beat their models. In this environment, comparing frontier models and open-weights models doesn't make much sense, because open-weights developers simply don't have the raw compute power where frontier models lead.

As for custom single-pass kernels, Triton is close to or slightly better than PyTorch GPU, but since it is a one-pass matrix multiplication, it's hard to drastically beat a cuBLAS baseline. We will see a real difference if we do a single forward pass going through multiple fused layers.

In our next work, we will try to write fused Triton kernels (like SwiGLU or RMSNorm) to see if we can beat standard PyTorch baselines in real layer-by-layer forward passes. Until then, bye!

*Thanks for reading!*
