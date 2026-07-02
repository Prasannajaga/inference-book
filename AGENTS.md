# AGENTS.md

## Project Goal

This project is for proving whether a custom GPU kernel can improve real LLM inference serving performance for one open-weight model.

Each model lives in its own directory (for example `Qwen3.5-0.8B/`) and contains its own implementation files such as `modeling_qwen.py`, `rmsnorm.py`, and `swiglu.py`. All model-specific optimizations, patches, and experiments must stay inside that model directory.

The goal is not to write kernels blindly. The goal is to profile, identify a real bottleneck inside a specific model directory, implement one targeted kernel, and keep it only if end-to-end serving performance improves.

Target serving engines:

* vLLM
* SGLang

Initial kernel target:

* Start with RMSNorm (inside the model directory, e.g. `Qwen3.5-0.8B/rmsnorm.py`)
* Then try SiLU + Mul / SwiGLU (e.g. `Qwen3.5-0.8B/swiglu.py`)
* Then RoPE
* Do not start with attention, paged attention, KV-cache layout, or custom GEMM

## Core Rule

Never write or integrate a custom kernel before profiling.

A kernel is only useful if it improves real serving metrics:

* TTFT: time to first token
* ITL: inter-token latency
* output tokens/sec
* requests/sec
* p50/p95/p99 latency
* GPU memory usage
* max stable concurrency

If the kernel improves only a microbenchmark but does not improve serving performance, reject it.

## Non-Goals

Do not build a new serving engine.
Do not rewrite vLLM or SGLang scheduling.
Do not write custom attention first.
Do not write custom matmul/GEMM first.
Do not optimize without a baseline.
Do not claim performance wins without reproducible benchmark data.

## Six-Phase Workflow

### Phase 1: Baseline Serving Benchmark

Run the selected model directory (for example `Qwen3.5-0.8B/`) using `serve.sh` or `serve_vllm.py` without any custom kernel modifications.

Capture:

* model name (directory name)
* quantization mode
* GPU name
* GPU memory
* CUDA version
* PyTorch version
* vLLM/SGLang version
* prompt length
* output length
* concurrency
* TTFT
* ITL
* tokens/sec
* requests/sec
* p50/p95/p99 latency
* GPU memory usage
* GPU utilization

Benchmark workloads:

* short prompt, short output
* short prompt, long output
* long prompt, short output
* long prompt, long output
* low concurrency
* medium concurrency
* high concurrency until latency degrades

Required output (inside the model directory):

```text
Qwen3.5-0.8B/reports/baseline_results.jsonl
Qwen3.5-0.8B/reports/baseline_summary.md
Qwen3.5-0.8B/reports/plots/
```

Do not move to Phase 2 until baseline numbers exist.

### Phase 2: Profile Bottlenecks

Profile the baseline run before changing kernels.

Use at least one of:

* PyTorch profiler
* Nsight Systems
* Nsight Compute
* vLLM/SGLang internal metrics
* nvidia-smi logs

Identify where time is spent:

* attention
* GEMM
* RMSNorm
* RoPE
* activation
* sampling
* KV-cache movement
* scheduler overhead
* tokenization
* HTTP overhead

Required output (inside the model directory):

```text
Qwen3.5-0.8B/reports/profile_summary.md
Qwen3.5-0.8B/reports/profiles/
```

The profile summary must answer:

1. Which operation is hot?
2. How much time does it consume?
3. Is it hot in prefill, decode, or both?
4. Why is this operation a valid kernel target?
5. What existing implementation is used today (e.g. `rmsnorm.py`, `swiglu.py`)?

Do not write a kernel unless the profile justifies it.

### Phase 3: Implement One Small Kernel

Implement only one kernel target at a time inside the model directory.

First target:

```text
RMSNorm
```

Modify or extend:

```text
Qwen3.5-0.8B/rmsnorm.py
```

Add supporting files if needed:

```text
Qwen3.5-0.8B/kernels/
  rmsnorm_ref.py
  rmsnorm_triton.py

Qwen3.5-0.8B/tests/
  test_rmsnorm_correctness.py

Qwen3.5-0.8B/benchmarks/
  bench_rmsnorm.py
```

Correctness requirements:

* compare against PyTorch reference
* test fp16
* test bf16 if GPU supports it
* test fp32
* test decode-like shapes
* test prefill-like shapes
* test model hidden size
* test non-power-of-two token counts

Do not integrate into vLLM or SGLang until standalone correctness and microbenchmark results exist.

### Phase 4: Kernel Microbenchmark

Benchmark kernel variants independently.

Compare:

* PyTorch eager reference
* torch.compile reference if applicable
* Triton custom kernel
* existing implementation inside the model directory

Measure:

* latency in milliseconds
* bandwidth in GB/s
* output correctness
* dtype support
* shape coverage

Required benchmark shapes:

```text
Decode-like:
[1, hidden_dim]
[4, hidden_dim]
[8, hidden_dim]
[16, hidden_dim]
[32, hidden_dim]

Prefill-like:
[128, hidden_dim]
[512, hidden_dim]
[2048, hidden_dim]
[8192, hidden_dim]
```

Required output (inside the model directory):

```text
Qwen3.5-0.8B/reports/kernel_rmsnorm_results.jsonl
Qwen3.5-0.8B/reports/kernel_rmsnorm_summary.md
Qwen3.5-0.8B/reports/plots/
```

Move forward only if the custom kernel is correct and meaningfully faster for the shapes that matter.

### Phase 5: Serving Engine Integration

Integrate the kernel into one serving engine using the model directory.

Preferred order:

1. SGLang integration if the model path is easier to patch.
2. vLLM integration if using vLLM CustomOp or a local model layer patch.

Use existing entrypoints:

```text
serve.sh
serve_vllm.py
```

Rules:

* Keep the original path as fallback.
* Make the custom kernel enable/disable configurable.
* Do not permanently replace engine internals without an A/B switch.
* Avoid Python-side dynamic behavior in the hot path.
* Avoid file I/O, JIT compilation, imports, printing, or CPU sync inside model forward.

SGLang note:

* If the custom kernel is used inside the model forward path, wrap it with SGLang `register_custom_op` when needed for Piecewise CUDA Graph compatibility.

vLLM note:

* If using vLLM custom op path, register a vLLM `CustomOp` and make sure it can be enabled/disabled through custom op configuration.

### Phase 6: End-to-End Serving Benchmark

Run the same benchmark matrix from Phase 1 using the same model directory.

Compare:

* baseline engine
* custom kernel enabled
* custom kernel disabled/fallback

Measure:

* TTFT
* ITL
* tokens/sec
* requests/sec
* p50/p95/p99 latency
* GPU memory usage
* max stable concurrency

Required output (inside the model directory):

```text
Qwen3.5-0.8B/reports/e2e_results.jsonl
Qwen3.5-0.8B/reports/e2e_summary.md
Qwen3.5-0.8B/reports/plots/
```

Final decision must be one of:

```text
KEEP: custom kernel improves end-to-end serving performance.
REJECT: custom kernel does not improve end-to-end serving performance.
REWORK: custom kernel helps only specific shapes or breaks important paths.
```

Do not call the experiment successful unless end-to-end serving improves.

## Repository Structure

```text
repo-root/
  README.md
  serve.sh
  serve_vllm.py
  uv.lock

  Qwen3.5-0.8B/
    config.json
    modeling_qwen.py
    rmsnorm.py
    swiglu.py

    kernels/
    benchmarks/
    tests/
    reports/
```

Each model directory is self-contained. All optimizations, kernels, benchmarks, and reports must live inside that model directory.

## Agent Behavior

When asked to implement something, follow this order:

1. Identify which model directory is being worked on.
2. Check whether the required previous phase output exists inside that directory.
3. If it does not exist, create the missing phase output first.
4. If the user asks to skip profiling and write a kernel, reject that path and explain that profiling is required.
5. If the user asks to start with attention, push back and recommend RMSNorm first.
6. If the benchmark does not show end-to-end improvement, reject the kernel.
7. Do not claim success from microbenchmark numbers alone.

## Success Criteria

The project succeeds only if it produces a reproducible answer to this question:

> Can a targeted custom kernel improve real serving performance for this specific model directory on this GPU?

A successful result can be either:

* Yes, the kernel improves serving performance and we keep it.
* No, the kernel does not improve serving performance and we reject it with evidence.

Both are valid outcomes.
