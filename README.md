# Kernel Swing

This project tests one question: can a targeted custom GPU kernel improve real
end-to-end LLM serving performance for a specific open-weight model on this GPU?

The first target model directory is auto-detected as:

```text
Qwen3.5-0.8B/
```

All model-specific kernels, tests, benchmarks, and reports live inside that
directory. Top-level scripts such as `serve.sh` and `serve_vllm.py` remain shared
entrypoints.

## Six-Phase Workflow

1. Baseline Serving Benchmark
2. Profile Bottlenecks
3. Implement One Small Kernel
4. Kernel Microbenchmark
5. Serving Engine Integration
6. End-to-End Serving Benchmark

Do not write or enable a custom kernel before profiling. Microbenchmark wins do
not count as success unless the same change improves serving metrics such as
TTFT, ITL, output tokens/sec, requests/sec, p50/p95/p99 latency, GPU memory, and
stable concurrency.

## Current Phase Gate

The repo now has phase-gated tooling, but Phase 1 has not been run yet.

Missing required artifacts:

```text
Qwen3.5-0.8B/reports/baseline_results.jsonl
Qwen3.5-0.8B/reports/baseline_summary.md
Qwen3.5-0.8B/reports/profile_summary.md
```

Because those files do not exist yet, RMSNorm kernel implementation and
microbenchmarking are blocked.

## Phase 1: Baseline Serving Benchmark

Start a server with custom kernels disabled:

```bash
./serve.sh Qwen3.5-0.8B --backend vllm
```

In another shell, run the required matrix:

```bash
python Qwen3.5-0.8B/benchmarks/bench_serving.py \
  --model-dir Qwen3.5-0.8B \
  --engine vllm \
  --base-url http://localhost:8000 \
  --output Qwen3.5-0.8B/reports/baseline_results.jsonl
```

This writes:

```text
Qwen3.5-0.8B/reports/baseline_results.jsonl
Qwen3.5-0.8B/reports/baseline_summary.md
```

If the server is not reachable, the benchmark exits non-zero and does not fake
results.

## Phase 2: Profile Bottlenecks

After Phase 1 exists, create the profile summary template:

```bash
python Qwen3.5-0.8B/benchmarks/bench_serving.py \
  --model-dir Qwen3.5-0.8B \
  --engine vllm \
  --phase profile-template
```

Then fill `Qwen3.5-0.8B/reports/profile_summary.md` with real profiler evidence
from PyTorch profiler, Nsight Systems, Nsight Compute, vLLM/SGLang metrics, or
nvidia-smi logs. Phase 3 remains blocked until the file explicitly contains:

```text
RMSNORM_JUSTIFIED: yes
```

Only add that line when profiling shows RMSNorm is a real bottleneck.

## Phase 3: RMSNorm Correctness

After profiling justifies RMSNorm, replace the placeholder in:

```text
Qwen3.5-0.8B/kernels/rmsnorm_triton.py
```

with one focused Triton RMSNorm implementation, then run:

```bash
python Qwen3.5-0.8B/tests/test_rmsnorm_correctness.py \
  --model-dir Qwen3.5-0.8B
```

This checks fp32, fp16, bf16 when supported, decode-like token counts, prefill-like
token counts, and the model hidden size from `config.json`.

## Phase 4: Kernel Microbenchmark

Run only after correctness passes:

```bash
python Qwen3.5-0.8B/benchmarks/bench_rmsnorm.py \
  --model-dir Qwen3.5-0.8B
```

This writes:

```text
Qwen3.5-0.8B/reports/kernel_rmsnorm_results.jsonl
Qwen3.5-0.8B/reports/kernel_rmsnorm_summary.md
```

Phase 4 compares PyTorch eager, torch.compile when available, and Triton RMSNorm.

## Phase 5: Serving Engine Integration

Only integrate after Phase 4 shows a meaningful win for relevant shapes. Keep the
fallback path and use this opt-in toggle:

```bash
USE_CUSTOM_RMSNORM=1 ./serve.sh Qwen3.5-0.8B --backend vllm
```

Record the integration state in:

```text
Qwen3.5-0.8B/reports/integration_notes.md
```

Phase 6 only unlocks when that file contains:

```text
INTEGRATION_READY: yes
```

If integration is not safe, explain the blocker there and leave that marker out.

## Phase 6: End-to-End Serving Benchmark

Run the same matrix with custom RMSNorm enabled and disabled:

```bash
python Qwen3.5-0.8B/benchmarks/bench_serving.py \
  --model-dir Qwen3.5-0.8B \
  --engine vllm \
  --phase e2e \
  --base-url http://localhost:8000 \
  --output Qwen3.5-0.8B/reports/e2e_results.jsonl
```

This writes:

```text
Qwen3.5-0.8B/reports/e2e_results.jsonl
Qwen3.5-0.8B/reports/e2e_summary.md
```

## Final Decision

Use one of these outcomes:

```text
KEEP
```

The custom kernel improves end-to-end serving performance meaningfully.

```text
REJECT
```

The custom kernel does not improve serving, even if the microbenchmark improved.

```text
REWORK
```

The custom kernel helps only some shapes, breaks important paths, increases
memory, affects CUDA graphs, or is unstable.
