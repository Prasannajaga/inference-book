# Inference Book

This is inference-book. everything you need to know about llm inference from writing custom CUDA kernels to building production cluster infrastructure happens here.

## 30-Day Series

- [30-Day LLM Inference Series](30-day-series/README.md)
  - [Day 0: Prerequisites & Roadmap](30-day-series/day-0/Day0.md)
  - [Day 1: What Happens When a User Sends an LLM Request?](30-day-series/day-1/Day1.md)
  - [Day 2: Prefill vs Decode](30-day-series/day-2/README.md)
  - [Day 3: KV Cache & Quantization](30-day-series/day-3/README.md)
  - [Day 4: Inside vLLM and SGLang](30-day-series/day-4/README.md)
  - [Day 5: PagedAttention, Paged Memory & RadixTree](30-day-series/day-5/README.md)
  - [Day 6: Static Batching vs Continuous Batching](30-day-series/day-6/README.md)
  - [Day 7: Chunked Prefill](30-day-series/day-7/README.md)
  - [Day 8: The Inference Scheduler](30-day-series/day-8/README.md)
  - [Day 10: Backpressure and Overload Protection](30-day-series/day-10/README.md)
  - [Day 11: Deploying an LLM as a Service on Kubernetes](30-day-series/day-11/README.md)
  - [Day 12: Deploying Multi-Node Inference Clusters](30-day-series/day-12/README.md)
  - [Day 13: Distributed Prefix Caching & RadixAttention](30-day-series/day-13/README.md)

## Kernels

- [CUDA Kernels](kernels/)
  - [CUDA Matmul Notebook](kernels/basics/matmul.ipynb)
  - [CUDA Fundamentals Blog](kernels/basics/blog.md)
  - [GEMV CUDA Kernels](kernels/GEMV/gemv_kernels.cu)
  - [GEMV C++ PyTorch Bindings](kernels/GEMV/gemv_bindings.cpp)
  - [GEMV Deep Dive](kernels/GEMV/gemv.md)
  - [GEMV Benchmark Notebook](kernels/GEMV/gemv_benchmark.ipynb)

## Serving Engines & Runtimes

- [vLLM Engine](vllm/)
  - [vLLM Architecture Deep Dive](vllm/vllm.md)
  - [vLLM Dev Guide](vllm/dev.md)
- [SGLang Engine](sglang/)
  - [SGLang Overview](sglang/sglang.md)
- [llm-d Router](llm-d/)
  - [llm-d Design Spec](llm-d/design.md)
