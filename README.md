# Inference Book

This is inference-book. everything you need to know about llm inference from writing custom CUDA kernels to building production cluster infrastructure happens here.

## 30-Day Series

- [30-Day LLM Inference Series](30-day-series/README.md)
  - [Day 0: Prerequisites & Roadmap](30-day-series/day-0/Day0.md)
  - [Day 1: Request Lifecycle](30-day-series/day-1/Day1.md)

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
  - [llm-d PR Overview](llm-d/PR.md)
