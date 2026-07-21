# Kernel Swings

Everything I learned about kernels goes here.

You can find kernel notebooks under `kernels/`.

## CLI Usage

### 1. Serving Models

Use the root `serve.sh` script to launch the serving backend. This script automates setting up the environment and configuration parameters (such as sequence length and VRAM limits).

#### Serve with vLLM
```bash
./serve.sh <model_path> --backend vllm [additional_args...]
```
*Example with a local model:*
```bash
./serve.sh /data/nemostation/outputs/Marlin-2B-gptq --backend vllm
```

#### Serve with SGLang
```bash
./serve.sh <model_path> --backend sglang [additional_args...]
```
*Example with a local model:*
```bash
./serve.sh /data/nemostation/outputs/Marlin-2B-gptq --backend sglang
```

---

### 2. Benchmarking

Use `bench/llm_bench.py` to benchmark the running model server.

#### Standard Benchmark
Runs a fixed number of requests at a set concurrency:
```bash
python bench/llm_bench.py --url http://localhost:8000 --requests 50 --concurrency 5
```

#### Custom Output Directory Name
Use the `--name` parameter to save results to a specific subdirectory under `--out` (defaults to `bench_results/<name>`):
```bash
python bench/llm_bench.py --url http://localhost:8000 --requests 50 --concurrency 5 --name my_benchmark_run
```

#### Concurrency Sweep
Runs benchmarking sequentially across a sweep of concurrency levels:
```bash
python bench/llm_bench.py --url http://localhost:8000 --concurrency-sweep 1,2,4,8,16 --name my_sweep_run
```

#### Plotting Existing Results
If a benchmark run has already completed (and `summary.json` exists in the output folder), you can re-run the same command with `--plot` to generate the plots without running the benchmark again:
```bash
python bench/llm_bench.py --url http://localhost:8000 --name my_benchmark_run --plot
```

#### Isolation / Contamination Test
Runs a basic concurrent prompt contamination test:
```bash
python bench/llm_bench.py --url http://localhost:8000 --isolation-test --requests 10 --concurrency 2
```

