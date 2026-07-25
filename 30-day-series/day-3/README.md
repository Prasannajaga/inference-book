Day 3/30 of inference infrastructure series

kv cache: the state behind LLM inference

yesterday on day 2 we broke down how prefill and decode run on the gpu — prefill chunking prompt tokens together to max out TFLOPS, while decode generates one token at a time bound by HBM memory bandwidth. today we dive into the core state that connects these two phases and consumes most of your GPU memory: the KV cache.

# part 1: why keys and values exist

when a transformer processes a prompt, it calculates three vectors for every token at every layer:

* Query (Q) — what this token looks for in other tokens
* Key (K) — what information this token exposes for matching
* Value (V) — the content representation passed forward when attention matches

during decode, generating token N+1 requires computing attention against all past tokens 1...N.

without a cache, the GPU would have to re-run matrix projections across all past tokens on every single decode step. the KV cache solves this by storing Key and Value vectors in GPU VRAM during prefill. each decode step computes Q, K, and V for only the newest token, appends K and V to the cache pool, and attends over all cached keys and values.

saving past keys and values avoids repeating full matrix projections for old tokens, but storing this state requires substantial GPU memory.

# part 2: how the KV cache affects inference

storing past keys and values accelerates decode speed, but it also creates a massive, dynamically growing footprint in GPU memory.

its size directly dictates five core serving limits:

* maximum context length — more tokens mean more cached vectors, setting a physical limit on sequence length before VRAM runs out.
* concurrent requests — every active request holds its own cache, limiting how many users a single GPU can serve at once.
* decode latency — reading past keys and values off HBM on every step makes larger caches slow down token generation speed.
* request preemption — when VRAM is full, the engine must pause, swap to CPU, or evict active requests and recompute them later.
* serving cost and throughput — smaller KV caches let you run larger batch sizes, directly lowering serving cost per token.

here is the formula to calculate raw KV memory required per token:

KV bytes per token = 2 x layers x KV heads x head dimension x precision bytes

where:

* 2 = one Key vector and one Value vector per layer
* layers = total transformer layers storing state
* KV heads = count of Key and Value heads per layer
* head dimension = vector dimension size per head (e.g. 128)
* precision bytes = 2 bytes for FP16/BF16, 1 byte for FP8/INT8

let's look at standard architectural configurations using an 80-layer model in 16-bit precision (2 bytes):

1. standard multi-head attention (64 KV heads, head dim 128):
KV bytes per token = 2 x 80 x 64 x 128 x 2 = 2,621,440 bytes ≈ 2.62 MB per token

2. grouped-query attention (8 KV heads, head dim 128):
KV bytes per token = 2 x 80 x 8 x 128 x 2 = 327,680 bytes ≈ 0.328 MB per token

3. multi-query attention (1 KV head, head dim 128):
KV bytes per token = 2 x 80 x 1 x 128 x 2 = 40,960 bytes ≈ 0.041 MB per token

4. multi-head latent attention (512-dim latent + 64-dim positional Key):
KV bytes per token = 80 x (512 + 64) x 2 = 92,160 bytes ≈ 0.092 MB per token

here is how raw KV cache memory payload scales per request as context length grows across these attention types:

Context Tokens | 64 KV Heads (MHA) | 8 KV Heads (GQA) | 1 KV Head (MQA) | MLA (Absorbed Layout)
---------------|-------------------|------------------|------------------|----------------------
1,000          | 2.62 GB           | 0.33 GB          | 0.041 GB         | 0.092 GB
4,000          | 10.49 GB          | 1.31 GB          | 0.164 GB         | 0.369 GB
16,000         | 41.94 GB          | 5.24 GB          | 0.655 GB         | 1.475 GB
32,000         | 83.89 GB          | 10.49 GB         | 1.311 GB         | 2.949 GB
128,000        | 335.54 GB         | 41.94 GB         | 5.243 GB         | 11.796 GB

# part 3: weight and KV-cache quantization

quantization reduces the numerical precision used to store values in GPU memory, converting 16-bit floating point numbers (FP16 or BF16) into 8-bit (FP8 or INT8) or 4-bit (INT4) formats.

in an inference server, quantization can be applied to two separate memory regions:

1. model weights
2. KV-cache state

these target two distinct memory bottlenecks.

model weight quantization reduces the memory required to load the model parameters into VRAM at startup. for example, a 70-billion parameter model requires:

* FP16 / BF16 (2 bytes per weight): ~140 GB raw payload
* FP8 / INT8 (1 byte per weight): ~70 GB raw payload
* INT4 (0.5 bytes per weight): ~35 GB raw payload

weight quantization frees base GPU memory so more VRAM can be allocated to the KV cache pool.

KV-cache quantization directly shrinks the stored size of Key and Value vectors in VRAM:

* FP16 / BF16 KV cache: 2 bytes per element (baseline)
* FP8 / INT8 KV cache: 1 byte per element (~2x raw memory reduction)
* INT4 KV cache: 0.5 bytes per element (~4x raw memory reduction)

here is a side-by-side comparison of quantization combinations and memory savings:

Quantization Combo | Weight Payload (70B Scale) | Weight Saving | KV Cache Element Size | KV Cache Saving
-------------------|----------------------------|---------------|-----------------------|----------------
FP16 Weights + FP16 KV | ~140 GB | Baseline (1x) | 2 bytes | Baseline (1x)
FP8 Weights + FP8 KV | ~70 GB | ~2x smaller | 1 byte | ~2x smaller
INT4 Weights + FP16 KV | ~35 GB | ~4x smaller | 2 bytes | Baseline (1x)
INT4 Weights + FP8 KV | ~35 GB | ~4x smaller | 1 byte | ~2x smaller

it is important to note that you do not always need full BF16 or FP16 precision in production.

if you run evaluations on lower-bit formats (like FP8 or INT4) using your specific downstream tasks and the output quality holds up, deploying in low-bit precision can save massive compute and hardware costs — sometimes cutting serving infrastructure overhead by orders of magnitude.

run your own evals first. if low-bit quantization passes your quality threshold, you unlock huge savings in VRAM, memory bandwidth, and GPU deployment costs.

# final takeaway

the KV cache is the core persistent state that connects prefill and decode phases in autoregressive LLM inference.

storing Key and Value vectors avoids recomputing attention projections for past prompt tokens, but requires careful memory capacity planning as context lengths and concurrency scale.

understanding the formula for KV memory bytes per token, the structural trade-offs of attention variants like MHA, MQA, GQA, and MLA, and the mechanics of quantization allows you to optimize serving throughput and maximize hardware efficiency.

tomorrow on Day 4, we step inside engine architectures to explore how schedulers, block managers, and paged attention execute this state management in real time.
