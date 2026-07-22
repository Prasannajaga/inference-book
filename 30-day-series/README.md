# 30 Days of LLM Inference Infrastructure

## Goal of the Series

Build a complete mental model of how a single model running on one GPU becomes a reliable production inference platform capable of serving large-scale traffic.

The series follows one continuous progression:

```text
Single Request
    ↓
Single GPU
    ↓
Efficient Scheduling
    ↓
KV Cache Management
    ↓
Multiple GPU Pods
    ↓
Intelligent Routing
    ↓
Multi-GPU / Multi-Node
    ↓
Autoscaling
    ↓
Reliability + SLOs
    ↓
Benchmarking + Capacity Planning
```

The central question throughout the series is:

> How do we serve more users with the GPUs we have, without causing OOMs, latency spikes, bad cache reuse, or SLO violations?

---

# Phase 1 — Understand the Inference Request

## Day 1 — What Happens When a User Sends an LLM Request?

### Goal

Understand the complete lifecycle of one inference request.

### Cover

- OpenAI-compatible API request
- API gateway
- tokenizer
- request validation
- scheduler
- model engine
- GPU execution
- token streaming
- response completion

### Request flow

```text
User
  ↓
API Gateway
  ↓
Request Validation
  ↓
Tokenizer
  ↓
Scheduler
  ↓
Inference Engine
  ↓
GPU
  ↓
Generated Tokens
  ↓
Streaming Response
```

### Questions to answer

- Where does TTFT begin?
- Where does queue latency happen?
- What component owns the KV cache?
- What component decides when a request reaches the GPU?

### Tokenizer fundamentals

- BPE, SentencePiece, tiktoken
- why token count ≠ word count
- chat templates and special tokens
- token counting for billing and capacity planning
- tokenizer latency at scale

### Streaming infrastructure

- Server-Sent Events (SSE) vs WebSockets vs HTTP chunked transfer
- proxy and load balancer configuration for long-lived connections
- client-side token assembly
- timeout handling for slow generations
- streaming interaction with tool calling

### Multimodal and vision-language model ingestion

- image and video preprocessing on GPU before LLM prefill
- vision encoder token expansion (1,000+ tokens per image)
- impact on prefill latency and token budgets
- VLM chunked prefill considerations

---

## Day 2 — Prefill vs Decode

### Goal

Understand why one LLM request contains two very different workloads.

### Prefill

The model processes the input prompt.

```text
Prompt Tokens
    ↓
Prefill
    ↓
Attention Computation
    ↓
KV Cache Created
```

Characteristics:

- processes many tokens together
- relatively compute-heavy
- affects TTFT
- large prompts create expensive prefill work

### Decode

The model generates output tokens.

```text
KV Cache
   ↓
Generate Token 1
   ↓
Generate Token 2
   ↓
Generate Token 3
```

Characteristics:

- normally generates one token per sequence at each decoding step
- repeatedly reads model weights and KV cache
- often memory-bandwidth-sensitive
- affects TPOT / inter-token latency

### Why this matters

Prefill and decode compete for the same GPU resources.

This leads directly to:

- continuous batching
- chunked prefill
- scheduler design
- P/D disaggregation

### GPU fundamentals for inference

Understanding why GPU inference is expensive and what hardware constraints drive every optimization in this series.

```text
Compute Throughput (TFLOPS)
    ↓
How fast the GPU does math

Memory Bandwidth (TB/s)
    ↓
How fast the GPU reads data

HBM Capacity (GB)
    ↓
How much state fits on the GPU
```

Key concepts:

- GPU architecture at the infrastructure level: streaming multiprocessors, HBM, memory bandwidth
- why decode is memory-bandwidth-bound (reads all model weights per generated token, very low arithmetic intensity)
- why prefill is compute-bound (processes many tokens in parallel, high arithmetic intensity)
- roofline model: determines whether an operation is compute-limited or memory-limited
- hardware specs that matter for capacity planning: A100 / H100 / H200 / B200
- NVLink vs PCIe bandwidth (relevant for tensor parallelism in Day 18)

Every optimization in this series exists because GPU memory and bandwidth are scarce and expensive.

---

## Day 3 — KV Cache: The State Behind LLM Inference

### Goal

Understand why KV cache is one of the main infrastructure constraints in LLM serving.

### Cover

- what keys and values represent
- why attention reuses previous tokens
- why recomputing previous tokens is expensive
- why KV memory grows with sequence length
- how active requests compete for KV memory
- KV cache quantization (FP8 / INT8 KV cache) and HBM footprint reduction
- attention architecture variants and their KV cache impact

### Attention architecture variants

Different attention architectures determine KV cache size per token. This directly affects memory planning and maximum concurrency.

```text
MHA (Multi-Head Attention)
    Full KV per head
    Baseline memory cost

MQA (Multi-Query Attention)
    Shared KV across all heads
    Much smaller KV footprint

GQA (Grouped-Query Attention)
    KV shared within groups
    Middle ground (Llama 3)

MLA (Multi-head Latent Attention)
    Compressed latent KV
    Smallest footprint (DeepSeek-V2/V3)
```

Questions:

- how many KV bytes per token per layer for each variant
- why GQA allows more concurrent requests than MHA for the same GPU memory
- how this directly affects the capacity math in Day 30

### Model weight quantization

Quantization is often the single most impactful deployment decision for inference infrastructure.

```text
70B Model in FP16
    ~140 GB
    Requires 2× A100-80GB

70B Model in INT4 (AWQ / GPTQ)
    ~35 GB
    Fits on 1× A100-80GB
```

Cover:

- weight quantization formats: FP16 → FP8 / INT8 / INT4
- quantization methods for deployment: AWQ, GPTQ, FP8 dynamic, GGUF
- quality vs throughput trade-offs (when quality degrades noticeably)
- how quantization interacts with tensor parallelism
- fast quantized inference kernels: Marlin, Machete
- when to quantize (almost always) and when not to (quality-critical applications)

### Mental model

```text
Request A
Tokens: 10,000
    ↓
KV Cache A

Request B
Tokens: 20,000
    ↓
KV Cache B

Request C
Tokens: 50,000
    ↓
KV Cache C

GPU Memory
┌──────────────────────────┐
│ Model Weights            │
├──────────────────────────┤
│ KV Cache A               │
│ KV Cache B               │
│ KV Cache C               │
├──────────────────────────┤
│ Runtime / Workspace      │
└──────────────────────────┘
```

### Core infrastructure problem

More concurrent users usually means more active KV state.

Eventually:

```text
Concurrency ↑
    ↓
KV Usage ↑
    ↓
Available GPU Memory ↓
    ↓
Preemption / Recompute / OOM
```

---

## Day 4 — Inside vLLM and SGLang

### Goal

Understand the major components inside a modern inference engine.

### Architecture

```text
HTTP Server
    ↓
Tokenizer
    ↓
Request Queue
    ↓
Scheduler
    ↓
Batch Formation
    ↓
Model Executor
    ↓
GPU Workers
    ↓
KV Cache Manager
```

### Explore

- request lifecycle
- engine loop
- scheduler
- worker
- model executor
- KV cache manager (hash-based blocks vs RadixTree / Trie indexing)
- CUDA Graphs & Piecewise CUDA Graphs (CPU launch overhead reduction)
- distributed executor
- streaming output

### Compare

Focus on architecture rather than declaring a universal winner.

Study:

- vLLM
- SGLang

Questions:

- Who owns scheduling?
- Who owns KV blocks?
- How are distributed workers created?
- How does the engine expose metrics?

---

## Day 5 — PagedAttention and Paged KV Memory

### Goal

Understand how paged KV allocation improves memory management.

### Traditional idea

```text
Request
    ↓
Large Contiguous KV Allocation
```

Problems:

- fragmentation
- unpredictable sequence length
- wasted reserved memory

### Paged model

```text
Request A
 ├── Block 2
 ├── Block 7
 └── Block 11

Request B
 ├── Block 1
 └── Block 8
```

### Cover

- logical blocks
- physical blocks
- allocation
- freeing
- block tables
- fragmentation reduction
- relationship to concurrency
- FlashAttention-2/3 vs PagedAttention
- FlashDecoding (Split-K parallelization over sequence length during decode steps)

---

# Phase 2 — Make One GPU Serve Traffic Efficiently

## Day 6 — Static Batching vs Continuous Batching

### Goal

Understand how inference engines keep GPUs busy while requests arrive and finish at different times.

### Static batching

```text
Batch
├── Request A ─────── done
├── Request B ───────────── done
└── Request C ───── done

GPU waits for batch boundary
```

### Continuous batching

```text
Step 1: A B C
Step 2: A B C
Step 3: A B
Step 4: A B D
Step 5: B D E
```

### Cover

- iteration-level scheduling
- dynamic admission
- sequence completion
- throughput vs latency
- GPU utilization

---

## Day 7 — The Inference Scheduler

### Goal

Understand who gets GPU time and why scheduling policy matters.

### Flow

```text
Waiting Requests
      ↓
Scheduler
 ┌────┼─────┐
 ↓    ↓     ↓
R1   R2    R3
      ↓
Token Budget
      ↓
GPU Batch
```

### Cover

- FCFS
- priority scheduling
- token budgets
- maximum batched tokens
- maximum active sequences
- decode prioritization
- guided generation / structured output overhead (FSM schema compilation, logits processor masking latency)
- starvation
- fairness
- preemption

### Key question

> When GPU memory and compute are limited, which request should run next?

---

## Day 8 — Chunked Prefill

### Goal

Understand how large prompts can be prevented from blocking decode traffic.

### Without chunking

```text
100K Token Prompt
       ↓
Very Large Prefill
       ↓
Decode Requests Wait
       ↓
TPOT / latency spikes
```

### With chunking

```text
Prefill Chunk 1
      ↓
Decode Batch
      ↓
Prefill Chunk 2
      ↓
Decode Batch
```

### Cover

- prefill chunk size
- token budget
- mixing prefill and decode
- TTFT vs TPOT trade-offs

---

## Day 9 — Admission Control

### Goal

Understand why a production inference system should sometimes queue or reject work before it reaches the engine.

### Architecture

```text
Incoming Request
      ↓
Admission Control
 ┌────┼─────────┐
 ↓    ↓         ↓
RUN  QUEUE     REJECT
```

### Admission signals

- prompt length
- requested output length
- context limit
- queue depth
- running request count
- estimated KV requirement
- current GPU memory pressure
- request deadline
- input/output guardrail model overhead (parallel safety filtering, prompt injection screening latency)

### Possible decisions

```text
RUN_NOW
QUEUE
REJECT_TOO_LARGE
REJECT_OVERLOADED
```

### Multi-turn and agentic inference patterns

Production LLM traffic is increasingly stateful and agentic.

```text
Tool Calling Flow

User Prompt
    ↓
Generate (call function)
    ↓
Parse Tool Call
    ↓
Execute External API
    ↓
Re-prompt with Result
    ↓
Generate Final Response
```

Admission control must account for:

- multi-turn conversations with growing context windows across turns
- tool calling loops: one user action may trigger 5–20 inference calls
- agentic orchestration: each step compounds KV cache usage and compute
- session affinity: which pod holds the conversation's KV state
- estimated total work across a full agentic loop, not just one call

---

## Day 10 — Backpressure and Overload Protection

### Goal

Understand what happens when traffic exceeds serving capacity.

### Failure pattern

```text
Traffic ↑
   ↓
Queue ↑
   ↓
Waiting Time ↑
   ↓
TTFT ↑
   ↓
Timeouts ↑
   ↓
Retries ↑
   ↓
Even More Traffic
```

### Cover

- bounded queues
- queue timeout
- request timeout
- rate limiting
- load shedding
- retry budgets
- exponential backoff
- avoiding retry storms

### Core lesson

An overloaded system should fail predictably instead of collapsing unpredictably.

---

# Phase 3 — Scale From One GPU Pod to Many

## Day 11 — Deploying an LLM as a Service

### Goal

Move from a local inference server to a production deployment.

### Architecture

```text
Client
  ↓
Load Balancer
  ↓
Gateway
  ↓
Kubernetes Service
  ↓
Inference Pod
  ↓
vLLM / SGLang
  ↓
GPU
```

### Cover

- containers
- GPU-enabled Kubernetes nodes
- GPU device plugins
- deployments
- pods
- services
- readiness checks
- liveness checks
- startup probes

### Model loading and weight management

- model storage formats: SafeTensors, sharded checkpoints, GGUF
- model registries: HuggingFace Hub, S3, GCS
- download optimization: parallel downloads, pre-cached NVMe, shared filesystems (JuiceFS, EFS, Lustre)
- weight loading sequence: download → shard → load to GPU → NCCL init → ready
- why startup takes 2–10 minutes for large models
- container image optimization for large CUDA base images

### Security and multi-tenancy

- API key management and authentication
- per-tenant rate limiting and quotas
- data isolation (KV cache contains user data)
- prompt injection screening from an infrastructure perspective
- audit logging for compliance
- model access control: which tenants can access which models

---

## Day 12 — One Model, Multiple Replicas

### Goal

Understand horizontal replication.

```text
                   ┌── GPU Pod A
Client → Gateway ──┼── GPU Pod B
                   └── GPU Pod C
```

### Cover

- replicas
- model duplication
- memory cost
- routing
- independent queues
- independent KV caches
- Multi-LoRA / dynamic adapter serving (serving 100+ fine-tuned adapters per base model pod, dynamic weight loading, S-LoRA/Punica BGMV batched kernels)

### Key realization

LLM replicas may run the same model, but they do not necessarily have the same runtime state.

---

## Day 13 — Why Round-Robin Is Not Enough

### Goal

Understand why traditional stateless load balancing can waste useful KV state.

### Example

```text
Request 1
Prefix X
   ↓
Pod A
   ↓
KV for Prefix X cached

Request 2
Prefix X
   ↓
Round Robin
   ↓
Pod B

Result:
Prefix recomputed
```

### Better routing

```text
Request 2
Prefix X
   ↓
Cache-Aware Router
   ↓
Pod A
```

### Cover

- cache locality
- queue imbalance
- hot replicas
- cache reuse
- trade-off between locality and load

---

## Day 14 — Prefix-Cache-Aware Routing

### Goal

Understand how routers estimate which worker already has useful KV state.

### Flow

```text
Prompt
  ↓
Tokenize
  ↓
Split Into Blocks
  ↓
Hash Prefix Blocks
  ↓
Cache Directory / Metadata
  ↓
Estimate Matching Prefix

Pod A = 80%
Pod B = 40%
Pod C = 0%

  ↓
Choose Pod A
```

### Cover

- token blocks
- block hashes
- cumulative hashes
- longest-prefix match
- cache ownership
- cache eviction
- stale metadata

---

## Day 15 — Load-Aware Routing

### Goal

Understand why maximum cache reuse is not always the best routing decision.

### Example

```text
Pod A
Cache Match: 95%
Queue: 40

Pod B
Cache Match: 60%
Queue: 2
```

Choosing Pod A may reuse more KV but create unacceptable latency.

### Conceptual score

```text
score =
    cache_benefit
  - queue_penalty
  - KV_pressure
  - active_request_penalty
  - predicted_latency
```

### Cover

- queue depth
- running sequences
- KV utilization
- estimated work
- cache affinity
- latency prediction

---

# Phase 4 — Kubernetes-Native Inference Routing

## Day 16 — Gateway, InferencePool, and Endpoint Picker

### Goal

Understand model-aware routing in Kubernetes inference infrastructure.

### Conceptual flow

```text
User
  ↓
Gateway
  ↓
Inference Routing Logic
  ↓
Endpoint Picker
  ↓
InferencePool
  ↓
Selected GPU Pod
```

### Study

- Gateway API
- HTTPRoute
- InferencePool
- endpoint selection
- model-aware routing
- metrics and routing signals

### Questions

- Why is a normal Kubernetes Service insufficient for sophisticated LLM routing?
- Where does endpoint selection happen?
- What metadata is required to make a good routing decision?

---

## Day 17 — Designing the Router Data Plane

### Goal

Understand the hot path of every user request.

### Flow

```text
Request
  ↓
Proxy / Gateway
  ↓
Routing Metadata
  ↓
Endpoint Selection
  ↓
Selected Pod
  ↓
Inference
```

### Cover

- request interception
- routing decisions
- proxy overhead
- data plane vs control plane
- avoiding extra network hops
- failure behavior when routing logic is unavailable

---

# Phase 5 — Scale Across GPUs and Nodes

## Day 18 — Tensor Parallelism

### Goal

Understand how one model can be distributed across several GPUs.

```text
Model Layer
   ↓
┌─────────────┬─────────────┐
│ GPU 0       │ GPU 1       │
│ Tensor Part │ Tensor Part │
└─────────────┴─────────────┘
         ↓
Communication
```

### Cover

- why TP is needed
- model memory
- tensor splitting
- collective communication
- communication overhead
- NVLink
- PCIe

---

## Day 19 — TP, PP, DP, and EP

### Goal

Understand the main parallelism strategies used in inference.

### Tensor Parallelism

Split operations across GPUs.

### Pipeline Parallelism

Split model layers across stages.

### Data Parallelism

Replicate the model and serve independent request groups.

### Expert Parallelism

Distribute Mixture-of-Experts (MoE) experts across workers (e.g. DeepSeek-V2/V3, Mixtral, Qwen-MoE).

MoE inference infrastructure details:

- expert routing: how the gating network selects active experts per token
- expert load balancing: uneven expert activation creates GPU utilization imbalance
- capacity factor: limiting how many tokens each expert processes
- all-to-all communication overhead: tokens must reach the GPU holding the selected expert
- memory implications: MoE models have more total parameters but fewer active per token
- DeepSeek-V2/V3: shared experts + routed experts + MLA attention
- infrastructure impact: expert placement strategy across GPUs and nodes

### Comparison

```text
TP → split tensors
PP → split layers
DP → replicate models
EP → split experts (MoE all-to-all communication)
```

### Discuss

- what problem each strategy solves
- communication cost (AllReduce vs AllToAll)
- latency impact
- memory impact

---

## Day 20 — Multi-Node Inference

### Goal

Understand what changes when a model spans more than one server.

```text
Node A
┌──────────────┐
│ GPU GPU GPU  │
└──────────────┘
       ↕
High-Speed Network
       ↕
Node B
┌──────────────┐
│ GPU GPU GPU  │
└──────────────┘
```

### Cover

- intra-node communication
- inter-node communication
- NCCL
- RDMA
- network topology
- distributed worker startup
- failure domains
- spot instance preemption & node resiliency (graceful failover in multi-node TP/PP setups)

### Core lesson

Once inference crosses machines, network performance becomes part of model performance.

---

# Phase 6 — Disaggregated Inference Serving (E/P/D Topologies)

## Day 21 — Disaggregation Topologies: EPD, P/D, E/PD, and E/P/D

### Goal

Understand why separating multimodal encoding, prefill, and decode onto specialized worker pools optimizes LLM & VLM inference latency and resource utilization.

### Disaggregation Topologies

```text
1. EPD  (No Disaggregation)  : [Encode + Prefill + Decode] on 1 Worker
2. P/D  (Prefill / Decode)   : [Encode + Prefill Worker] ──KV Transfer──> [Decode Worker]
3. E/PD (Encode / P-D)       : [Encode Worker] ──Embeddings──> [Prefill + Decode Worker]
4. E/P/D (Full 3-Stage Split): [Encode Worker] ──Embeddings──> [Prefill Worker] ──KV Transfer──> [Decode Worker]
```

### Multimodal Encode Disaggregation Workflow

For multimodal requests (images, video, audio), vision encoders generate massive token overhead (1,000+ tokens per image). Offloading encoding to dedicated GPU workers isolates heavy vision processing from autoregressive decode workers.

```text
Client
  ↓
Inference Gateway / Envoy
  ↓ (Headers: x-encoder-hosts-ports, x-prefiller-host-port)
Decode Worker Sidecar
  ├── 1. Send multimodal content ──> [Encode Worker] (Processes image/video, returns embedding metadata)
  ├── 2. Send prompt + embeddings ─> [Prefill Worker] (Reads embeddings via EC_Connector, generates KV cache)
  └── 3. Execute local decode ─────> [Decode Worker] (Reads KV cache, streams generated tokens to client)
```

### Cover

- **Disaggregation Topologies**: EPD, P/D (EP/D), E/PD, and full E/P/D
- **Encode Worker Role**: Processing multimodal inputs (images, video, audio) into embedding references
- **Sidecar Coordination**: Decode-sidecar orchestration of `x-encoder-hosts-ports` and `x-prefiller-host-port` headers
- **Stage Deciders**: `prefix-based-pd-decider` and `always-disagg-multimodal-decider` logic
- **Trade-offs**: TTFT vs TPOT, network hop overhead (encode → prefill → decode), and stranded memory risks

---

## Day 22 — KV & Embedding Transfer in Disaggregated Serving

### Goal

Understand the transfer mechanisms, sidecar protocols, and connectors that enable high-speed data movement between Encode, Prefill, and Decode workers.

### Data Movement Pipeline

```text
[Encode Worker]
      │
      │ Embeddings (EC_Connector)
      ▼
[Prefill Worker]
      │
      │ KV Cache (Nixl / Mooncake / OffloadingConnector / P2P)
      ▼
[Decode Worker]
```

### Cover

- **Embedding Transfer**: `EC_Connector` for reading multimodal embedding references between Encode and Prefill workers
- **KV Transfer Connectors**:
  - `OffloadingConnector`: KV transfer over host CPU memory tier and P2P secondary tier
  - `NixlConnector`: NIXL-based GPU-Direct / RDMA KV transfer
  - `MooncakeConnector`: RDMA-accelerated KV transfer for Mooncake architecture
- **Sidecar Responsibilities**: Header parsing, validation of encode/prefill responses, pre-allocation of KV blocks, and zero-sidecar lightness on prefill/encode nodes
- **Labeling & Filtering**: Kubernetes `EndpointPickerConfig` label matching (`llm-d.ai/role`: `encode`, `prefill`, `decode`, `encode-prefill-decode`)

---

# Phase 7 — Distributed KV Cache Infrastructure

## Day 23 — Prefix Caching

### Goal

Understand reuse across independent requests.

```text
Shared System Prompt
        ↓
Request A computes prefix
        ↓
Prefix KV cached
        ↓
Request B has same prefix
        ↓
Reuse cached KV
```

### Cover

- exact prefix reuse
- block-level reuse
- system prompts
- repeated conversations
- RAG prefixes
- cache hit rate
- multi-turn conversation prefix reuse
- session affinity and prefix caching interaction
- agentic loop prefix reuse across tool-calling rounds

---

## Day 24 — KV Cache Affinity and Offloading

### Goal

Understand KV as a tiered storage problem.

```text
Fastest / Smallest

GPU HBM
   ↓
CPU RAM
   ↓
Local NVMe
   ↓
Remote KV Store

Slowest / Largest
```

### Cover

- GPU KV cache (FP16 vs FP8)
- CPU offload
- remote cache
- cache promotion
- eviction
- cache affinity
- bandwidth vs latency
- tiered storage tradeoffs (HBM → CPU RAM → Local NVMe)
- when offloading helps
- when offloading hurts

---

# Phase 8 — Speculative Decoding

## Day 25 — Speculative Decoding

### Goal

Understand how draft-and-verify can accelerate token generation.

```text
Draft Model
    ↓
Predict A B C D
    ↓
Target Model Verification
    ↓
Accept A B C
Reject D
```

### Cover

- draft model
- target model
- acceptance rate
- verification
- additional memory
- extra scheduling complexity
- cases where speculation helps
- cases where speculation adds overhead

---

# Phase 9 — Autoscaling GPU Inference

Autoscaling deserves its own dedicated phase because GPU inference does not scale like a normal stateless web application.

## Day 26 — Autoscaling Signals: What Should Trigger Scaling?

### Goal

Understand why CPU utilization alone is usually a poor signal for LLM inference.

### Candidate signals

```text
Request Rate
Queue Depth
Pending Tokens
Running Requests
KV Cache Utilization
TTFT
TPOT
GPU Utilization
GPU Memory
```

### Example

```text
Incoming Traffic ↑
       ↓
Queue Depth ↑
       ↓
TTFT ↑
       ↓
Scale-Up Decision
```

### Cover

- reactive metrics
- predictive metrics
- queue-based scaling
- token-based scaling
- SLO-driven scaling

### Key lesson

Scale based on **inference pressure**, not merely generic CPU load.

---

## Day 27 — Pod Autoscaling vs GPU Node Autoscaling

### Goal

Understand the two different scaling loops.

### Layer 1 — Pod scaling

```text
Traffic ↑
   ↓
Autoscaler
   ↓
Inference Replicas
3 → 6
```

### Layer 2 — Node scaling

```text
New GPU Pods Pending
       ↓
No GPU Capacity
       ↓
Node Autoscaler
       ↓
New GPU Node
       ↓
Pod Scheduled
```

### Full flow

```text
Traffic ↑
   ↓
Inference Metric ↑
   ↓
Pod Autoscaler
   ↓
More GPU Pods Requested
   ↓
Insufficient GPU Capacity?
   ├── No → Schedule Pods
   └── Yes
        ↓
   GPU Node Autoscaler
        ↓
   Provision New Node
        ↓
   Load Model
        ↓
   Pod Ready
```

### Cover

- HPA concepts
- custom metrics
- KEDA-style event/queue signals
- cluster/node autoscaling
- GPU scheduling
- startup probes

---

## Day 28 — Cold Starts, Scale-to-Zero, and Predictive Scaling

### Goal

Understand why GPU workloads are hard to scale quickly.

### Cold-start path

```text
Traffic Spike
    ↓
Create Pod
    ↓
Provision GPU Node
    ↓
Pull Container
    ↓
Download Model
    ↓
Load Weights Into GPU
    ↓
Warm Runtime
    ↓
Ready
```

This can be much slower than starting a normal web server.

### Cover

- minimum warm replicas
- model-loading time
- image pulling
- node provisioning
- scale-to-zero
- warm pools
- scheduled scaling
- predictive autoscaling
- hysteresis
- scale-down stabilization

### Core trade-off

```text
More Warm GPUs
    ↓
Lower Latency
Higher Cost

Fewer Warm GPUs
    ↓
Lower Cost
Higher Cold-Start Risk
```

---

# Phase 10 — Production Reliability

## Day 29 — Observability, SLOs, and Graceful Degradation

### Goal

Understand how to determine whether the system is healthy from the user's perspective.

### Core metrics

```text
Request Start
   │
   ├──── TTFT ────→ First Token
   │
   └──────── E2E Latency ───────→ Complete

First Token
    ↓
Token
    ↓ TPOT
Token
```

Track:

- TTFT
- TPOT / ITL
- end-to-end latency
- queue latency
- tokens/sec
- requests/sec
- GPU utilization
- GPU memory
- KV utilization
- cache hit rate
- rejection rate
- error rate

### Example SLOs

```text
p95 TTFT < target
p95 TPOT < target
Availability > target
```

### Overload handling

```text
Traffic ↑
   ↓
SLO Risk
   ↓
Admission Control
   ├── Queue
   ├── Reject
   ├── Reduce Max Output
   └── Route to Fallback Model
```

### Production Cluster Fault Tolerance & High Availability

Inference clusters fail differently than traditional web apps due to stateful KV caches, multi-GPU NCCL bindings, and expensive GPU hardware failures.

```text
Cluster Failure Signal
   ↓
Automated Detection (XID / Watchdog / Readiness)
   ↓
Isolation & Fast Eviction (Remove Pod from Endpoint Picker)
   ↓
Inflight Failover / Graceful Degradation
```

Cover:

- **Inflight Request Resiliency**:
  - mid-generation client disconnections: immediately canceling decode steps to reclaim allocated KV cache
  - transparent prefill retries: automatically re-routing prefill onto alternate healthy replicas when a worker pod dies mid-request

- **Cluster & Kubernetes Resilience**:
  - Pod Disruption Budgets (PDBs) for GPU nodes: preventing K8s node drains during long-running generation phases
  - zero-downtime rolling updates: warm-loading 100GB+ checkpoints on new pods before updating Gateway routing endpoints
  - Multi-AZ & Cross-Region Failover: high availability strategies when facing localized GPU shortages or cloud availability zone failures

---

# Phase 11 — Benchmarking and Capacity Planning

## Day 30 — Benchmark the Complete Production System

### Goal

Answer the most important production question:

> How much traffic can this deployment safely handle while still meeting its SLOs?

### Benchmark matrix

```text
Concurrency
1
2
4
8
16
32
64
128
```

Measure at every level:

- TTFT
- TPOT
- throughput
- tokens/sec
- requests/sec
- queue latency
- error rate
- rejection rate
- GPU memory
- GPU utilization
- KV utilization
- prefix-cache hit rate

### Plot

```text
Concurrency
    ↓
Throughput ────────╮
                   │
Latency      ──────╯ begins rising
                   ↓
Saturation Point
                   ↓
Overload
```

### Capacity planning

Determine:

```text
Requests / second per GPU

Tokens / second per GPU

Concurrent requests per GPU

Safe KV capacity

Number of replicas required

Number of GPU nodes required

Cost per 1M tokens
```

### Cost modeling

```text
GPU-hours per request type (short prompt vs long prompt)

Spot vs reserved vs on-demand GPU pricing

Quantization impact on GPU count and total cost

Right-sizing GPU selection for workload (A100 vs H100 vs L40S)

Batch size optimization: throughput vs cost per token

Multi-model serving: running multiple small models per GPU

Cache hit rate impact on cost (prefix reuse avoids recomputation)
```

### Final architecture

```text
                            Users
                              ↓
                         Load Balancer
                              ↓
                            Gateway
                              ↓
                       Admission Control
                    RUN | QUEUE | REJECT
                              ↓
                 Cache + Load-Aware Router
                              ↓
                       Endpoint Selection
                              ↓
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
         GPU Pod A        GPU Pod B        GPU Pod C
         vLLM/SGLang      vLLM/SGLang      vLLM/SGLang
             │                │                │
             └────────── Distributed KV ──────┘
                              │
                         KV Offloading
                              │
                    Metrics / Observability
                              │
           TTFT | TPOT | Queue | KV | GPU | SLO
                              │
                              ↓
                         Autoscaling
                    ┌─────────┴─────────┐
                    ↓                   ↓
                Pod Scaling        GPU Node Scaling
```

### Final request flow

```text
User Request
    ↓
Gateway
    ↓
Admission Control
    ↓
Token / Work Estimation
    ↓
Prefix Detection
    ↓
Cache-Aware + Load-Aware Routing
    ↓
Selected Worker
    ↓
Scheduler
    ↓
Continuous Batch
    ↓
Prefill
    ↓
KV Allocation / Reuse
    ↓
Decode
    ↓
Streaming Response
    ↓
Metrics
    ↓
Autoscaling + Capacity Decisions
```

---

# Complete Series Map

| Day | Topic |
|---|---|
| 0 | [Prerequisites & Roadmap](Day0.md) |
| 1 | LLM Request Lifecycle, Tokenization, Streaming & VLM Ingestion |
| 2 | Prefill vs Decode & GPU Fundamentals for Inference |
| 3 | KV Cache, Attention Variants (MHA/MQA/GQA/MLA) & Model Quantization |
| 4 | Inside vLLM and SGLang (RadixTree & CUDA Graphs) |
| 5 | PagedAttention, FlashAttention, and FlashDecoding |
| 6 | Continuous Batching |
| 7 | Inference Scheduler & Guided Generation Overhead |
| 8 | Chunked Prefill |
| 9 | Admission Control, Guardrails & Multi-Turn/Agentic Patterns |
| 10 | Backpressure and Overload Protection |
| 11 | Production Deployment, Model Loading & Security |
| 12 | Multi-Pod Replication & Multi-LoRA Serving |
| 13 | Why Round-Robin Fails for LLMs |
| 14 | Prefix-Cache-Aware Routing |
| 15 | Load-Aware Routing |
| 16 | Gateway, InferencePool, and Endpoint Picker |
| 17 | Inference Router Data Plane |
| 18 | Tensor Parallelism |
| 19 | TP, PP, DP, and EP (MoE Architecture Details) |
| 20 | Multi-Node Inference & Spot Resiliency |
| 21 | Disaggregation Topologies: EPD, P/D, E/PD, and E/P/D |
| 22 | KV & Multimodal Embedding Transfer (EC_Connector, Nixl, Mooncake) |
| 23 | Prefix Caching & Multi-Turn Reuse |
| 24 | KV Cache Affinity, Quantization, and Offloading |
| 25 | Speculative Decoding |
| 26 | Autoscaling Signals |
| 27 | Pod Autoscaling and GPU Node Autoscaling |
| 28 | Cold Starts, Scale-to-Zero, and Predictive Scaling |
| 29 | Observability, SLOs & Cluster Fault Tolerance (GPU/NCCL Failures & HA) |
| 30 | Benchmarking, Capacity Planning, Cost Modeling & Final Architecture |

---

# Series Learning Progression

```text
Days 1–5
Understand what happens inside one inference request
        ↓
Days 6–10
Learn how one GPU handles concurrent traffic
        ↓
Days 11–17
Scale from one GPU pod to an intelligently routed replica fleet
        ↓
Days 18–22
Scale inference across GPUs, nodes, and P/D worker pools
        ↓
Days 23–25
Treat KV cache and decoding optimization as infrastructure problems
        ↓
Days 26–28
Autoscale GPU workloads without destroying latency or cost efficiency
        ↓
Day 29
Operate the system using observability, SLOs, and fault tolerance
        ↓
Day 30
Benchmark the whole system and calculate real production capacity
```

---

# Core Story of the Series

The series should never feel like 30 disconnected topics.

Every day introduces a production problem that creates the need for the next concept.

```text
One request is easy.

Many requests create concurrency problems.
    ↓
Concurrency requires scheduling.
    ↓
Scheduling is limited by KV memory.
    ↓
One GPU eventually reaches capacity.
    ↓
We add replicas.
    ↓
Replicas create routing problems.
    ↓
Routing must understand KV locality and load.
    ↓
Large models require multiple GPUs and nodes.
    ↓
Prefill and decode create different scaling pressures.
    ↓
KV becomes distributed state.
    ↓
Traffic changes over time.
    ↓
We need pod and GPU-node autoscaling.
    ↓
Autoscaling introduces cold starts and SLO risks.
    ↓
We need observability and overload protection.
    ↓
Finally, benchmarking tells us the safe capacity and cost of the system.
```

This gives the 30-day series one clear narrative:

> **How do we turn one model running on one GPU into a reliable, scalable, cost-efficient production inference platform?**
