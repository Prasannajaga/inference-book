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

# Phase 6 — Prefill/Decode Disaggregation

## Day 21 — Why Disaggregate Prefill and Decode?

### Goal

Understand why separate worker pools can make sense.

```text
Incoming Request
      ↓
Prefill Pool
      ↓
KV Transfer
      ↓
Decode Pool
      ↓
Generated Tokens
```

### Cover

- different compute characteristics
- independent scaling
- long-prefill interference
- TTFT
- TPOT
- resource specialization

---

## Day 22 — KV Transfer in P/D Disaggregation

### Goal

Understand the infrastructure cost introduced by separating prefill and decode.

### Flow

```text
Prefill Worker
      ↓
Create KV
      ↓
Transfer KV
      ↓
Decode Worker
      ↓
Continue Generation
```

### Cover

- KV transfer latency
- GPU-to-GPU transfer
- host staging
- RDMA concepts
- networking requirements
- transfer overlap
- scheduling coordination

### Key question

> When does the benefit of separating prefill and decode outweigh the cost of moving KV state?

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

### Fault tolerance

Cover:

- pod failure
- GPU failure
- node failure
- health checks
- endpoint removal
- retries
- retry storms
- fallback models
- graceful degradation

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

# Complete 30-Day Map

| Day | Topic |
|---|---|
| 1 | LLM Request Lifecycle |
| 2 | Prefill vs Decode |
| 3 | KV Cache Fundamentals & Quantization (FP8/INT8) |
| 4 | Inside vLLM and SGLang (RadixTree & CUDA Graphs) |
| 5 | PagedAttention, FlashAttention, and FlashDecoding |
| 6 | Continuous Batching |
| 7 | Inference Scheduler & Guided Generation Overhead |
| 8 | Chunked Prefill |
| 9 | Admission Control & Guardrails |
| 10 | Backpressure and Overload Protection |
| 11 | Production Model Deployment |
| 12 | Multi-Pod Replication & Multi-LoRA Serving |
| 13 | Why Round-Robin Fails for LLMs |
| 14 | Prefix-Cache-Aware Routing |
| 15 | Load-Aware Routing |
| 16 | Gateway, InferencePool, and Endpoint Picker |
| 17 | Inference Router Data Plane |
| 18 | Tensor Parallelism |
| 19 | TP, PP, DP, and EP (MoE Architectures) |
| 20 | Multi-Node Inference & Spot Resiliency |
| 21 | P/D Disaggregation |
| 22 | KV Transfer Between Prefill and Decode |
| 23 | Prefix Caching |
| 24 | KV Cache Affinity, Quantization, and Offloading |
| 25 | Speculative Decoding |
| 26 | Autoscaling Signals |
| 27 | Pod Autoscaling and GPU Node Autoscaling |
| 28 | Cold Starts, Scale-to-Zero, and Predictive Scaling |
| 29 | Observability, SLOs, Fault Tolerance, and Graceful Degradation |
| 30 | Benchmarking, Capacity Planning, and Final Production Architecture |

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
