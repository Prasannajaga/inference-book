# SGLang-Omni — Architecture, Usage, MPS, and Multimodal Serving Notes

## Overview

This README summarizes the discussion around:

- what SGLang-Omni is;
- how it differs from SGLang;
- what problems it solves;
- how its multi-stage runtime works;
- the four Basic Usage sections;
- why TTS serving behaves differently from text-only LLM serving;
- why GPUs can stay idle during TTS inference;
- how same-GPU data parallelism with CUDA MPS helps;
- the main bottlenecks in multimodal and speech inference.

---

# 1. What is SGLang-Omni?

SGLang-Omni is a **multi-stage serving runtime for omni, speech, TTS, ASR, and multimodal models**.

It is designed for models where generation is not just:

```text
prompt
  ↓
transformer
  ↓
tokens
```

Instead, the request may pass through several heterogeneous stages:

```text
text / image / audio / video
          ↓
     preprocessing
          ↓
        encoders
          ↓
        thinker
          ↓
        talker
          ↓
      audio codes
          ↓
       vocoder
          ↓
         audio
```

SGLang-Omni owns:

- pipeline topology;
- stage lifecycle;
- stage-to-stage communication;
- streaming;
- multimodal request handling;
- model-family integration;
- OpenAI-compatible serving APIs.

It reuses **SGLang** for high-performance autoregressive execution where appropriate.

---

# 2. SGLang vs SGLang-Omni

The cleanest distinction is:

```text
SGLang:
Optimize execution inside an autoregressive model.

SGLang-Omni:
Coordinate execution across multiple heterogeneous model stages.
```

## SGLang

Main responsibilities:

- continuous batching;
- prefill/decode scheduling;
- KV-cache management;
- RadixAttention;
- prefix caching;
- CUDA graphs;
- model execution;
- optimized kernels;
- tensor/expert parallelism;
- speculative decoding.

Typical flow:

```text
Prompt
  ↓
Prefill
  ↓
Decode
  ↓
Decode
  ↓
Decode
  ↓
Text
```

## SGLang-Omni

Main responsibilities:

- multi-stage pipeline orchestration;
- thinker/talker/vocoder coordination;
- different scheduler per stage;
- multimodal transport;
- streaming speech;
- stage placement across GPUs/processes/nodes;
- OpenAI-compatible speech, transcription, and omni APIs.

Typical flow:

```text
audio + image + text
        ↓
preprocess
        ↓
encoders
        ↓
thinker
        ↓
talker
        ↓
vocoder
        ↓
text + speech
```

SGLang-Omni is therefore **not replacing SGLang**.

It builds around it.

---

# 3. Why SGLang-Omni is needed

A single LLM scheduler is not enough for omni models.

Different stages have very different workloads.

For example:

```text
Thinker
- autoregressive transformer
- KV-cache heavy
- batching is useful

Talker
- autoregressive speech token generation
- different decode characteristics

Encoder
- one-shot execution
- no KV cache

Vocoder
- streaming
- chunk-based execution
- latency sensitive

Preprocessing
- mostly CPU-side
```

Trying to force all of these through one scheduling policy would be inefficient.

SGLang-Omni instead gives each stage an execution model that matches its workload.

---

# 4. High-level architecture

The repository describes the runtime approximately as:

```text
HTTP API
   ↓
Client
   ↓
Coordinator
   ↓
Stage
   ↓
Scheduler
   ↓
ModelRunner
   ↓
Model Forward
```

---

# 5. Coordinator

The Coordinator manages the global request lifecycle.

Responsibilities:

```text
new request
    ↓
send to entry stage
    ↓
track request state
    ↓
collect intermediate/terminal results
    ↓
merge outputs
    ↓
return to client
```

It handles:

- pending requests;
- running requests;
- completion;
- failures;
- cancellation;
- streaming results;
- pipelines with multiple terminal outputs.

It does not need to understand the internals of each stage.

---

# 6. Stage

A Stage is mainly an **I/O and communication shell**.

Examples:

```text
preprocessor
vision_encoder
audio_encoder
thinker
talker
vocoder
aggregator
```

A stage is responsible for:

- receiving control messages;
- reading payloads;
- handling fan-in;
- forwarding work to its scheduler;
- sending outputs to downstream stages;
- sending streaming chunks;
- propagating errors.

Useful mental model:

```text
Stage
  = communication + routing

Scheduler
  = execution policy

ModelRunner
  = model forward implementation
```

---

# 7. Scheduler types

## OmniScheduler

Used for autoregressive stages.

It reuses SGLang's:

- batch selection;
- KV-cache management;
- prefill/decode scheduling;
- prefix cache;
- overlap scheduling.

Typical stages:

```text
thinker
talker
```

---

## SimpleScheduler

Used for non-autoregressive stages.

Typical stages:

```text
preprocessing
encoders
aggregation
simple decode
```

Conceptually:

```text
inbox.get()
   ↓
compute()
   ↓
outbox.put()
```

No KV cache.

No autoregressive batching.

---

## Code2WavScheduler

Used for streaming vocoder-style workloads.

Flow:

```text
new request
   ↓
initialize state
   ↓
receive audio-code chunk
   ↓
decode chunk
   ↓
emit waveform chunk
   ↓
repeat
   ↓
flush final audio
```

---

# 8. ModelRunner

The ModelRunner performs autoregressive model execution.

Conceptually:

```text
ForwardBatch
   ↓
prepare model-specific inputs
   ↓
model forward
   ↓
post-process outputs
   ↓
scheduler result
```

For a multimodal thinker, it may inject:

```text
image embeddings
audio embeddings
video embeddings
multimodal features
```

before the transformer forward pass.

---

# 9. Communication architecture

SGLang-Omni separates communication into:

```text
Control Plane
+
Data Plane
```

## Control Plane

Uses ZMQ for small messages such as:

```text
SubmitMessage
DataReadyMessage
CompleteMessage
StreamMessage
AbortMessage
ShutdownMessage
```

---

## Data Plane

Moves actual tensors and payloads.

Possible mechanisms:

```text
same process
    → Python object reference

same GPU/node
    → CUDA IPC

same node CPU
    → shared memory

cross node
    → Mooncake / other relay backend
```

This avoids putting large tensors inside ordinary control messages.

---

# 10. Basic Usage — the four main pages

The SGLang-Omni Basic Usage section covers four areas:

1. Omni Model Usage
2. TTS Model Usage
3. Omni Router Usage
4. Same-GPU Data Parallelism with CUDA MPS

---

# 11. Omni Model Usage

This section explains how to serve a full multimodal model such as Qwen3-Omni.

Possible inputs:

```text
text
image
audio
video
```

Possible outputs:

```text
text
audio
text + audio
```

## Text-only mode

Simplified path:

```text
image / audio / video / text
            ↓
          thinker
            ↓
           text
```

Example:

```text
Input:
car.jpg
"How many cars are visible?"

        ↓

Qwen3-Omni Thinker

        ↓

"There are four cars."
```

Use this mode when speech output is not needed.

---

# 12. Full speech mode

Speech mode uses a larger multi-stage pipeline.

Conceptually:

```text
multimodal input
      ↓
preprocessing
      ↓
encoders
      ↓
thinker
      ↓
talker
      ↓
audio code prediction
      ↓
code-to-wave
      ↓
streaming speech
```

Stages can be placed on different GPUs.

Example:

```text
GPU 0
└── Thinker

GPU 1
├── Talker
├── Code Predictor
└── Vocoder
```

This is one of the main reasons SGLang-Omni exists.

---

# 13. TTS Model Usage

The TTS section focuses on:

```text
text
  ↓
speech model
  ↓
audio
```

Typical API:

```text
POST /v1/audio/speech
```

Possible use cases:

- normal TTS;
- voice cloning;
- voice design;
- streaming speech;
- voice agents.

Example:

```text
"Welcome to my inference engineering series."

        ↓

TTS model

        ↓

output.wav
```

---

# 14. Voice cloning

Input:

```text
reference audio
+
reference transcript
+
new text
```

Example:

```text
Reference audio:
"I'm testing my inference system."

New text:
"Welcome to today's episode."

          ↓

TTS

          ↓

speech resembling the reference voice
```

---

# 15. Streaming TTS

Instead of waiting for the whole sentence:

```text
generate full audio
      ↓
return result
```

streaming works like:

```text
generate first chunk
      ↓
play

generate second chunk
      ↓
play

generate next chunk
      ↓
play
```

Main benefit:

```text
lower Time-To-First-Audio
```

This is critical for voice assistants.

---

# 16. Omni Router

The Router sits in front of multiple SGLang-Omni workers.

```text
                  ┌── Worker A
Client → Router ──┤
                  ├── Worker B
                  └── Worker C
```

The client sees one endpoint.

The router decides which complete worker handles each request.

It does **not** split one request across workers.

---

# 17. Router policies

Typical policies include:

```text
round_robin
least_request
random
```

Example:

```text
Worker A → 8 requests
Worker B → 2 requests
Worker C → 5 requests
```

With least-request routing:

```text
next request → Worker B
```

---

# 18. Health and capability-aware routing

The router can avoid unhealthy workers.

Example:

```text
Worker A → healthy
Worker B → crashed
Worker C → healthy
```

Traffic only goes to A and C.

It can also route based on capabilities.

Example:

```text
Worker A
- chat
- image
- video

Worker B
- chat
- TTS
- audio
```

Request:

```text
POST /v1/audio/speech
```

should go only to Worker B.

---

# 19. Same-GPU Data Parallelism with CUDA MPS

This section solves a surprising problem:

> A TTS server can be fully saturated with requests while the GPU is still mostly idle.

In the documented Higgs TTS profiling, the single replica could plateau while GPU SM activity was only around 29%.

That means:

```text
software:
SATURATED

GPU:
UNDERUTILIZED
```

This is very different from a large text LLM.

---

# 20. Text-only LLM execution

A normal LLM loop is roughly:

```text
request
   ↓
tokenizer
   ↓
prefill
   ↓
decode
   ↓
decode
   ↓
decode
   ↓
text
```

The GPU sees almost continuous transformer work:

```text
██████████████████████████████
```

Main bottlenecks:

- HBM bandwidth;
- KV cache;
- attention;
- GEMM compute;
- batching;
- scheduling.

---

# 21. TTS execution is more fragmented

TTS can look like:

```text
HTTP request
    ↓
CPU preprocessing
    ↓
GPU autoregressive model
    ↓
CPU sampling
    ↓
GPU autoregressive model
    ↓
CPU chunk handling
    ↓
GPU vocoder
    ↓
CPU PCM assembly
    ↓
HTTP streaming
```

GPU timeline:

```text
█████░░░████░░░███░░░████
```

The gaps are periods where the GPU has nothing ready to run.

---

# 22. Why TTS GPUs can become idle

There are two major reasons.

## 1. Latency-capped batch sizes

For maximum throughput:

```text
larger batch
   ↓
better GPU utilization
```

But for streaming speech:

```text
larger batch
   ↓
higher waiting time
   ↓
worse Time-To-First-Audio
```

So TTS serving cannot always increase batch size enough to saturate the GPU.

This creates small kernels and low occupancy.

---

## 2. Host-side work between GPU operations

TTS has more CPU/host work than a simple text decode loop.

Examples:

- sampling;
- vocoder scheduling;
- audio chunk management;
- PCM assembly;
- HTTP streaming;
- preprocessing;
- stage coordination.

Illustrative sequence:

```text
GPU decode
  ↓
CPU sampling
  ↓
GPU decode
  ↓
CPU bookkeeping
  ↓
GPU vocoder
  ↓
CPU streaming
```

Those CPU intervals create GPU bubbles.

---

# 23. Dispatch bubbles

Consider:

```text
GPU AR         1.0 ms
CPU sampling   0.5 ms
CPU routing    0.3 ms
GPU AR         1.0 ms
CPU vocoder scheduling
GPU vocoder
CPU streaming
```

GPU timeline:

```text
████░░░████░░░███░░░████
```

These idle spaces are dispatch bubbles.

The GPU is not necessarily slow.

The serving pipeline is failing to keep it continuously fed with work.

---

# 24. Why continuous batching alone cannot fix this

Continuous batching works extremely well when most work is inside one transformer loop:

```text
A ─┐
B ─┤
C ─┼── one decode batch
D ─┤
E ─┘
```

But a TTS pipeline includes stages outside that loop:

```text
AR transformer
    ↓
audio token sampling
    ↓
chunk generation
    ↓
vocoder
    ↓
PCM assembly
    ↓
network streaming
```

Those stages have different scheduling requirements.

That is why SGLang-Omni uses multiple scheduler types.

---

# 25. Pipeline bubbles

Different stages may run at different speeds.

Example:

```text
Thinker = 5 ms
Talker  = 12 ms
Vocoder = 3 ms
```

The Talker becomes the bottleneck.

Timeline:

```text
Thinker:
█████            █████

Talker:
     ████████████     ████████████

Vocoder:
                 ███              ███
```

Even with powerful GPUs, one slow stage can limit the whole pipeline.

---

# 26. How same-GPU DP helps

Suppose one replica behaves like this:

```text
Replica A:
██████       █████       ████
      idle        idle
```

Now run another independent replica:

```text
Replica A:
██████       █████       ████

Replica B:
      ██████      ██████      ████
```

Combined GPU timeline:

```text
██████████████████████████████
```

When Replica A is waiting on CPU work, Replica B can use the GPU.

---

# 27. Why CUDA MPS matters

Without MPS, multiple CUDA processes have separate contexts and are often time-sliced.

Conceptually:

```text
Process A
   ↓
CUDA Context A

Process B
   ↓
CUDA Context B
```

Execution may look like:

```text
AAAAAA
      BBBBBB
            AAAAAA
```

With CUDA MPS, kernels from multiple processes can execute concurrently when GPU resources allow.

Conceptually:

```text
Replica A kernels
Replica B kernels
Replica C kernels
        ↓
same GPU
```

Aggregate timeline:

```text
██████████████████████████████
```

---

# 28. Important point: MPS improves throughput, not necessarily single-request latency

The purpose is:

```text
more requests / second
```

not:

```text
make one request dramatically faster
```

It improves utilization by overlapping independent replicas.

---

# 29. MPS example

Suppose:

```text
Concurrency = 128
Queued requests = many
Throughput = 22 QPS
SM Active = 29%
```

Increasing concurrency:

```text
128 → 256
```

still gives:

```text
~22 QPS
```

The serving engine has plateaued.

Instead:

```text
                 H100
                  │
              CUDA MPS
        ┌─────────┼─────────┐
        ↓         ↓         ↓
     TTS-0      TTS-1      TTS-2
```

Each replica has its own:

- scheduler;
- CPU execution path;
- KV pool;
- pipeline state.

Their GPU work can overlap.

---

# 30. Example Higgs TTS throughput

Observed ranges discussed:

```text
Single replica
≈ 21.7–22.1 QPS

DP2 + MPS
≈ 31.5–37.7 QPS

DP3 + MPS
≈ 39.9–46.9 QPS
```

Approximate improvements:

```text
DP2
≈ 1.4–1.7×

DP3
≈ 1.8–2.1×
```

These are workload-specific results, not universal guarantees.

---

# 31. Why the same trick often does not help large text LLMs

A large LLM can already fill the GPU through batching.

Example:

```text
batch 1
→ low utilization

batch 32
→ higher utilization

batch 128
→ ~90–100% utilization
```

Once the GPU is already full:

```text
Replica A
████████████████████████████
```

adding Replica B creates contention for:

- SMs;
- HBM bandwidth;
- cache;
- VRAM.

There is little idle capacity left to recover.

---

# 32. Memory cost of same-GPU DP

Without weight sharing:

```text
Replica A
├── weights
├── KV cache
└── runtime buffers

Replica B
├── duplicate weights
├── KV cache
└── runtime buffers

Replica C
├── duplicate weights
├── KV cache
└── runtime buffers
```

Therefore MPS DP trades:

```text
more VRAM
    ↓
more independent execution streams
    ↓
higher GPU utilization
```

Some architectures can experiment with CUDA-IPC weight sharing:

```text
          shared model weights
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
     Replica A Replica B Replica C
        KV A      KV B      KV C
```

But this is not universally supported.

---

# 33. Text-only vs TTS processor

## Text-only

```text
HTTP
 ↓
tokenizer
 ↓
scheduler
 ↓
prefill
 ↓
KV cache
 ↓
decode
 ↓
sampling
 ↓
text
```

Main problems:

```text
compute
HBM bandwidth
KV cache
attention
batching
```

---

## TTS

```text
HTTP
 ↓
text preprocessing
 ↓
reference audio processing
 ↓
AR speech model
 ↓
audio code sampling
 ↓
chunk accumulation
 ↓
vocoder
 ↓
PCM assembly
 ↓
streaming
```

Main problems:

```text
GPU compute
+
CPU orchestration
+
multiple GPU stages
+
streaming
+
pipeline dependencies
+
small latency-sensitive batches
```

---

# 34. Main multimodal serving bottlenecks

## 1. Heterogeneous stages

Different stages behave differently:

```text
thinker
talker
encoder
vocoder
preprocessor
```

There is no single perfect scheduling strategy.

---

## 2. Stage imbalance

One slow stage can limit the pipeline.

Example:

```text
Thinker: 5 ms
Talker: 12 ms
Vocoder: 3 ms

bottleneck = Talker
```

---

## 3. Host dispatch bottlenecks

CPU scheduling and preprocessing can prevent the GPU from receiving work fast enough.

This can produce high GPU idle time even when requests are queued.

---

## 4. Small latency-sensitive kernels

Streaming often prefers:

```text
small batch
small chunk
run immediately
```

instead of:

```text
large batch
wait longer
run efficiently
```

This reduces GPU occupancy.

---

## 5. Stage-to-stage data movement

Omni models may move:

```text
embeddings
hidden states
audio codes
stream chunks
```

between:

- processes;
- GPUs;
- nodes.

Every transition adds:

- synchronization;
- transfer overhead;
- queueing;
- memory-lifetime complexity.

---

## 6. Multiple scheduling domains

A normal LLM scheduler mostly asks:

```text
who prefills?
who decodes?
how much KV is free?
```

An omni runtime also asks:

```text
which thinker runs?
which talker runs?
which audio chunks are ready?
which vocoder stream is ready?
which stage is backpressured?
should TTFA be prioritized?
```

This is a pipeline scheduling problem, not just a token scheduling problem.

---

## 7. Memory management

A multimodal runtime may need memory for:

```text
thinker weights
talker weights
vision encoder
audio encoder
vocoder
codec
KV caches
stream buffers
intermediate tensors
CUDA graphs
```

Adding multiple replicas makes this more difficult.

---

# 35. Practical decision rule for MPS

Do not use MPS simply because the workload is TTS.

First profile the system.

If:

```text
queue is full
throughput has plateaued
GPU SM Active is only 20–50%
latency SLO prevents larger batching
```

then same-GPU DP with MPS may help.

If:

```text
GPU SM Active ≈ 90–100%
```

then MPS will usually provide little benefit and can introduce contention.

The important question is:

```text
Is the bottleneck GPU capacity,
or the system's ability to feed the GPU?
```

---

# 36. Final mental model

## Traditional LLM

```text
Prompt
  ↓
Transformer engine
  ↓
continuous batching
  ↓
GPU stays busy
  ↓
Text
```

Primary optimization goal:

> Keep one transformer engine as efficient as possible.

---

## TTS / Omni

```text
Input
 ↓
CPU preprocess
 ↓
Encoder
 ↓
Thinker
 ↓
Talker
 ↓
Vocoder
 ↓
CPU streaming
 ↓
Output
```

Primary optimization goal:

> Keep multiple heterogeneous stages balanced while continuously feeding the GPU and preserving low streaming latency.

---

# 37. Final takeaway

SGLang and SGLang-Omni solve different layers of the inference problem.

```text
SGLang
    ↓
efficient transformer execution

SGLang-Omni
    ↓
efficient multi-stage model orchestration
```

For standard text inference, the main problems remain:

```text
KV cache
memory bandwidth
batching
attention
compute
```

For multimodal and TTS systems, additional bottlenecks appear:

```text
CPU dispatch
stage imbalance
pipeline bubbles
vocoder scheduling
cross-stage communication
streaming latency
small batch sizes
multiple memory pools
```

Same-GPU DP + CUDA MPS is useful when a serving pipeline is **software-saturated but GPU-underutilized**.

It works by running multiple independent replicas on one GPU so that one replica can use GPU resources while another is blocked on CPU or pipeline work.

The broader lesson is:

> Optimizing multimodal inference is not only about making GPU kernels faster. It is about keeping the entire pipeline — CPU, schedulers, encoders, transformers, vocoders, communication, and streaming — continuously productive.
