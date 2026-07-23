
## Day 1/30 of inference infrastructure

request lifecycle

the request lifecycle shows you how your query goes through the server infrastructure

the big difference today is theres no simple HTTP response anymore, we use SSE to stream tokens line by line as soon as they generate

we hit an openAI-compatible endpoint /v1/chat/completions, sending the prompt, model configs, temperature, and max_tokens

single turn chats were old news. now we pass multi-turn chat history inside the context window, and every new turn adds more tokens to the prompt filling up GPU memory with KV cache

to handle this without dropping connections or running out of memory, your request passes through 3 layers:

* gateway (envoy)
* cluster router (llm-d)
* multi-pod runtime (vllm/sglang)

here is the high-level architecture showing how a request moves through each layer of the infrastructure:

```mermaid
graph TD
    User["User Request<br/>/v1/chat/completions API"]
    Envoy["Gateway Layer<br/>(Envoy Proxy)"]

    subgraph Cluster["Kubernetes Cluster"]
        LLMD["Cluster Layer: llm-d Router<br/>(Prefix Cache & Load-Aware Routing)"]
        
        subgraph MultiPod["Multi-Pod Inference Runtime Layer"]
            Pod1["Inference Pod 1<br/>vLLM / SGLang"]
            Pod2["Inference Pod 2<br/>vLLM / SGLang"]
            Pod3["Inference Pod N<br/>vLLM / SGLang"]
        end
    end

    User --> Envoy
    Envoy --> LLMD
    LLMD --> Pod1
    LLMD --> Pod2
    LLMD --> Pod3
    Pod1 -.->|"SSE Response Stream"| Envoy
    Envoy -.->|"Stream Response"| User
```

# Part 2

we saw the high level flow, but let's go a bit lower level to see what each layer actually does step-by-step:

1/ Gateway (Envoy): checks API keys, enforces rate limits, and opens a long-lived unbuffered SSE connection. no buffering allowed tokens must flush instantly.

2/ Cluster Router (llm-d): checks prompt hashes for KV cache matches. if Pod A has warm cache for your prompt, it routes to Pod A. otherwise, it picks the pod with the lowest queue depth.

3/ Runtime Pod (vLLM / SGLang): host CPU tokenizes your text with tiktoken, applies chat templates (<|im_start|>), and the scheduler puts your request in the pending queue.

4/ GPU Prefill (TTFT): the GPU processes all prompt tokens at once, populates KV cache blocks in VRAM, and outputs token #1. this first token streams all the way back to your screen.

5/ Autoregressive Decode (TPOT): the GPU generates remaining tokens one by one in a loop, reading KV cache from memory. each new token gets de-tokenized and flushed as an SSE chunk.

6/ Finish & Cleanup: once the model done generating or max_tokens, the engine releases VRAM blocks or CPU/storage offloading and closes the SSE stream.

here is the exact sequence diagram mapping this entire loop:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Gateway as Gateway Layer<br/>(Envoy Proxy)
    participant Cluster as Cluster Layer<br/>(Kubernetes / llm-d)
    participant Runtime as Multi-Pod / Runtime<br/>(vLLM / SGLang Engine)
    participant GPU as Model Execution<br/>(GPU Tensor Cores)

    Client->>Gateway: 1. POST /v1/chat/completions (stream: true)
    Note over Gateway: TLS termination, auth, rate limiting,<br/>SSE long-lived HTTP connection setup
    Gateway->>Cluster: 2. Forward request payload
    Note over Cluster: Pod selection, cache-aware routing,<br/>cluster-level admission control
    Cluster->>Runtime: 3. Dispatch to target pod endpoint
    Note over Runtime: Tokenization (BPE/tiktoken),<br/>Chat template formatting,<br/>Sequence Scheduler queueing
    Runtime->>GPU: 4. Prefill phase GEMM execution
    Note over GPU: Compute-heavy matrix multiplication,<br/>KV cache allocation (PagedAttention),<br/>TTFT calculated
    GPU-->>Runtime: 5. First token generated
    Runtime-->>Gateway: 6. SSE chunk (data: {"delta": {"content": "Hello"}})
    Gateway-->>Client: 7. Stream TTFT chunk to client
    
    loop Autoregressive Decode Loop (TPOT)
        Runtime->>GPU: 8. Decode step GEMV (single token + KV cache read)
        GPU-->>Runtime: 9. Next token generated
        Runtime-->>Gateway: 10. SSE chunk stream
        Gateway-->>Client: 11. Stream chunk to client
    end
    
    Runtime-->>Gateway: 12. [DONE] signal & release KV cache
    Gateway-->>Client: 13. Close SSE connection
```
