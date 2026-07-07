# Skill: Write CUDA GEMM Kernel

## Purpose
Guide the agent through designing and implementing a correct, performance-aware CUDA GEMM kernel (C = alpha * A * B + beta * C) for a specific problem configuration, including decisions about tiling strategy, memory hierarchy usage, tensor core eligibility, and when to defer to cuBLAS or CUTLASS instead.

## Use this when
- You need a custom GEMM or GEMM-like operation that cuBLAS does not support (fused epilogue, custom accumulation, non-standard layouts, sparse masks)
- You are implementing a batched GEMM variant with irregular batch structure
- You need to fuse the GEMM with a downstream operation (bias add, activation, quantization) and cannot tolerate the memory round-trip of a separate kernel
- You are on a constrained embedded or inference target where you control the tiling strategy precisely
- You are building a learning or research kernel and need explicit control over every memory access

## Do not use this when
- Standard sgemm/hgemm/dgemm without custom epilogue: use cuBLAS (`cublasGemmEx`) — it will outperform any first-attempt custom kernel on all shipping hardware
- GEMM with tensor core acceleration and standard epilogues: use CUTLASS — it exposes MMA-level tiling with a composable epilogue framework that is already highly tuned
- Batch GEMM with fixed batch sizes and standard shapes: use `cublasGemmStridedBatchedEx`
- FP8 GEMM on Hopper: use `cublasLtMatmul` with FP8 descriptors or CUTLASS 3.x FP8 kernels
- The problem is memory-bound rather than compute-bound (small K relative to M and N): tiling will not help, and a simpler kernel may be better

## Inputs the agent should gather first
- **M, N, K**: exact or typical sizes; whether they are statically known or runtime-dynamic
- **dtypes**: A dtype, B dtype, accumulator dtype, output C dtype (e.g., A=fp16, B=fp16, acc=fp32, C=fp16)
- **Layout of A and B**: row-major or column-major; leading dimension stride if non-standard
- **Transpose flags**: transA, transB (determines which dimension is the inner product dimension)
- **Hardware target**: SM architecture (SM70/Volta, SM75/Turing, SM80/Ampere, SM89/Ada, SM90/Hopper) — determines warp MMA availability, shared memory capacity, async copy support
- **Precision requirements**: is fp32 accumulation required for fp16 inputs, or is fp16 accumulation acceptable
- **Epilogue requirements**: is there an alpha/beta scaling, bias addition, activation function, quantization step, or other per-element operation to fuse
- **Batch dimension**: is this a single GEMM or batched; are batch strides uniform
- **Tensor core eligibility**: does the problem shape and dtype satisfy alignment requirements (K divisible by 8 for fp16 wmma, 16 for MMA with specific fragment sizes)

## Required reasoning process
1. **Decide whether to write a custom kernel at all.** If cuBLAS or CUTLASS covers the use case, state that explicitly and stop. Only proceed if there is a concrete reason a custom kernel is necessary.

2. **Determine tensor core eligibility.** For SM70+: WMMA API requires M/N/K tiles of 16x16x16 (fp16) or 8x32x16 / 32x8x16 variants. For SM80+: PTX MMA instructions offer more shapes. Requirements: fp16 or bf16 inputs, fp32 accumulator, K dimension divisible by the fragment K size, A and B pointers 16-byte aligned. If these are not met, fall back to scalar (SIMT) accumulation and document why.

3. **Choose the tiling hierarchy.** A standard 3-level tiling strategy:
   - **Thread block tile** (BM x BN x BK): typically 128x128x8 or 64x64x16 or 128x256x32 depending on target occupancy and register budget. Larger tiles = more reuse but fewer active blocks.
   - **Warp tile** (WM x WN): subdivides the thread block tile among warps. Each warp handles a contiguous WM x WN sub-tile.
   - **Thread tile / register tile**: each thread accumulates a small sub-matrix in registers (e.g., 8x8 for SIMT, or one MMA fragment pair for tensor cores).
   Justify the chosen tile sizes given smem capacity and register pressure.

4. **Design the shared memory layout.** Load A tile as BM x BK and B tile as BK x BN into shared memory. Determine whether row-major or column-major smem layout reduces bank conflicts for the warp access pattern. Add padding columns/rows to eliminate bank conflicts if needed (typically +1 or +2 columns).

5. **Plan the load pattern.** Each thread must participate in loading the A and B tiles collaboratively. Assign contiguous global memory addresses to contiguous threads to ensure coalesced loads. Use vectorized loads (float4, uint4, int4) when alignment and size allow. On SM80+, use `cp.async` (or `cuda::memcpy_async`) to overlap global loads with computation via double buffering.

6. **Design double buffering.** Allocate two smem buffers for A and two for B. While computing on buffer[ping], issue async loads for buffer[pong]. This hides global memory latency behind arithmetic. Requires `cp.async.wait_group` or `__pipeline_commit` / `__pipeline_wait_prior` fence placement to be correct.

7. **Write the inner loop.** The K-loop iterates over BK-wide slices. Each iteration: load tile into smem, sync, compute warp-level MMA or thread-level outer product, sync before next load. If double buffering: restructure so loads and computes are interleaved without extra syncs blocking overlap.

8. **Handle boundary conditions.** For M % BM != 0, N % BN != 0, K % BK != 0: guard loads with bounds checks. Either pad tensors to multiples of tile size externally, or predicate the load/store with per-thread index checks. Predicated loads should return 0 for out-of-bounds positions so partial tiles accumulate correctly.

9. **Write the epilogue.** Apply alpha scaling to the accumulator, load the C tile, apply beta scaling, add, then store. If beta == 0, skip the C load entirely. If fusing a bias or activation, apply it here in registers before the final store. Use vectorized stores when alignment allows.

10. **Estimate register pressure.** The accumulator array dominates. For a thread tile of TM x TN with fp32 accumulators, that is TM*TN registers. Add registers for A fragments and B fragments in the inner loop. Total must stay under the per-thread register limit (255 on current hardware). If over, reduce tile size or annotate `__launch_bounds__`.

## Kernel design rules
- Always accumulate in fp32 when inputs are fp16 or bf16 unless the application explicitly accepts fp16 accumulation and you have verified it does not degrade accuracy for the problem.
- Shared memory layout must be chosen to avoid 4-way or higher bank conflicts for the inner loop access pattern. Verify the warp access pattern against smem bank indices before finalizing.
- Thread block dimensions must be a multiple of 32 (warp size). Common choices: (128,1), (256,1), or (16,16). For tensor core kernels, warp-level tile assignment must align with MMA fragment shapes.
- Do not use `__syncthreads()` inside the inner compute loop if it is not needed. Each unnecessary sync stalls all warps in the block.
- Use `__ldg()` or `__restrict__` on read-only pointers to hint L1 cache for non-tiled accesses.
- The epilogue must handle alpha == 0 (output is all zeros) and beta == 0 (C input is unused) as fast paths to avoid unnecessary loads.
- For WMMA kernels: use `wmma::load_matrix_sync`, `wmma::mma_sync`, `wmma::store_matrix_sync` with correct layout specifiers (`WMMA_ROW_MAJOR` vs `WMMA_COL_MAJOR`). Fragment layouts must match the smem layout used.
- For PTX MMA kernels (SM80+): use `mma.sync.aligned.m16n8k16` (or appropriate variant) only when the surrounding tile structure ensures correct data placement. Errors in fragment lane mapping are silent and produce wrong results.

## Correctness requirements
- **Indexing**: every global A[row][col] access must correctly account for transA, leading dimension (lda), and the thread's position in the tile hierarchy. Verify the formula: `A[blockRow*BM + warpRow*WM + threadRow][k_base + k_offset]` maps to the correct flat index `(blockRow*BM + warpRow*WM + threadRow) * lda + (k_base + k_offset)`.
- **Boundary guards**: threads that map outside [0, M) x [0, K) for A or [0, K) x [0, N) for B must not issue out-of-bounds loads. Predicate these with explicit if-checks or masked vector loads.
- **Shared memory synchronization**: every smem tile write (by all threads in the block) must be followed by `__syncthreads()` before any thread reads from that tile. Every read must complete before the next write overwrites the buffer.
- **Accumulator initialization**: all accumulator registers must be initialized to 0 before the K-loop. For WMMA fragments: call `wmma::fill_fragment(acc_frag, 0.0f)`.
- **Alpha/beta application**: apply alpha to the accumulator result, not to partial sums during the K-loop. Apply beta to the loaded C value, not to the accumulator.
- **Dtype conversion**: when loading fp16 into shared memory and accumulating in fp32, the conversion to fp32 must happen before the fused multiply-add, not after.
- **Vectorized load alignment**: float4 loads require 16-byte aligned addresses. Verify that the base pointer plus per-thread offset is always a multiple of 16 bytes before using `float4` loads. Misaligned vector loads are undefined behavior.
- **Output write bounds**: the store to C must also be predicated for threads outside the valid M x N range.

## Performance requirements
- State the theoretical peak FLOP/s for the target SM and dtype (e.g., SM80 A100: ~312 TFLOPS fp16 with tensor cores, ~19.5 TFLOPS fp32 SIMT).
- Estimate arithmetic intensity for the chosen tile size: for a BM x BN x BK tile, AI = 2*BM*BN*BK / (2*(BM*BK + BK*BN) * sizeof(dtype)) in FLOPs/byte. A well-tiled GEMM should be compute-bound for large M, N, K.
- Target minimum 70-80% of cuBLAS throughput as a bar for "good enough to ship." If unable to reach 50%, defer to cuBLAS.
- Occupancy: target at least 50% theoretical occupancy (2+ active blocks per SM, or 4+ active warps per SM beyond the minimum). Compute `(smem_per_block + register_per_block)` consumption and check against SM limits.
- Double buffering should be used for all SM80+ kernels targeting high arithmetic intensity. On SM70 and SM75, double buffering is possible without `cp.async` using register prefetch or manual smem ping-pong.
- Vectorized loads (float4 for fp32, int4 for int8, float2 for fp16 pairs) should be used wherever alignment guarantees hold. This reduces instruction count and improves memory throughput.
- The inner loop (MMA or outer product) must be structured to maximize instruction-level parallelism. Avoid data dependencies between consecutive MMA calls if possible.

## Output format
The final response must include:
1. **Decision rationale**: whether a custom kernel is appropriate, or whether cuBLAS/CUTLASS should be used instead — with explicit reasoning.
2. **Tile configuration table**: BM, BN, BK, WM, WN, TM, TN (or MMA fragment shape), thread block dimensions, grid dimensions, smem size per block.
3. **Memory layout diagram** (ASCII or description): how A and B tiles are laid out in smem, including any padding.
4. **Kernel code**: complete, compilable CUDA kernel with all helper functions (device functions for warp reduction, fragment load, etc.). No pseudocode.
5. **Host launch code**: grid/block dims, smem size, kernel call with all parameters.
6. **Correctness notes**: explicit list of boundary conditions handled and how.
7. **Performance notes**: expected arithmetic intensity, estimated occupancy, known bottlenecks.
8. **Known limitations**: shapes or hardware where this kernel will underperform or is incorrect.

## Common failure modes
- **Shared memory bank conflicts in the inner loop**: accessing smem with a stride equal to 32 (or a multiple) causes 32-way conflicts. Fix: add +1 or +4 padding to the smem tile column dimension.
- **Wrong tile boundary handling**: tiles at the M or N edge that are smaller than BM/BN require predicated loads. Forgetting this causes out-of-bounds reads and incorrect results on non-round shapes.
- **Misaligned `float4` loads**: base pointer + per-thread byte offset is not 16-byte aligned. This causes a misaligned memory access error at runtime. Fix: check alignment statically or fall back to scalar loads for the last partial vector.
- **Register spilling**: tile sizes are too large for the register file. The compiler spills to local memory, causing 20-100x slowdown on accesses. Fix: reduce TM x TN, add `__launch_bounds__`, or check ptxas output for `spill stores/loads`.
- **Incorrect alpha/beta**: beta is applied to the accumulator instead of C, or alpha is not applied at all, or the accumulator is stored before alpha scaling.
- **Missing `__syncthreads` after smem load**: threads in the block race: some threads start reading A/B tiles from smem while other threads are still writing them. Always sync after the collaborative load phase.
- **Wrong transposition handling**: when transA=true, the A tile loads from the transposed dimension. A common bug is using the wrong stride (lda vs K) or swapping row/col indices without updating the smem layout.
- **WMMA layout mismatch**: loading a row-major A fragment with `WMMA_COL_MAJOR` specifier or vice versa produces silently wrong results. Carefully match smem layout to the fragment load specifier.
- **Double buffer fence errors**: `cp.async.wait_group(N)` leaves N async groups in flight. Using N=0 blocks until all are done; using N=1 allows one group to still be in flight. Wrong N causes reading data before it has arrived from global memory.

## Review checklist
- [ ] Have I verified that cuBLAS or CUTLASS cannot cover this use case? If yes, is the reason documented?
- [ ] Are M, N, K boundary conditions handled for non-multiples of the tile size?
- [ ] Are all smem accesses free of bank conflicts, or has padding been added and verified?
- [ ] Is every smem write followed by `__syncthreads()` before the corresponding read?
- [ ] Are float4/float2 loads used only on verified-aligned pointers?
- [ ] Is the accumulator initialized to zero before the K-loop?
- [ ] Is alpha applied to the accumulator and beta applied to the C load, not the other way around?
- [ ] Does the epilogue skip the C load when beta == 0?
- [ ] Are output stores predicated for threads outside the valid [0,M) x [0,N) range?
- [ ] For WMMA kernels: do fragment layout specifiers match the actual smem layout?
- [ ] For double buffering: are `cp.async` commit and wait fences placed correctly?
- [ ] Has ptxas output been checked for register spills?
- [ ] Has the kernel been validated against a reference (cuBLAS or CPU GEMM) on at least: square shapes, rectangular shapes, non-power-of-two shapes, M=1 (vector-matrix), K=1?
- [ ] Is the launch configuration correct (grid covers all M x N outputs, block dims are warp multiples)?