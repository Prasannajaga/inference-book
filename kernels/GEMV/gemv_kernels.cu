#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <type_traits>

#define CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        const cublasStatus_t status = (expr);                                  \
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,                           \
                    "cuBLAS call failed with status ", static_cast<int>(status)); \
    } while (0)

namespace {

// Memory map:
// - weight, vector, output: global memory / off-chip DRAM.
// - partial_sum, weight_pair, vector_pair: registers / on-chip per-thread state.
// - warp_sums: shared memory / on-chip per-block scratch.

void check_inputs(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    const torch::Tensor& output) {
    CHECK_CUDA(weight);
    CHECK_CUDA(vector);
    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(vector);
    CHECK_CONTIGUOUS(output);

    TORCH_CHECK(weight.scalar_type() == at::kHalf ||
                weight.scalar_type() == at::kFloat,
                "weight must be float16 or float32");
    TORCH_CHECK(vector.scalar_type() == weight.scalar_type(),
                "vector must have the same dtype as weight");
    TORCH_CHECK(output.scalar_type() == weight.scalar_type(),
                "output must have the same dtype as weight");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [num_rows, num_cols]");
    TORCH_CHECK(vector.dim() == 1, "vector must have shape [num_cols]");
    TORCH_CHECK(output.dim() == 1, "output must have shape [num_rows]");
    TORCH_CHECK(weight.size(1) == vector.size(0),
                "weight.size(1) must equal vector.size(0)");
    TORCH_CHECK(weight.size(0) == output.size(0),
                "output.size(0) must equal weight.size(0)");
    TORCH_CHECK(weight.device() == vector.device() &&
                weight.device() == output.device(),
                "weight, vector, and output must be on the same CUDA device");
    TORCH_CHECK(weight.size(0) <= INT_MAX && weight.size(1) <= INT_MAX,
                "num_rows and num_cols must fit in a 32-bit integer");
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float load_as_float<__half>(__half value) {
    return __half2float(value);
}

template <typename scalar_t>
__device__ __forceinline__ void store_output(
    scalar_t* output, int output_row, float value) {
    // register -> global memory
    output[output_row] = static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ void store_output<__half>(
    __half* output, int output_row, float value) {
    // register -> fp16 -> global memory
    output[output_row] = __float2half_rn(value);
}

template <typename scalar_t>
__global__ void naive_gemv_kernel(
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ vector,
    scalar_t* __restrict__ output,
    int num_rows,
    int num_cols) {
    const int output_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_row >= num_rows) {
        return;
    }

    // Pointer to weight[output_row, :] in global memory.
    const scalar_t* weight_row =
        weight + static_cast<int64_t>(output_row) * num_cols;

    // Per-thread register accumulator.
    float partial_sum = 0.0f;
    for (int col = 0; col < num_cols; ++col) {
        // Global memory -> registers, then FMA in registers.
        partial_sum = fmaf(
            load_as_float(weight_row[col]),
            load_as_float(vector[col]),
            partial_sum);
    }

    // Final output goes back to global memory.
    store_output(output, output_row, partial_sum);
}

__device__ __forceinline__ float warp_sum(float partial_sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Register exchange inside one warp; no global memory.
        partial_sum += __shfl_down_sync(0xffffffffu, partial_sum, offset);
    }
    return partial_sum;
}

__device__ __forceinline__ float strided_dot(
    const __half* weight_row,
    const __half* vector,
    int num_cols,
    int thread_id,
    int thread_count) {
    // Partial dot product held in a register.
    float partial_sum = 0.0f;

    // half2 is safe only when both row and vector pointers are 4-byte aligned
    // and the row has an even number of elements. Otherwise use scalar loads.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(weight_row) & 0x3u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(vector) & 0x3u) == 0u);
    if (aligned && (num_cols % 2 == 0)) {
        // Vectorized global loads: two fp16 values per load.
        const half2* weight_pairs = reinterpret_cast<const half2*>(weight_row);
        const half2* vector_pairs = reinterpret_cast<const half2*>(vector);
        const int num_pairs = num_cols / 2;

        for (int pair_col = thread_id;
             pair_col < num_pairs;
             pair_col += thread_count) {
            // weight and vector: global memory -> registers.
            const float2 weight_pair = __half22float2(weight_pairs[pair_col]);
            const float2 vector_pair = __half22float2(vector_pairs[pair_col]);

            // Register-only compute.
            partial_sum = fmaf(weight_pair.x, vector_pair.x, partial_sum);
            partial_sum = fmaf(weight_pair.y, vector_pair.y, partial_sum);
        }
    } else {
        for (int col = thread_id; col < num_cols; col += thread_count) {
            // Scalar global loads -> register FMA.
            partial_sum = fmaf(
                __half2float(weight_row[col]),
                __half2float(vector[col]),
                partial_sum);
        }
    }

    return partial_sum;
}

__device__ __forceinline__ float strided_dot(
    const float* weight_row,
    const float* vector,
    int num_cols,
    int thread_id,
    int thread_count) {
    // Partial dot product held in a register.
    float partial_sum = 0.0f;

    // float2 is safe only when both pointers are 8-byte aligned and the row
    // has an even number of elements. Otherwise use scalar float loads.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(weight_row) & 0x7u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(vector) & 0x7u) == 0u);
    if (aligned && (num_cols % 2 == 0)) {
        // Vectorized global loads: two fp32 values per load.
        const float2* weight_pairs = reinterpret_cast<const float2*>(weight_row);
        const float2* vector_pairs = reinterpret_cast<const float2*>(vector);
        const int num_pairs = num_cols / 2;

        for (int pair_col = thread_id;
             pair_col < num_pairs;
             pair_col += thread_count) {
            // weight and vector: global memory -> registers.
            const float2 weight_pair = weight_pairs[pair_col];
            const float2 vector_pair = vector_pairs[pair_col];

            // Register-only compute.
            partial_sum = fmaf(weight_pair.x, vector_pair.x, partial_sum);
            partial_sum = fmaf(weight_pair.y, vector_pair.y, partial_sum);
        }
    } else {
        for (int col = thread_id; col < num_cols; col += thread_count) {
            // Scalar global loads -> register FMA.
            partial_sum = fmaf(weight_row[col], vector[col], partial_sum);
        }
    }

    return partial_sum;
}

template <typename scalar_t>
__global__ void optimized_gemv_kernel(
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ vector,
    scalar_t* __restrict__ output,
    int num_rows,
    int num_cols) {
    const int output_row = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 31;
    const int warp_id = thread_id >> 5;
    const int num_warps = blockDim.x >> 5;

    if (output_row >= num_rows) {
        return;
    }

    // One block computes one output row.
    const scalar_t* weight_row =
        weight + static_cast<int64_t>(output_row) * num_cols;

    // Global memory -> registers; each thread handles a slice of columns.
    float partial_sum = strided_dot(
        weight_row,
        vector,
        num_cols,
        thread_id,
        blockDim.x);

    // Reduce registers inside each warp.
    partial_sum = warp_sum(partial_sum);

    // Shared memory stores one partial sum per warp.
    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = partial_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        // Shared memory -> registers, then final warp reduction.
        float row_sum = lane_id < num_warps ? warp_sums[lane_id] : 0.0f;
        row_sum = warp_sum(row_sum);
        if (lane_id == 0) {
            // Final row result: register -> global memory.
            store_output(output, output_row, row_sum);
        }
    }
}

template <typename scalar_t>
__global__ void optimized_gemv_v1_kernel(
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ vector,
    scalar_t* __restrict__ output,
    int num_rows,
    int num_cols) {
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 31;
    const int warp_id = thread_id >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp_id;

    if (output_row >= num_rows) {
        return;
    }

    // V1: one warp computes one row; no shared-memory row reduction.
    const scalar_t* weight_row =
        weight + static_cast<int64_t>(output_row) * num_cols;
    float partial_sum = strided_dot(weight_row, vector, num_cols, lane_id, 32);
    partial_sum = warp_sum(partial_sum);

    if (lane_id == 0) {
        store_output(output, output_row, partial_sum);
    }
}

template <typename scalar_t, int vector_tile_cols>
__global__ void optimized_gemv_v2_kernel(
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ vector,
    scalar_t* __restrict__ output,
    int num_rows,
    int num_cols) {
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 31;
    const int warp_id = thread_id >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp_id;
    const bool valid_row = output_row < num_rows;

    extern __shared__ __align__(16) unsigned char shared_storage[];
    scalar_t* shared_vector = reinterpret_cast<scalar_t*>(shared_storage);

    float partial_sum = 0.0f;
    const scalar_t* weight_row = valid_row
        ? weight + static_cast<int64_t>(output_row) * num_cols
        : nullptr;

    for (int tile_start = 0; tile_start < num_cols; tile_start += vector_tile_cols) {
        const int tile_cols =
            (tile_start + vector_tile_cols <= num_cols)
                ? vector_tile_cols
                : (num_cols - tile_start);

        // V2: cache a tile of vector in shared memory for all rows in the block.
        for (int col = thread_id; col < tile_cols; col += blockDim.x) {
            shared_vector[col] = vector[tile_start + col];
        }
        __syncthreads();

        if (valid_row) {
            partial_sum += strided_dot(
                weight_row + tile_start,
                shared_vector,
                tile_cols,
                lane_id,
                32);
        }
        __syncthreads();
    }

    if (valid_row) {
        partial_sum = warp_sum(partial_sum);
        if (lane_id == 0) {
            store_output(output, output_row, partial_sum);
        }
    }
}

__global__ void optimized_gemv_v3_half_kernel(
    const __half* __restrict__ weight,
    const __half* __restrict__ vector,
    __half* __restrict__ output,
    int num_rows,
    int num_cols) {
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 31;
    const int warp_id = thread_id >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp_id;
    const bool valid_row = output_row < num_rows;

    extern __shared__ __align__(16) unsigned char shared_storage[];
    __half* shared_vector = reinterpret_cast<__half*>(shared_storage);

    // V3: load the full FP16 vector once per block, then reuse it for 8 rows.
    for (int col = thread_id; col < num_cols; col += blockDim.x) {
        shared_vector[col] = vector[col];
    }
    __syncthreads();

    if (!valid_row) {
        return;
    }

    const __half* weight_row =
        weight + static_cast<int64_t>(output_row) * num_cols;
    float partial_sum = strided_dot(weight_row, shared_vector, num_cols, lane_id, 32);
    partial_sum = warp_sum(partial_sum);

    if (lane_id == 0) {
        store_output(output, output_row, partial_sum);
    }
}

template <typename scalar_t>
void launch_naive(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int num_rows = static_cast<int>(weight.size(0));
    const int num_cols = static_cast<int>(weight.size(1));
    const int blocks = (num_rows + threads - 1) / threads;

    // Naive: one thread computes one row.
    naive_gemv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        reinterpret_cast<const scalar_t*>(vector.data_ptr()),
        reinterpret_cast<scalar_t*>(output.data_ptr()),
        num_rows,
        num_cols);
}

template <typename scalar_t>
void launch_optimized(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int num_rows = static_cast<int>(weight.size(0));
    const int num_cols = static_cast<int>(weight.size(1));

    // Optimized: one block computes one row.
    optimized_gemv_kernel<scalar_t><<<num_rows, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        reinterpret_cast<const scalar_t*>(vector.data_ptr()),
        reinterpret_cast<scalar_t*>(output.data_ptr()),
        num_rows,
        num_cols);
}

template <typename scalar_t>
void launch_optimized_v1(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output,
    cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int num_rows = static_cast<int>(weight.size(0));
    const int num_cols = static_cast<int>(weight.size(1));
    const int blocks = (num_rows + warps_per_block - 1) / warps_per_block;

    // V1: one warp computes one row.
    optimized_gemv_v1_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        reinterpret_cast<const scalar_t*>(vector.data_ptr()),
        reinterpret_cast<scalar_t*>(output.data_ptr()),
        num_rows,
        num_cols);
}

template <typename scalar_t>
void launch_optimized_v2(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output,
    cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    constexpr int vector_tile_cols = 1024;
    const int num_rows = static_cast<int>(weight.size(0));
    const int num_cols = static_cast<int>(weight.size(1));
    const int blocks = (num_rows + warps_per_block - 1) / warps_per_block;
    const size_t shared_bytes = vector_tile_cols * sizeof(scalar_t);

    // V2: one warp per row, plus shared-memory vector tiling.
    optimized_gemv_v2_kernel<scalar_t, vector_tile_cols>
        <<<blocks, threads, shared_bytes, stream>>>(
            reinterpret_cast<const scalar_t*>(weight.data_ptr()),
            reinterpret_cast<const scalar_t*>(vector.data_ptr()),
            reinterpret_cast<scalar_t*>(output.data_ptr()),
            num_rows,
            num_cols);
}

void launch_optimized_v3_half(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output,
    cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    constexpr size_t max_default_shared_bytes = 48 * 1024;
    const int num_rows = static_cast<int>(weight.size(0));
    const int num_cols = static_cast<int>(weight.size(1));
    const int blocks = (num_rows + warps_per_block - 1) / warps_per_block;
    const size_t shared_bytes = static_cast<size_t>(num_cols) * sizeof(__half);

    if (shared_bytes <= max_default_shared_bytes) {
        optimized_gemv_v3_half_kernel<<<blocks, threads, shared_bytes, stream>>>(
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(vector.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_rows,
            num_cols);
    } else {
        launch_optimized_v1<__half>(weight, vector, output, stream);
    }
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(error));
}

}  // namespace

void naive_gemv_out_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output) {
    check_inputs(weight, vector, output);
    c10::cuda::CUDAGuard device_guard(weight.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (weight.scalar_type() == at::kFloat) {
        launch_naive<float>(weight, vector, output, stream);
    } else {
        launch_naive<__half>(weight, vector, output, stream);
    }
    check_launch();
}

void optimized_gemv_out_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output) {
    check_inputs(weight, vector, output);
    c10::cuda::CUDAGuard device_guard(weight.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (weight.scalar_type() == at::kFloat) {
        launch_optimized<float>(weight, vector, output, stream);
    } else {
        launch_optimized<__half>(weight, vector, output, stream);
    }
    check_launch();
}

void optimized_gemv_v1_out_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output) {
    check_inputs(weight, vector, output);
    c10::cuda::CUDAGuard device_guard(weight.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (weight.scalar_type() == at::kFloat) {
        launch_optimized_v1<float>(weight, vector, output, stream);
    } else {
        launch_optimized_v1<__half>(weight, vector, output, stream);
    }
    check_launch();
}

void optimized_gemv_v2_out_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output) {
    check_inputs(weight, vector, output);
    c10::cuda::CUDAGuard device_guard(weight.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (weight.scalar_type() == at::kFloat) {
        launch_optimized_v2<float>(weight, vector, output, stream);
    } else {
        launch_optimized_v2<__half>(weight, vector, output, stream);
    }
    check_launch();
}

void optimized_gemv_v3_out_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output) {
    check_inputs(weight, vector, output);
    c10::cuda::CUDAGuard device_guard(weight.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (weight.scalar_type() == at::kFloat) {
        launch_optimized_v1<float>(weight, vector, output, stream);
    } else {
        launch_optimized_v3_half(weight, vector, output, stream);
    }
    check_launch();
}

void cublas_gemv_out_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& vector,
    torch::Tensor& output) {
    check_inputs(weight, vector, output);
    c10::cuda::CUDAGuard device_guard(weight.device());

    const int num_rows = static_cast<int>(weight.size(0));
    const int num_cols = static_cast<int>(weight.size(1));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cublasPointerMode_t old_pointer_mode;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &old_pointer_mode));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool is_fp32 = weight.scalar_type() == at::kFloat;
    const cudaDataType_t data_type = is_fp32 ? CUDA_R_32F : CUDA_R_16F;
    const cublasGemmAlgo_t algorithm =
        is_fp32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // PyTorch stores weight as row-major [num_rows, num_cols]. cuBLAS reads the
    // same bytes as column-major [num_cols, num_rows], then CUBLAS_OP_T gives
    // the desired [num_rows, num_cols] view without copying.
    // cuBLAS handles its own global/shared/register tiling internally.
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        num_rows,
        1,
        num_cols,
        &alpha,
        weight.data_ptr(),
        data_type,
        num_cols,
        vector.data_ptr(),
        data_type,
        num_cols,
        &beta,
        output.data_ptr(),
        data_type,
        num_rows,
        CUBLAS_COMPUTE_32F,
        algorithm));

    CUBLAS_CHECK(cublasSetPointerMode(handle, old_pointer_mode));
}
