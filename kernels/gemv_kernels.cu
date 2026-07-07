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

void check_inputs(
    const torch::Tensor& W,
    const torch::Tensor& x,
    const torch::Tensor& out) {
    CHECK_CUDA(W);
    CHECK_CUDA(x);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(W.scalar_type() == at::kHalf ||
                W.scalar_type() == at::kFloat,
                "W must be float16 or float32");
    TORCH_CHECK(x.scalar_type() == W.scalar_type(),
                "x must have the same dtype as W");
    TORCH_CHECK(out.scalar_type() == W.scalar_type(),
                "out must have the same dtype as W");
    TORCH_CHECK(W.dim() == 2, "W must have shape [M, N]");
    TORCH_CHECK(x.dim() == 1, "x must have shape [N]");
    TORCH_CHECK(out.dim() == 1, "out must have shape [M]");
    TORCH_CHECK(W.size(1) == x.size(0), "W.size(1) must equal x.size(0)");
    TORCH_CHECK(W.size(0) == out.size(0), "out.size(0) must equal W.size(0)");
    TORCH_CHECK(W.device() == x.device() && W.device() == out.device(),
                "W, x, and out must be on the same CUDA device");
    TORCH_CHECK(W.size(0) <= INT_MAX && W.size(1) <= INT_MAX,
                "M and N must fit in a 32-bit integer");
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
    scalar_t* out, int row, float value) {
    out[row] = static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ void store_output<__half>(
    __half* out, int row, float value) {
    out[row] = __float2half_rn(value);
}

template <typename scalar_t>
__global__ void naive_gemv_kernel(
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int M,
    int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    }

    const scalar_t* row_ptr = W + static_cast<int64_t>(row) * N;
    float acc = 0.0f;
    for (int col = 0; col < N; ++col) {
        acc = fmaf(load_as_float(row_ptr[col]), load_as_float(x[col]), acc);
    }
    store_output(out, row, acc);
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float strided_dot(
    const __half* row_ptr,
    const __half* x,
    int N,
    int tid,
    int threads) {
    float acc = 0.0f;

    // half2 is safe only when both row and vector pointers are 4-byte aligned
    // and the row has an even number of elements. Otherwise use scalar loads.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(row_ptr) & 0x3u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(x) & 0x3u) == 0u);
    if (aligned && (N % 2 == 0)) {
        const half2* row2 = reinterpret_cast<const half2*>(row_ptr);
        const half2* x2 = reinterpret_cast<const half2*>(x);
        const int N2 = N / 2;

        for (int col2 = tid; col2 < N2; col2 += threads) {
            const float2 w_pair = __half22float2(row2[col2]);
            const float2 x_pair = __half22float2(x2[col2]);
            acc = fmaf(w_pair.x, x_pair.x, acc);
            acc = fmaf(w_pair.y, x_pair.y, acc);
        }
    } else {
        for (int col = tid; col < N; col += threads) {
            acc = fmaf(__half2float(row_ptr[col]), __half2float(x[col]), acc);
        }
    }

    return acc;
}

__device__ __forceinline__ float strided_dot(
    const float* row_ptr,
    const float* x,
    int N,
    int tid,
    int threads) {
    float acc = 0.0f;

    // float2 is safe only when both pointers are 8-byte aligned and the row
    // has an even number of elements. Otherwise use scalar float loads.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(row_ptr) & 0x7u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(x) & 0x7u) == 0u);
    if (aligned && (N % 2 == 0)) {
        const float2* row2 = reinterpret_cast<const float2*>(row_ptr);
        const float2* x2 = reinterpret_cast<const float2*>(x);
        const int N2 = N / 2;

        for (int col2 = tid; col2 < N2; col2 += threads) {
            const float2 w_pair = row2[col2];
            const float2 x_pair = x2[col2];
            acc = fmaf(w_pair.x, x_pair.x, acc);
            acc = fmaf(w_pair.y, x_pair.y, acc);
        }
    } else {
        for (int col = tid; col < N; col += threads) {
            acc = fmaf(row_ptr[col], x[col], acc);
        }
    }

    return acc;
}

template <typename scalar_t>
__global__ void optimized_gemv_kernel(
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int M,
    int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_count = blockDim.x >> 5;

    if (row >= M) {
        return;
    }

    const scalar_t* row_ptr = W + static_cast<int64_t>(row) * N;
    float acc = strided_dot(row_ptr, x, N, tid, blockDim.x);

    acc = warp_sum(acc);

    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            store_output(out, row, block_sum);
        }
    }
}

template <typename scalar_t>
void launch_naive(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int M = static_cast<int>(W.size(0));
    const int N = static_cast<int>(W.size(1));
    const int blocks = (M + threads - 1) / threads;

    naive_gemv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(W.data_ptr()),
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        M,
        N);
}

template <typename scalar_t>
void launch_optimized(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int M = static_cast<int>(W.size(0));
    const int N = static_cast<int>(W.size(1));

    optimized_gemv_kernel<scalar_t><<<M, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(W.data_ptr()),
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        M,
        N);
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(error));
}

}  // namespace

void naive_gemv_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out) {
    check_inputs(W, x, out);
    c10::cuda::CUDAGuard device_guard(W.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (W.scalar_type() == at::kFloat) {
        launch_naive<float>(W, x, out, stream);
    } else {
        launch_naive<__half>(W, x, out, stream);
    }
    check_launch();
}

void optimized_gemv_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out) {
    check_inputs(W, x, out);
    c10::cuda::CUDAGuard device_guard(W.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (W.scalar_type() == at::kFloat) {
        launch_optimized<float>(W, x, out, stream);
    } else {
        launch_optimized<__half>(W, x, out, stream);
    }
    check_launch();
}

void cublas_gemv_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out) {
    check_inputs(W, x, out);
    c10::cuda::CUDAGuard device_guard(W.device());

    const int M = static_cast<int>(W.size(0));
    const int N = static_cast<int>(W.size(1));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cublasPointerMode_t old_pointer_mode;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &old_pointer_mode));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool is_fp32 = W.scalar_type() == at::kFloat;
    const cudaDataType_t data_type = is_fp32 ? CUDA_R_32F : CUDA_R_16F;
    const cublasGemmAlgo_t algorithm =
        is_fp32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // PyTorch stores W as row-major [M, N]. The same bytes look like a
    // column-major [N, M] matrix to cuBLAS. Transposing that interpretation
    // gives the desired [M, N] operation without copying W:
    //     op(A_col) [M, N] * x [N, 1] -> y [M, 1].
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        M,
        1,
        N,
        &alpha,
        W.data_ptr(),
        data_type,
        N,
        x.data_ptr(),
        data_type,
        N,
        &beta,
        out.data_ptr(),
        data_type,
        M,
        CUBLAS_COMPUTE_32F,
        algorithm));

    CUBLAS_CHECK(cublasSetPointerMode(handle, old_pointer_mode));
}
