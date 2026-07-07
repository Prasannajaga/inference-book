#include <torch/extension.h>

void naive_gemv_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out);

void optimized_gemv_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out);

void optimized_gemv_v1_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out);

void optimized_gemv_v2_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out);

void optimized_gemv_v3_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out);

void cublas_gemv_out_cuda(
    const torch::Tensor& W,
    const torch::Tensor& x,
    torch::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_gemv_out", &naive_gemv_out_cuda,
          "Naive same-dtype FP16/FP32 GEMV (CUDA)");
    m.def("optimized_gemv_out", &optimized_gemv_out_cuda,
          "Cooperative same-dtype FP16/FP32 GEMV (CUDA)");
    m.def("optimized_gemv_v1_out", &optimized_gemv_v1_out_cuda,
          "V1 warp-per-row same-dtype FP16/FP32 GEMV (CUDA)");
    m.def("optimized_gemv_v2_out", &optimized_gemv_v2_out_cuda,
          "V2 shared-vector same-dtype FP16/FP32 GEMV (CUDA)");
    m.def("optimized_gemv_v3_out", &optimized_gemv_v3_out_cuda,
          "V3 full shared-vector FP16 GEMV with FP32 fallback (CUDA)");
    m.def("cublas_gemv_out", &cublas_gemv_out_cuda,
          "Direct same-dtype cuBLAS GEMMEx wrapper for GEMV (CUDA)");
}
