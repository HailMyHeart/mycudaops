#include <torch/extension.h>

// 前向声明
torch::Tensor reduce_cuda_forward(torch::Tensor input, int threads, int blocks);
torch::Tensor softmax_cuda_forward(torch::Tensor input, int threads); 
torch::Tensor vec_add_cuda_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor transpose_naive(torch::Tensor input);
torch::Tensor transpose_shared(torch::Tensor input);
torch::Tensor transpose_shared_padding(torch::Tensor input);
torch::Tensor transpose_shared_swizzle(torch::Tensor input);
torch::Tensor gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta, int version);
torch::Tensor layernorm_cuda_warp(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_cuda_forward, "CUDA Reduce Sum (Grid-Stride Loop)",
          py::arg("input"), py::arg("threads") = 1024, py::arg("blocks") = 512);
    
    m.def("softmax", &softmax_cuda_forward, "online Softmax",
          py::arg("input"), py::arg("threads") = 1024);
    
    m.def("layernorm", &layernorm_cuda_warp, "CUDA LayerNorm",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("epsilon"));
    
    m.def("vec_add", &vec_add_cuda_forward, "CUDA VecAdd (float4)",
          py::arg("a"), py::arg("b"));
    
    m.def("transpose_naive", &transpose_naive, "CUDA Transpose (Naive)",
          py::arg("input"));
    
    m.def("transpose_shared", &transpose_shared, "CUDA Transpose (Shared Memory)",
          py::arg("input"));
    
    m.def("transpose_shared_padding", &transpose_shared_padding, "CUDA Transpose (Shared Memory with Padding)",
          py::arg("input"));
    
    m.def("transpose_shared_swizzle", &transpose_shared_swizzle, "CUDA Transpose (Shared Memory with Swizzling)",
          py::arg("input"));
    
    // 补全了 gemm_forward 的参数定义
    m.def("gemm_forward", &gemm_forward, "GEMM 4-Kernel Evolution",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("version") = 0);
}