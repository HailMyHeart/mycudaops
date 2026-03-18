#include <torch/extension.h>

// 先声明其他文件里定义的函数（注意：这里不用 include .cu 文件，只需要声明签名）
torch::Tensor reduce_cuda_forward(torch::Tensor input, int threads, int blocks);
torch::Tensor softmax_cuda_forward(torch::Tensor input, int threads); 
torch::Tensor vec_add_cuda_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor transpose_naive(torch::Tensor input);
torch::Tensor transpose_shared(torch::Tensor input);
torch::Tensor transpose_shared_padding(torch::Tensor input);
torch::Tensor transpose_shared_swizzle(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_cuda_forward, "CUDA Reduce Sum (Grid-Stride Loop)",
          py::arg("input"), py::arg("threads") = 1024, py::arg("blocks") = 512);
    m.def("softmax", &softmax_cuda_forward, "online Softmax ",
          py::arg("input"), py::arg("threads") = 1024);
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
}