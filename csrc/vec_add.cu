#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vec_add_float4_kernel(const float* a, const float* b, float* c, int n) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n4 = n / 4; 

    for (int i = idx; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vb = b4[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c4[i] = vc;
    }

    int remainder = n % 4;
    if (remainder > 0 && idx == 0) { 
        for (int i = n - remainder; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

// Host 端封装
torch::Tensor vec_add_cuda_forward(torch::Tensor a, torch::Tensor b) {
    auto n = a.numel();
    auto c = torch::empty_like(a);
    int threads = 1024;
    // 因为每个线程处理 4 个，所以 blocks 数可以减小
    int blocks = (n / 4 + threads - 1) / threads;
    blocks = std::min(blocks, 128*8);

    vec_add_float4_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}