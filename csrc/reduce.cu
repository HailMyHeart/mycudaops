#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp 级别的规约：使用 Shuffle 指令
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block 级别的规约
__device__ __forceinline__ float block_reduce(float val) {
/***
编译期确定大小
不能像动态 shared memory 那样通过 launch 参数传大小。
内存布局固定
编译器知道这块 shared memory 的大小和位置。
适合固定上限的归约
注释里写“最多 32 个 Warp”，说明这里假设一个 block 最多 32 个 warp，所以开 32 个槽位刚好够用。
***/
    static __shared__ float shared[32]; // 最多 32 个 Warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 最后一个 Warp 规约所有 Warp 的结果
    val = threadIdx.x < (blockDim.x/32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce(val);

    return val;
}

__global__ void reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    float sum = 0.0f;
    
    // Grid-Stride Loop: 保证合并访存，同时处理超过 Grid 大小的数据
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = block_reduce(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor reduce_cuda_forward(torch::Tensor input, int threads, int blocks) {
    auto n = input.numel();
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({1}, options);

    reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );

    return output;
}
