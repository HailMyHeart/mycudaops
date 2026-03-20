#include<cuda_runtime.h>
#include<torch/extension.h>
__device__ __forceinline__ float warp_reduce_sum(float val){
#pragma unroll
    for(int offset = 16; offset>0; offset>>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val){
    static __shared__ float sm[32];
    int lane = threadIdx.x%32;
    int wid = threadIdx.x/32;
    val = warp_reduce_sum(val);
    if(lane==0) sm[wid] = val;
    __syncthreads();
    val = (threadIdx.x<blockDim.x/32) ? sm[lane] : 0.0f;
    if(wid==0) val = warp_reduce_sum(val);
    __syncthreads();
    return val;
}

__global__ void layernorm_kernel(const float* __restrict__ input,\
    const float* __restrict__ gamma, const float* __restrict__ beta, \
    int M, int N, float eps, float* __restrict__ output){
        int m = blockIdx.x;
        if(m>=M) return;
        const float * in = input+m*N;
        float* out = output+m*N;

        float sum = 0.f;
        float sum2 = 0.f;

        for(int i = threadIdx.x; i<N; i+=blockDim.x){
            float val = in[i];
            sum += val;
            sum2 += val*val;
        }
        sum = block_reduce_sum(sum);
        sum2 = block_reduce_sum(sum2);

        __shared__ float mean, rstd;
        if(threadIdx.x==0){
            mean = sum/N;
            rstd = rsqrtf(sum2/N - mean*mean + eps);
        }
        __syncthreads();

        for(int i = threadIdx.x; i<N; i+=blockDim.x){
            out[i] = (in[i]-mean)*rstd*gamma[i]+beta[i];
        }

    }

    torch::Tensor layernorm_cuda_warp(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float epsilon) {
    auto output = torch::empty_like(input);
    int N = input.size(0);
    int D = input.size(1);
    
    // 动态选择线程数，通常 256 或 512
    int threads = 1024; 
    layernorm_kernel<<<N, threads>>>(
        input.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        beta.data_ptr<float>(), 
        N, D, epsilon,
        output.data_ptr<float>()
    );
    return output;
}