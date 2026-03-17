#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
__device__ __forceinline__ float online_merge(float& m_a, float& d_a, float m_b, float d_b){
    if(m_b>m_a){
        d_a = d_a*expf(m_a-m_b)+d_b;
        m_a = m_b;
    }else{
        d_a = d_a + d_b*expf(m_b-m_a);
    }
}
__device__ __forceinline__ void warp_online_merge(float& m_a, float& d_a){
    #pragma unroll
    for(int offset = 16; offset>0; offset>>=1){
        float m_b = __shfl_down_sync(0xffffffff, m_a, offset);
        float d_b = __shfl_down_sync(0xffffffff, d_a, offset);
        online_merge(m_a, d_a, m_b, d_b);
    }
}
__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int K){
    const float* cur_input = input+blockIdx.x*K;
    float* cur_output = output+blockIdx.x*K;
    float m = -FLT_MAX;
    float d = 0.0f;
    for(int i = threadIdx.x; i<K; i+=blockDim.x){
        online_merge(m, d, cur_input[i], 1.0f);
    }
    warp_online_merge(m, d);
    __shared__ float s_m[32];
    __shared__ float s_d[32];
    int lane = threadIdx.x%32;
    int wid = threadIdx.x/32;
    if(lane==0){
        s_m[wid] = m;
        s_d[wid] = d;
    }
    __syncthreads();
    if(wid==0){
        m = lane<(blockDim.x/32)?s_m[lane]:-FLT_MAX;
        d = lane<(blockDim.x/32)?s_d[lane]:0.0f;
        warp_online_merge(m, d);
        if(lane==0){
            s_m[0] = m;
            s_d[0] = d;
        }
    }

    __syncthreads();
    float final_m = s_m[0];
    float final_d = s_d[0];
    for(int i = threadIdx.x; i<K; i+=blockDim.x){
        cur_output[i] = exp(cur_input[i]-final_m)/final_d;
    }
    
}

torch::Tensor softmax_cuda_forward(torch::Tensor input, int threads){
    auto rows = input.size(0);
    auto K = input.size(1);
    auto output = torch::empty_like(input);

    softmax_kernel<<<rows, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        K
    );

    return output;
}