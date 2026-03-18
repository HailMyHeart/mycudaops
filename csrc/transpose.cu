#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_DIM 32
#define PADDING 1
__global__ void transpose_naive_kernel(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if(x<n&&y<m){
        output[x*m+y] = input[y*n+x];
    }
}

__global__ void transpose_shared_kernel(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if(x<n && y<m){
        tile[threadIdx.y][threadIdx.x] = input[y*n+x];
    }
    __syncthreads();
    x = blockIdx.y*blockDim.y+threadIdx.x;
    y = blockIdx.x*blockDim.x+threadIdx.y;
    if(x<m && y<n){
        output[y*m+x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_padding_kernel(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    __shared__ float tile[TILE_DIM][TILE_DIM+PADDING];
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if(x<n && y<m){
        tile[threadIdx.y][threadIdx.x] = input[y*n+x];
    }
    __syncthreads();
    x = blockIdx.y*blockDim.y+threadIdx.x;
    y = blockIdx.x*blockDim.x+threadIdx.y;
    if(x<m && y<n){
        output[y*m+x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_swizzle_kernel(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    __shared__ float tile[TILE_DIM][TILE_DIM];
    if(x<n && y<m){
        tile[threadIdx.y][threadIdx.x^threadIdx.y] = input[y*n+x];
    }
    __syncthreads();
    x = blockIdx.y*blockDim.y+threadIdx.x;
    y = blockIdx.x*blockDim.x+threadIdx.y;
    if(x<m&&y<n){
        output[y*m+x] = tile[threadIdx.x][threadIdx.y^threadIdx.x];
    }
}

torch::Tensor transpose_naive(torch::Tensor input) {
    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty({width, height}, input.options());
    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM); // 1024 线程/Block，4090 满载
    transpose_naive_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), height, width);
    return output;
}

torch::Tensor transpose_shared(torch::Tensor input) {
    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty({width, height}, input.options());
    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM); // 1024 线程/Block，4090 满载
    transpose_shared_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), height, width);
    return output;
}

torch::Tensor transpose_shared_padding(torch::Tensor input) {
    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty({width, height}, input.options());
    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM); // 1024 线程/Block，4090 满载
    transpose_padding_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), height, width);
    return output;
}

torch::Tensor transpose_shared_swizzle(torch::Tensor input) {
    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty({width, height}, input.options());
    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM); // 1024 线程/Block，4090 满载
    transpose_swizzle_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), height, width);
    return output;
}