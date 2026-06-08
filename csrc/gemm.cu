#include <cuda_runtime.h>
#include <torch/extension.h>
#define CEIL(a, b) ((a)+(b)-1)/(b)
#define FLOAT4_VAL(a) (reinterpret_cast<float4*>(&(a))[0])
#define OFFSET(i, j, n) ((i)*(n)+(j))
__global__ void gemm_naive_kernel(int M, int N, int K, const float alpha, const float beta, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row < M && col < N){
        float sum = 0.f;
        for(int k = 0; k<K; k++){
            sum += A[row*K+k]*B[k*N+col];
        }
        C[row*N+col] = alpha * sum + beta * C[row*N+col];
    }

}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_tiled_kernel(int M, int N, int K, const float alpha, const float beta, const float*  A, const float*  B, float*  C){

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    int block_row_threads = BN/TN;
    int block_col_threads = BM/TM;
    int thread_num = block_row_threads*block_col_threads;

    int tx = (threadIdx.x%block_row_threads)*TN;
    int ty = (threadIdx.x/block_row_threads)*TM;
    A = A+blockIdx.y*BM*K;
    B = B+blockIdx.x*BN;
    C = C+blockIdx.y*BM*N+blockIdx.x*BN;

    int a_tile_row = threadIdx.x / BK, a_tile_col = threadIdx.x % BK;
    int b_tile_row = threadIdx.x / BN, b_tile_col = threadIdx.x % BN;

    int a_tile_stride = thread_num/BK;
    int b_tile_stride = thread_num/BN;

    float Csub[TM][TN] = {0.f};
#pragma unroll
    for(int k = 0; k<K; k+=BK){ //大循环，对每个tile边搬运边对Csub进行类加，循环结束后Csub就是最终结果
        
        //以下搬运A和B的tile到共享内存
#pragma unroll
        for(int i = 0; i<BM; i+=a_tile_stride){
            As[(a_tile_row+i)*BK+a_tile_col] = A[(a_tile_row+i)*K+a_tile_col];
        }
#pragma unroll
        for(int i = 0; i<BK; i+=b_tile_stride){
            Bs[(b_tile_row+i)*BN+b_tile_col] = B[(b_tile_row+i)*N+b_tile_col];
        }
        __syncthreads();

        //搬运一个tile之后移动A和B的指针
        A += BK;
        B += BK*N;

        //计算Csub
#pragma unroll
        for(int i = 0; i<BK; i++){
#pragma unroll
            for(int j = 0; j<TM; j++){
#pragma unroll
                for(int l = 0; l<TN; l++){
                    Csub[j][l] += As[(ty+j)*BK+i]*Bs[i*BN+tx+l];
                }
            }
        }
        __syncthreads();

    }
#pragma unroll
    for(int i = 0; i<TM; i++){
        for(int j = 0; j<TN; j++){
            C[(ty+i)*N+tx+j] = alpha * Csub[i][j] + beta * C[(ty+i)*N+tx+j];
        }
    }


}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_tiled_float4_kernel(int M, int N, int K, const float alpha, const float beta,  float*  A,  float*  B, float*  C){

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    const int block_row_threads = BN/TN;
    const int block_col_threads = BM/TM;
    const int thread_num = block_row_threads*block_col_threads;

    int tx = (threadIdx.x%block_row_threads)*TN;
    int ty = (threadIdx.x/block_row_threads)*TM;
    A = A+blockIdx.y*BM*K;
    B = B+blockIdx.x*BN;
    C = C+blockIdx.y*BM*N+blockIdx.x*BN;


    const int move_iters_a = BK*BM/4/thread_num;
    const int move_iters_b = BK*BN/4/thread_num;

    int a_tile_row = threadIdx.x / (BK/4), a_tile_col = (threadIdx.x % (BK/4)) * 4;
    int b_tile_row = threadIdx.x / (BN/4), b_tile_col = (threadIdx.x % (BN/4)) * 4;
    int a_tile_stride = BM/move_iters_a;
    int b_tile_stride = BK/move_iters_b;

    float Csub[TM][TN] = {0.f};
    float a_tile_reg[move_iters_a*4] = {0.f};   
    float a_frag[TM], b_frag[TN];
#pragma unroll
    for(int k = 0; k<K; k+=BK){ //大循环，对每个tile边搬运边对Csub进行类加，循环结束后Csub就是最终结果
        
        /***
         * 以下搬运A和B的tile到共享内存,计算CSub的时候，
        要将shared memory中的数据用lds 128.质量向量化取到寄存器中,
        为了这个操作铺垫，这里需要把从显存中读的数据转置，
        所以用regs做一下中转，先从显存读到regs中，转置后再写到shared memory中.
        这里可能会有bank conflicts，但是这里循环次数比计算Csub时load的循环次数要少，
        所以优先把冲突放到这里。
        ***/
#pragma unroll
        for(int i = 0; i<BM; i+=a_tile_stride){
            int reg_idx = i/a_tile_stride*4;
            FLOAT4_VAL(a_tile_reg[reg_idx]) = FLOAT4_VAL(A[OFFSET(a_tile_row+i, a_tile_col, K)]);
            As[OFFSET(a_tile_col, a_tile_row+i, BM)] = a_tile_reg[reg_idx];
            As[OFFSET(a_tile_col+1, a_tile_row+i, BM)] = a_tile_reg[reg_idx+1];
            As[OFFSET(a_tile_col+2, a_tile_row+i, BM)] = a_tile_reg[reg_idx+2];
            As[OFFSET(a_tile_col+3, a_tile_row+i, BM)] = a_tile_reg[reg_idx+3];
        }
#pragma unroll
        for(int i = 0; i<BK; i+=b_tile_stride){
            FLOAT4_VAL(Bs[OFFSET(b_tile_row+i, b_tile_col, BN)]) = \
            FLOAT4_VAL(B[OFFSET(b_tile_row+i, b_tile_col, N)]);
        }
        __syncthreads();

        //搬运一个tile之后移动A和B的指针
        A += BK;
        B += BK*N;

        //计算Csub
#pragma unroll
        for(int i = 0; i<BK; i++){
            //预先取得外积所需的As中大小为TM的一列（转置后时一行）以及Bs中大小为TN的一行到寄存器中，减少对shared memory的访问次数
            //提升计算强度（相对于分母为shared mem的访存）
#pragma unroll
        
            for(int m = 0; m<TM; m+=4){
                FLOAT4_VAL(a_frag[m]) = FLOAT4_VAL(As[OFFSET(i,ty+m,BM)]);
            }
#pragma unroll
            for(int n = 0; n<TN; n+=4){
                FLOAT4_VAL(b_frag[n]) = FLOAT4_VAL(Bs[OFFSET(i,tx+n,BN)]);
            }
#pragma unroll
            for(int j = 0; j<TM; j++){
#pragma unroll
                for(int l = 0; l<TN; l++){
                    Csub[j][l] += a_frag[j]*b_frag[l];
                }
            }
        }
        __syncthreads();

    }
#pragma unroll
    for(int i = 0; i<TM; i++){
        for(int j = 0; j<TN; j+=4){
            float4 tmp = FLOAT4_VAL(C[OFFSET(ty+i, tx+j, N)]);
            tmp.x = alpha*Csub[i][j]+beta*tmp.x;
            tmp.y = alpha*Csub[i][j+1]+beta*tmp.y;
            tmp.z = alpha*Csub[i][j+2]+beta*tmp.z;
            tmp.w = alpha*Csub[i][j+3]+beta*tmp.w;
            FLOAT4_VAL(C[OFFSET(ty+i, tx+j, N)]) = tmp;
        }
    }


}


/**
 * 总体思路：先ldg/lds,再计算。以掩盖延迟。
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_pingpong_kernel(int M, int N, int K, const float alpha, const float beta,  float*  A,  float*  B, float*  C){

    __shared__ float As[2][BM*BK];
    __shared__ float Bs[2][BK*BN];
    const int block_row_threads = BN/TN;
    const int block_col_threads = BM/TM;
    const int thread_num = block_row_threads*block_col_threads;

    int tx = (threadIdx.x%block_row_threads)*TN;
    int ty = (threadIdx.x/block_row_threads)*TM;
    A = A+blockIdx.y*BM*K;
    B = B+blockIdx.x*BN;
    C = C+blockIdx.y*BM*N+blockIdx.x*BN;


    const int move_iters_a = BK*BM/4/thread_num;
    const int move_iters_b = BK*BN/4/thread_num;

    // A 是 (M, K)，所以行偏移由 blockIdx.y 决定
    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = (threadIdx.x % (BK / 4)) * 4;

    // B 是 (K, N)，所以列偏移由 blockIdx.x 决定
    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
    int a_tile_stride = BM/move_iters_a;
    int b_tile_stride = BK/move_iters_b;

    float Csub[TM][TN] = {0.f};
    float a_tile_reg[move_iters_a*4] = {0.f};   
    float b_tile_reg[move_iters_b*4] = {0.f};   

    float a_frag[2][TM], b_frag[2][TN];

    /***
     * pingpong 预取，为第一个buffer的Csub计算做准备
     */
#pragma unroll
    for(int i = 0; i<BM; i+=a_tile_stride){
        int reg_idx = i/a_tile_stride*4;
        FLOAT4_VAL(a_tile_reg[reg_idx]) = FLOAT4_VAL(A[OFFSET(a_tile_row+i, a_tile_col, K)]);
        As[0][OFFSET(a_tile_col, a_tile_row+i, BM)] = a_tile_reg[reg_idx];
        As[0][OFFSET(a_tile_col+1, a_tile_row+i, BM)] = a_tile_reg[reg_idx+1];
        As[0][OFFSET(a_tile_col+2, a_tile_row+i, BM)] = a_tile_reg[reg_idx+2];
        As[0][OFFSET(a_tile_col+3, a_tile_row+i, BM)] = a_tile_reg[reg_idx+3];
    }
#pragma unroll
    for(int i = 0; i<BK; i+=b_tile_stride){
        FLOAT4_VAL(Bs[0][OFFSET(b_tile_row+i, b_tile_col, BN)]) = \
        FLOAT4_VAL(B[OFFSET(b_tile_row+i, b_tile_col, N)]);
    }
    //搬运一个tile之后移动A和B的指针
    A += BK;
    B += BK*N;
    __syncthreads();

#pragma unroll
    for(int m = 0; m<TM; m+=4){
        FLOAT4_VAL(a_frag[0][m]) = FLOAT4_VAL(As[0][OFFSET(0,ty+m,BM)]);
    }
#pragma unroll
    for(int n = 0; n<TN; n+=4){
        FLOAT4_VAL(b_frag[0][n]) = FLOAT4_VAL(Bs[0][OFFSET(0,tx+n,BN)]);
    }

    int write_index = 1, load_index = 0;

    int k = 0;
#pragma unroll
    do{
        k+=BK;
        /***
         * 循环内，先将global mem->register,
         * 这里接下来不->shared mem中，
         * 是因为指令依赖关系，先load到reg中，
         * 等计算Csub的时候再load到shared mem中，提升指令级并行，
         * 这样可以隐藏global mem的延迟
         */
        if(k<K){
#pragma unroll
            for(int i = 0; i<BM; i+=a_tile_stride){
                int reg_idx = i/a_tile_stride*4;
                FLOAT4_VAL(a_tile_reg[reg_idx]) = FLOAT4_VAL(A[OFFSET(a_tile_row+i, a_tile_col, K)]);
            }
#pragma unroll
            for(int i = 0; i<BK; i+=b_tile_stride){
                int reg_idx = i/move_iters_b*4;
                FLOAT4_VAL(b_tile_reg[reg_idx]) = \
                FLOAT4_VAL(B[OFFSET(b_tile_row+i, b_tile_col, N)]);
            }
            //搬运一个tile之后移动A和B的指针
            A += BK;
            B += BK*N;
        }

        //计算Csub
#pragma unroll
        for(int bk = 0; bk<BK-1; bk++){
#pragma unroll
            /***
             * 这里为什么要先load再计算？
             * load指令返回周期很长，
             * 所以要尽早发出，这样才能有效利用之后issue大量计算指令的周期，
             * 更好的掩盖load访存延迟
             */
            for(int m = 0; m<TM; m+=4){
                FLOAT4_VAL(a_frag[(bk+1)%2][m]) = FLOAT4_VAL(As[load_index][OFFSET(bk+1,ty+m,BM)]);
            }
#pragma unroll
            for(int n = 0; n<TN; n+=4){
                FLOAT4_VAL(b_frag[(bk+1)%2][n]) = FLOAT4_VAL(Bs[load_index][OFFSET(bk+1,tx+n,BN)]);
            }
#pragma unroll
            for(int j = 0; j<TM; j++){
#pragma unroll
                for(int l = 0; l<TN; l++){
                    Csub[j][l] += a_frag[(bk)%2][j]*b_frag[(bk)%2][l];
                }
            }
        }
        //这里为什么不直接用BK-1位置的元素计算？因为要尾迹补偿（epilogue computation）
        //，直接看循环最后


        if(k<K){
            /***
             * register->shared mem
             */
#pragma unroll
            for(int i = 0; i<BM; i+=a_tile_stride){
                int reg_idx = i/a_tile_stride*4;
                As[write_index][OFFSET(a_tile_col, i+a_tile_row, BM)] =\
                a_tile_reg[reg_idx];
                As[write_index][OFFSET(a_tile_col+1, i+a_tile_row, BM)] =\
                a_tile_reg[reg_idx+1];
                As[write_index][OFFSET(a_tile_col+2, i+a_tile_row, BM)] =\
                a_tile_reg[reg_idx+2];
                As[write_index][OFFSET(a_tile_col+3, i+a_tile_row, BM)] =\
                a_tile_reg[reg_idx+3];
            }
            for(int i=0; i<BK; i+=b_tile_stride){
                int reg_idx = i/move_iters_b*4;
                FLOAT4_VAL(Bs[write_index][OFFSET(b_tile_row+i, b_tile_col, BN)]) = \
                FLOAT4_VAL(b_tile_reg[reg_idx]);
            }
            __syncthreads();
            /***
             * shared_mem->register (仅一行/列) 为新一轮Csub计算做准备
             * 这里不
             */
            for(int m = 0; m<TM; m+=4){
                FLOAT4_VAL(a_frag[0][m]) = FLOAT4_VAL(As[write_index][OFFSET(0, ty+m, BM)]);
            }
            for(int n = 0; n<TN; n+=4){
                FLOAT4_VAL(b_frag[0][n]) = FLOAT4_VAL(Bs[write_index][OFFSET(0, tx+n, BN)]);
            }
            load_index = write_index;
            write_index ^= 1;
        }

        /***
         * epilogue computation,
         * 处理剩余循环项，
         * 掩盖 __syncthreads()（切换warp） 和 STS(ILP) 的延迟
         * 计算单元不能闲着
         */
        for(int j = 0; j<TM; j++){
            for(int l = 0; l<TN; l++){
                Csub[j][l]+=a_frag[(BK - 1) % 2][j]*b_frag[(BK - 1) % 2][l];
            }
        }
    
    }while(k<K);
#pragma unroll
    for(int i = 0; i<TM; i++){
        for(int j = 0; j<TN; j+=4){
            float4 tmp = FLOAT4_VAL(C[OFFSET(ty+i, tx+j, N)]);
            tmp.x = alpha*Csub[i][j]+beta*tmp.x;
            tmp.y = alpha*Csub[i][j+1]+beta*tmp.y;
            tmp.z = alpha*Csub[i][j+2]+beta*tmp.z;
            tmp.w = alpha*Csub[i][j+3]+beta*tmp.w;
            FLOAT4_VAL(C[OFFSET(ty+i, tx+j, N)]) = tmp;
        }
    }
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_final_kernel(int M, int N, int K, const float alpha, const float beta,  float*  A,  float*  B, float*  C){

    __shared__ float As[2][BK*(BM+4)];
    __shared__ float Bs[2][BK*(BN+4)];
    const int block_row_threads = BN/TN;
    const int block_col_threads = BM/TM;
    const int thread_num = block_row_threads*block_col_threads;

    int tx = (threadIdx.x%block_row_threads)*TN;
    int ty = (threadIdx.x/block_row_threads)*TM;
    A = A+blockIdx.y*BM*K;
    B = B+blockIdx.x*BN;
    C = C+blockIdx.y*BM*N+blockIdx.x*BN;


    const int move_iters_a = BK*BM/4/thread_num;
    const int move_iters_b = BK*BN/4/thread_num;

    // A 是 (M, K)，所以行偏移由 blockIdx.y 决定
    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = (threadIdx.x % (BK / 4)) * 4;

    // B 是 (K, N)，所以列偏移由 blockIdx.x 决定
    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
    int a_tile_stride = BM/move_iters_a;
    int b_tile_stride = BK/move_iters_b;

    float Csub[TM][TN] = {0.f};
    float a_tile_reg[move_iters_a*4] = {0.f};   
    float b_tile_reg[move_iters_b*4] = {0.f};   

    float a_frag[2][TM], b_frag[2][TN];

    /***
     * pingpong 预取，为第一个buffer的Csub计算做准备
     */
#pragma unroll
    for(int i = 0; i<BM; i+=a_tile_stride){
        int reg_idx = i/a_tile_stride*4;
        FLOAT4_VAL(a_tile_reg[reg_idx]) = FLOAT4_VAL(A[OFFSET(a_tile_row+i, a_tile_col, K)]);
        As[0][OFFSET(a_tile_col, a_tile_row+i, BM+4)] = a_tile_reg[reg_idx];
        As[0][OFFSET(a_tile_col+1, a_tile_row+i, BM+4)] = a_tile_reg[reg_idx+1];
        As[0][OFFSET(a_tile_col+2, a_tile_row+i, BM+4)] = a_tile_reg[reg_idx+2];
        As[0][OFFSET(a_tile_col+3, a_tile_row+i, BM+4)] = a_tile_reg[reg_idx+3];
    }
#pragma unroll
    for(int i = 0; i<BK; i+=b_tile_stride){
        FLOAT4_VAL(Bs[0][OFFSET(b_tile_row+i, b_tile_col, BN)]) = \
        FLOAT4_VAL(B[OFFSET(b_tile_row+i, b_tile_col, N)]);
    }
    //搬运一个tile之后移动A和B的指针
    A += BK;
    B += BK*N;
    __syncthreads();

#pragma unroll
    for(int m = 0; m<TM; m+=4){
        FLOAT4_VAL(a_frag[0][m]) = FLOAT4_VAL(As[0][OFFSET(0,ty+m,BM+4)]);
    }
#pragma unroll
    for(int n = 0; n<TN; n+=4){
        FLOAT4_VAL(b_frag[0][n]) = FLOAT4_VAL(Bs[0][OFFSET(0,tx+n,BN+4)]);
    }

    int write_index = 1, load_index = 0;

    int k = 0;
#pragma unroll
    do{
        k+=BK;
        /***
         * 循环内，先将global mem->register,
         * 这里接下来不->shared mem中，
         * 是因为指令依赖关系，先load到reg中，
         * 等计算Csub的时候再load到shared mem中，提升指令级并行，
         * 这样可以隐藏global mem的延迟
         */
        if(k<K){
#pragma unroll
            for(int i = 0; i<BM; i+=a_tile_stride){
                int reg_idx = i/a_tile_stride*4;
                FLOAT4_VAL(a_tile_reg[reg_idx]) = FLOAT4_VAL(A[OFFSET(a_tile_row+i, a_tile_col, K)]);
            }
#pragma unroll
            for(int i = 0; i<BK; i+=b_tile_stride){
                int reg_idx = i/move_iters_b*4;
                FLOAT4_VAL(b_tile_reg[reg_idx]) = \
                FLOAT4_VAL(B[OFFSET(b_tile_row+i, b_tile_col, N)]);
            }
            //搬运一个tile之后移动A和B的指针
            A += BK;
            B += BK*N;
        }

        //计算Csub
#pragma unroll
        for(int bk = 0; bk<BK-1; bk++){
#pragma unroll
            /***
             * 这里为什么要先load再计算？
             * load指令返回周期很长，
             * 所以要尽早发出，这样才能有效利用之后issue大量计算指令的周期，
             * 更好的掩盖load访存延迟
             */
            for(int m = 0; m<TM; m+=4){
                FLOAT4_VAL(a_frag[(bk+1)%2][m]) = FLOAT4_VAL(As[load_index][OFFSET(bk+1,ty+m,BM+4)]);
            }
#pragma unroll
            for(int n = 0; n<TN; n+=4){
                FLOAT4_VAL(b_frag[(bk+1)%2][n]) = FLOAT4_VAL(Bs[load_index][OFFSET(bk+1,tx+n,BN+4)]);
            }
#pragma unroll
            for(int j = 0; j<TM; j++){
#pragma unroll
                for(int l = 0; l<TN; l++){
                    Csub[j][l] += a_frag[(bk)%2][j]*b_frag[(bk)%2][l];
                }
            }
        }
        //这里为什么不直接用BK-1位置的元素计算？因为要尾迹补偿（epilogue computation）
        //，直接看循环最后


        if(k<K){
            /***
             * register->shared mem
             */
#pragma unroll
            for(int i = 0; i<BM; i+=a_tile_stride){
                int reg_idx = i/a_tile_stride*4;
                As[write_index][OFFSET(a_tile_col, i+a_tile_row, BM+4)] =\
                a_tile_reg[reg_idx];
                As[write_index][OFFSET(a_tile_col+1, i+a_tile_row, BM+4)] =\
                a_tile_reg[reg_idx+1];
                As[write_index][OFFSET(a_tile_col+2, i+a_tile_row, BM+4)] =\
                a_tile_reg[reg_idx+2];
                As[write_index][OFFSET(a_tile_col+3, i+a_tile_row, BM+4)] =\
                a_tile_reg[reg_idx+3];
            }
            for(int i=0; i<BK; i+=b_tile_stride){
                int reg_idx = i/move_iters_b*4;
                FLOAT4_VAL(Bs[write_index][OFFSET(b_tile_row+i, b_tile_col, BN+4)]) = \
                FLOAT4_VAL(b_tile_reg[reg_idx]);
            }
            __syncthreads();
            /***
             * shared_mem->register (仅一行/列) 为新一轮Csub计算做准备
             * 这里不
             */
            for(int m = 0; m<TM; m+=4){
                FLOAT4_VAL(a_frag[0][m]) = FLOAT4_VAL(As[write_index][OFFSET(0, ty+m, BM+4)]);
            }
            for(int n = 0; n<TN; n+=4){
                FLOAT4_VAL(b_frag[0][n]) = FLOAT4_VAL(Bs[write_index][OFFSET(0, tx+n, BN+4)]);
            }
            load_index = write_index;
            write_index ^= 1;
        }

        /***
         * epilogue computation,
         * 处理剩余循环项，
         * 掩盖 __syncthreads()（切换warp） 和 STS(ILP) 的延迟
         * 计算单元不能闲着
         */
        for(int j = 0; j<TM; j++){
            for(int l = 0; l<TN; l++){
                Csub[j][l]+=a_frag[(BK - 1) % 2][j]*b_frag[(BK - 1) % 2][l];
            }
        }
    
    }while(k<K);
#pragma unroll
    for(int i = 0; i<TM; i++){
        for(int j = 0; j<TN; j+=4){
            float4 tmp = FLOAT4_VAL(C[OFFSET(ty+i, tx+j, N)]);
            tmp.x = alpha*Csub[i][j]+beta*tmp.x;
            tmp.y = alpha*Csub[i][j+1]+beta*tmp.y;
            tmp.z = alpha*Csub[i][j+2]+beta*tmp.z;
            tmp.w = alpha*Csub[i][j+3]+beta*tmp.w;
            FLOAT4_VAL(C[OFFSET(ty+i, tx+j, N)]) = tmp;
        }
    }
}


torch::Tensor gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta, int version) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // 预设对齐的模板参数
    const int BM = 128, BN = 128, BK = 8; 
    const int TM = 8, TN = 8;

        // --- 新增的边界条件与 Shape 断言 ---
    // 1. 基础维度匹配断言
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "Input tensors must be 2 dimensions");
    TORCH_CHECK(B.size(0) == K, "Matrix multiply dimension mismatch: A columns != B rows");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Matrix C shape must be (M, N)");

    // 2. 强需求：因为 kernel 内部没有边界检查（为了 ncu profile 性能），必须约束 M, N, K 为分块大小的整数倍
    if (version > 0) {
        TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128 to match BM=128");
        TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128 to match BN=128");
        TORCH_CHECK(K % 8 == 0,   "K must be a multiple of 8 to match BK=8");
    }

    // 3. Float4 向量化约束 (version 2 和 3 用到了 float4 强制转换)
    if (version >= 2 && version <= 4) {
        // float4 要求连续读取 4 个 float，对 K 和 N 的维度做进一步对齐约束（虽然前面的 128 和 8 已经满足了，但这属于逻辑闭环）
        TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4 for A matrix float4 vectorization");
        TORCH_CHECK(N % 4 == 0, "N must be a multiple of 4 for B/C matrix float4 vectorization");
        
        // 地址 16字节 (4个float) 对齐断言，防止 float4 触发非法内存访问
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(A.data_ptr<float>()) % 16 == 0, "Tensor A pointer not 16-byte aligned");
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(B.data_ptr<float>()) % 16 == 0, "Tensor B pointer not 16-byte aligned");
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(C.data_ptr<float>()) % 16 == 0, "Tensor C pointer not 16-byte aligned");
    }

    dim3 block(BN / TN * BM / TM);
    dim3 grid(N / BN, M / BM); // 因为不考虑非对齐，直接除

    switch(version) {

        case 0: {
            dim3 grid_naive(CEIL(N, 32), CEIL(M, 32));
            gemm_naive_kernel<<<grid_naive, dim3(32, 32)>>>(M, N, K, alpha, beta, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()); 
            break;
            
        }
        case 1: gemm_tiled_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, beta, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()); break;
        case 2: gemm_tiled_float4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, beta, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()); break;
        case 3: gemm_pingpong_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, beta, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()); break;
        case 4: gemm_final_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, beta, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()); break;
        
        default: TORCH_CHECK(false, "Unsupported kernel version: ", version);
    }
    return C;
}
