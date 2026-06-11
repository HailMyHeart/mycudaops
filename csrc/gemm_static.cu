
#include <cuda_runtime.h>
#include <torch/extension.h>
#define CEIL(a, b) ((b)+(a)-1)/(b)
#define FLOAT4_VAL(a) (reinterpret_cast<float4*>(&(a))[0])
#definie OFFSET(i, j, n) ((i)*(n)+(j))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm(int M, int N, int K, const float alpha, const float beta, float* A, float* B, float* C){
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    
    __shared__ float As[2][BM*BK];
    __shared__ float Bs[2][BN*BK];

    A = A+OFFSET(BM*blockIdx.y,0, K);
    B = B+OFFSET(0, BN*blockIdx.x, N);
    C = C+OFFSET(blockIdx.y*BM, blockIdx.x*BN, N);


    int a_tile_row = threadIdx.x/(BK/4);
    int a_tile_col = threadIdx.x%(BK/4)*4;
    int b_tile_row = threadIdx.x/(BN/4);
    int b_tile_col = threadIdx.x%(BN/4)*4;

    float Csub[TM][TN]; = {0.f};
    float a_tile_reg[4] = {0.f};
    float b_tile_reg[4] = {0.f};

    float a_frag[2][TM], b_frag[2][TN];

    const int warp_id = threadIdx.x/32;
    const int lane_id = threadIdx.x%32;


    int ty_base = warp_id/2*16+lane_id/8*4;
    int tx_base = warp_id%2*32+lane_id%8*4;

    FLOAT4_VAL(a_tile_reg[0]) = FLOAT4_VAL(A[OFFSET(a_tile_row, a_tile_col, K)]);
    As[0][OFFSET(a_tile_col, a_tile_row, BM)] = a_tile_reg[0];
    As[0][OFFSET(a_tile_col+1, a_tile_row, BM)] = a_tile_reg[1];
    As[0][OFFSET(a_tile_col+2, a_tile_row, BM)] = a_tile_reg[2];
    As[0][OFFSET(a_tile_col+3, a_tile_row, BM)] = a_tile_reg[3];

    FLOAT4_VAL(b_tile_reg[0]) = FLOAT4_VAL(B[OFFSET(b_tile_row, b_tile_col, N)]);

    A+=BK; B+=BK*N;
    __syncthreads();

    FLOAT4_VAL(a_frag[0][0]) = FLOAT4_VAL(As[0][OFFSET(0, ty_base, BM)]);
    FLOAT4_VAL(a_frag[0][4]) = FLOAT4_VAL(As[0][OFFSET(0, ty_base+64, BM)]);
    FLOAT4_VAL(b_frag[0][0]) = FLOAT4_VAL(Bs[0][OFFSET(0, tx_base, BN)]);
    FLOAT4_VAL(b_frag[0][4]) = FLOAT4_VAL(Bs[0][OFFSET(0, tx_base+64, BN)]);

    int write_idx = 1, load_idx = 0;
    int k = 0;

    do{
        k+=BK;
        if(k<K){
            FLOAT4_VAL(a_tile_reg[0]) = FLOAT4_VAL(A[OFFSET(a_tile_row, a_tile_col, K)]);
            FLOAT4_VAL(b_tile_reg[0]) = FLOAT4_VAL(B[OFFSET(b_tile_row, b_tile_col, N)]);
            A+=BK; B+=BK*N;;
        }
        for(int bk = 0; bk<BK-1; bk++){
            int nextbk = bk+1;
            FLOAT4_VAL(a_frag[(bk+1)%2][0]) = FLOAT4_VAL(As[load_idx][OFFSET(nextbk, ty_base, BM)]);
            FLOAT4_VAL(a_frag[(bk+1)%2][4]) = FLOAT4_VAL(As[load_idx][OFFSET(nextbk, ty_base+64, BM)]);
            FLOAT4_VAL(b_frag[(bk+1)%2][0]) = FLOAT4_VAL(Bs[load_idx][OFFSET(nextbk, tx_base, BN)]);
            FLOAT4_VAL(b_frag[(bk+1)%2][4]) = FLOAT4_VAL(Bs[load_idx][OFFSET(nextbk, tx_base+64, BN)]);
            load_idx = write_idx;
            
            for(int j = 0; j<TM; j++){
                for(int l = 0; l<TN; l++){
                    Csub[j][l] += a_frag[(bk)%2][j]*b_frag[bk%2][l];
                }
            }
        }

        if(k<K){
            As[write_idx][OFFSET(a_tile_col, a_tile_row, BM)] = a_tile_reg[0];
            As[write_idx][OFFSET(a_tile_col+1, a_tile_row, BM)] = a_tile_reg[1];
            As[write_idx][OFFSET(a_tile_col+2, a_tile_row, BM)] = a_tile_reg[2];
            As[write_idx][OFFSET(a_tile_col+3, a_tile_row, BM)] = a_tile_reg[3];
            FLOAT4_VAL(Bs[write_idx][OFFSET(b_tile_row, b_tile_col, BN)]) = FLOAT4_VAL(b_tile_reg[0]);
            __syncthreads();

            FLOAT4_VAL(a_frag[0][0]) = FLOAT4_VAL(As[write_idx][OFFSET(0, ty_base, BM)]);
            FLOAT4_VAL(a_frag[0][4]) = FLOAT4_VAL(As[write_idx][OFFSET(0, ty_base+64, BM)]);
            FLOAT4_VAL(b_frag[0][0]) = FLOAT4_VAL(Bs[write_idx][OFFSET(0, tx_base, BN)]);
            FLOAT4_VAL(b_frag[0][4]) = FLOAT4_VAL(Bs[write_idx][OFFSET(0, tx_base+64, BN)]);
        
            write_idx = 1-write_idx;
        }

        for(int j = 0; j<TM; j++){
            for(int l = 0; l<TN; l++){
                Csub[j][l]+=a_frag[1][j]*b_frag[1][l];
            }
        }
    }while(k<K);

    for(int i = 0; i<TM/2; i++){
        float4 tmp = FLOAT4_VAL(C[OFFSET(ty_base+i, tx_base, N)]);
        tmp.x = alpha*Csub[i][0]+beta*tmp.x;
        tmp.y = alpha*Csub[i][1]+beta*tmp.y;
        tmp.z = alpha*Csub[i][2]+beta*tmp.z;
        tmp.w = alpha*Csub[i][3]+beta*tmp.w;
        FLOAT4_VAL(C[OFFSET(ty_base+i, tx_base, N)]) = tmp;
    }

    for(int i = 0; i<TM/2; i++){
        float4 tmp = FLOAT4_VAL(C[OFFSET(ty_base+i, tx_base+64, N)]);
        tmp.x = alpha*Csub[i][4]+beta*tmp.x;
        tmp.y = alpha*Csub[i][5]+beta*tmp.y;
        tmp.z = alpha*Csub[i][6]+beta*tmp.z;
        tmp.w = alpha*Csub[i][7]+beta*tmp.w;
        FLOAT4_VAL(C[OFFSET(ty_base+i, tx_base+64, N)]) = tmp;
    }

    for(int i = 0; i<TM/2; i++){
        float4 tmp = FLOAT4_VAL(C[OFFSET(ty_base+i+64, tx_base, N)]);
        tmp.x = alpha*Csub[i+4][0]+beta*tmp.x;
        tmp.y = alpha*Csub[i+4][1]+beta*tmp.y;
        tmp.z = alpha*Csub[i+4][2]+beta*tmp.z;
        tmp.w = alpha*Csub[i+4][3]+beta*tmp.w;
        FLOAT4_VAL(C[OFFSET(ty_base+i+64, tx_base, N)]) = tmp;
    }

    for(int i = 0; i<TM/2; i++){
        float4 tmp = FLOAT4_VAL(C[OFFSET(ty_base+i+64, tx_base+64, N)]);
        tmp.x = alpha*Csub[i+4][4]+beta*tmp.x;
        tmp.y = alpha*Csub[i+4][5]+beta*tmp.y;
        tmp.z = alpha*Csub[i+4][6]+beta*tmp.z;
        tmp.w = alpha*Csub[i+4][7]+beta*tmp.w;
        FLOAT4_VAL(C[OFFSET(ty_base+i+64, tx_base+64, N)]) = tmp;
    }

}
