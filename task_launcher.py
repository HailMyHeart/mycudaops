import torch
import my_lib 
import sys

if __name__ == "__main__":
    M, N, K, v_id = map(int, sys.argv[1:])
    # 统一使用 float32 类型
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    # 为 C 预分配显存空间
    C = torch.zeros((M, N), device='cuda', dtype=torch.float32)

    # 预热循环：跑 10 次
    # NCU 会看到这些启动，但我们会告诉它跳过
    warmup_iters = 10
    for _ in range(warmup_iters):
        # 增加传入 C
        my_lib.gemm_forward(A, B, C, 1.0, 0.0, v_id)

    # 第 11 次：这是我们要采集的目标
    my_lib.gemm_forward(A, B, C, 1.0, 0.0, v_id)