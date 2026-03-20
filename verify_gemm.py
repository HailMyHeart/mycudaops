import torch
import my_lib  # 你的 pybind11 模块
import numpy as np

def verify(M, N, K, alpha=1.0, beta=0.0):
    print(f"Verification for Shape: {M}x{N}x{K}, alpha={alpha}, beta={beta}")
    
    # 1. 准备随机数据
    # 使用 float32，注意误差积累
    device = torch.device("cuda")
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)
    C_initial = torch.randn((M, N), device=device, dtype=torch.float32)

    # 2. 计算参考答案 (cuBLAS)
    # PyTorch 的 addmm 实现: out = beta * input + alpha * (mat1 @ mat2)
    ref_C = torch.addmm(C_initial, A, B, beta=beta, alpha=alpha)

    # 3. 验证四个版本
    versions = ["Naive", "Tiled", "Float4", "PingPong"]
    all_passed = True

    for v_id, name in enumerate(versions):
        # 深度拷贝 C_initial，确保每个 kernel 面对的初始 C 是一样的
        test_C = C_initial.clone()
        
        # 调用你的 CUDA 算子
        # 结果直接原地更新在 test_C 中 (或者根据你的 wrapper 逻辑返回)
        output_C = my_lib.gemm_forward(A, B, test_C,  alpha, beta, v_id)

        # 4. 误差检查
        # 由于浮点数累加顺序不同，会有微小误差。使用 allclose 检查。
        # rtol: 相对误差, atol: 绝对误差
        is_close = torch.allclose(output_C, ref_C, rtol=1e-3, atol=1e-3)
        
        max_diff = torch.max(torch.abs(output_C - ref_C)).item()
        
        if is_close:
            print(f"  [v] Version {v_id} ({name:10}): PASSED (Max Diff: {max_diff:.6f})")
        else:
            print(f"  [x] Version {v_id} ({name:10}): FAILED!")
            print(f"      Max Diff: {max_diff:.6f}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    # 必须是 Tile 的倍数，比如 BM=128, BN=128, BK=8/16
    # 建议先从较小的 Shape 开始 debug
    test_shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024)
    ]
    
    for M, N, K in test_shapes:
        if not verify(M, N, K, alpha=1.5, beta=0.5):
            print("\nCritical Error: Some kernels failed verification!")
            break
    else:
        print("\nAll kernels passed verification for all shapes!")