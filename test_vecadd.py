import torch
from my_lib import vec_add
import time

def profile_vec_add(N = 2**26): #  256MB
    print(f"--- Testing VecAdd (N={N}) ---")
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10): vec_add(a, b)
    
    # 测量 Custom VecAdd (float4)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        c_custom = vec_add(a, b)
    end.record()
    torch.cuda.synchronize()
    
    custom_ms = start.elapsed_time(end) / 100
    
    # 测量 PyTorch 原生 (+)
    start.record()
    for _ in range(100):
        c_torch = a + b
    end.record()
    torch.cuda.synchronize()
    
    torch_ms = start.elapsed_time(end) / 100
    
    # 带宽计算: 2读1写 (3 * N * 4 bytes)
    total_gb = (N * 4 * 3) / (1024**3)
    custom_bw = total_gb / (custom_ms / 1000)
    torch_bw = total_gb / (torch_ms / 1000)
    
    print(f"Custom (float4) Time: {custom_ms:.4f} ms | Bandwidth: {custom_bw:.2f} GB/s")
    print(f"PyTorch (+)     Time: {torch_ms:.4f} ms | Bandwidth: {torch_bw:.2f} GB/s")
    
    # 精度检查
    torch.testing.assert_close(c_custom, c_torch)
    print("✅ Correctness Verified!")

if __name__ == "__main__":
    profile_vec_add()