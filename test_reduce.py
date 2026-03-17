import torch
# 之前的写法 `import my_lib.reduce_sum as reduce_sum` 会把 Python 当作包/子模块导入。
# 这里 my_lib 是一个 C++ 扩展模块（module），绑定了函数 `reduce_sum`。
# 正确的导入方式是直接从扩展模块导入属性：
from my_lib import reduce_sum
import time

def benchmark(N=1 << 30, threads=1024, blocks=2048):
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    
    # 1. 预热
    for _ in range(20):
        _ = reduce_sum(x, threads, blocks)
    torch.cuda.synchronize()

    # 2. 计时
    iters = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        _ = reduce_sum(x, threads, blocks)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_time_ms = start_event.elapsed_time(end_event) / iters
    avg_time_s = avg_time_ms / 1000

    # 3. 结果验证
    custom_res = reduce_sum(x, threads, blocks)
    torch_res = x.sum()
    print(f"Check: {'PASSED' if torch.allclose(custom_res, torch_res.view_as(custom_res), atol=1e-2) else 'FAILED'}")

    # 4. 指标计算
    # 带宽利用率 (4090 峰值按 1008 GB/s 计)
    bandwidth_gbps = (N * 4) / (avg_time_s * 1024**3)
    bw_util = (bandwidth_gbps / 1008) * 100
    
    # 算力利用率 (4090 FP32 峰值按 82.6 TFLOPS = 82600 GFLOPS 计)
    gflops = N / (avg_time_s * 1e9)
    comp_util = (gflops / 82600) * 100

    print("-" * 30)
    print(f"Config: Threads={threads}, Blocks={blocks}")
    print(f"Latency: {avg_time_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s (Util: {bw_util:.2f}%)")
    print(f"Compute: {gflops:.2f} GFLOPS (Util: {comp_util:.4f}%)")

if __name__ == "__main__":
    benchmark()