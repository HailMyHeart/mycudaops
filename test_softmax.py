import torch
from my_lib import softmax
import time

def test_softmax_correctness(M=128, K=2048):
    print(f"Testing Correctness: Shape({M}, {K})...")
    # 构造数据
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    
    # 获取结果
    res_custom = softmax(x)
    res_torch = torch.softmax(x, dim=-1)
    
    # 验证精度 (atol/rtol 设为 1e-5)
    try:
        torch.testing.assert_close(res_custom, res_torch, atol=1e-5, rtol=1e-5)
        print("✅ Correctness Check Passed!")
    except Exception as e:
        print(f"❌ Correctness Check Failed!\n{e}")

def profile_softmax_full_report(M=16384, K=4096, threads=1024):
    x = torch.randn(M, K, device='cuda')
    # 4090 理论峰值 (FP32)
    PEAK_TFLOPS = 82.58 
    
    # Warmup
    for _ in range(20): softmax(x, threads=threads)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        softmax(x, threads=threads)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_ms = start_event.elapsed_time(end_event) / 100
    avg_sec = avg_ms / 1000.0
    
    # 1. 带宽计算
    data_size_gb = (M * K * 4) / (1024**3)
    # 由于 online softmax 需要读写两次（输入和输出），所以乘以 2(没有L2优化的情况下，是3次)
    effective_bw = (data_size_gb * 2) / avg_sec
    
    # 2. 算力计算 (每个元素 8 FLOPs)
    total_flops = M * K * 8
    tflops = (total_flops / 1e12) / avg_sec
    utilization = (tflops / PEAK_TFLOPS) * 100
    
    print(f"--- 4090 综合性能报告 ---")
    print(f"平均耗时: {avg_ms:.4f} ms")
    print(f"算法有效带宽: {effective_bw:.2f} GB/s")
    print(f"实际算力吞吐: {tflops:.2f} TFLOPS")
    print(f"算力利用率:   {utilization:.2f}%")
    
    if utilization < 5:
        print(f"💡 分析: 算力利用率极低，确认该算子为 [访存受限型]。")
    print(f"--------------------------")

if __name__ == "__main__":
    # 1. 验证各种大小 (验证你之前提到的 lane 判断是否生效)
    test_softmax_correctness(M=1, K=1024)   # 单行测试
    test_softmax_correctness(M=64, K=333)   # 非对齐长度测试
    
    # 2. 性能测试
    profile_softmax_full_report(M=16384, K=4096, threads=1024)