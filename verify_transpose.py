import torch
import my_lib # 你的编译模块
import numpy as np

def verify(m, n, mode):
    # 1. 准备数据 (使用 float32)
    # 建议使用递增序列而非随机数，这样报错时更容易看清索引哪里错了
    input_tensor = torch.arange(m * n, device='cuda', dtype=torch.float32).view(m, n)
    expected = input_tensor.t().contiguous()
    
    # 2. 调用你的算子
    try:
        if mode == "naive":
            output = my_lib.transpose_naive(input_tensor)
        elif mode == "shared_p0":
            output = my_lib.transpose_shared(input_tensor)
        elif mode == "shared_p1":
            output = my_lib.transpose_shared_padding(input_tensor) # 对应 padding kernel
        elif mode == "swizzle":
            output = my_lib.transpose_shared_padding(input_tensor)
        
        # 3. 结果比对
        # rtol/atol 对 FP32 精度要求较高
        if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
            print(f"✅ [PASS] Mode: {mode:<10} Shape: {m}x{n}")
            return True
        else:
            print(f"❌ [FAIL] Mode: {mode:<10} Shape: {m}x{n}")
            # 打印前几个不一致的元素看看规律
            mask = ~torch.isclose(output, expected)
            print(f"First 5 mismatches:\nOutput:   {output[mask][:5]}\nExpected: {expected[mask][:5]}")
            return False
            
    except Exception as e:
        print(f"💥 [ERROR] Mode: {mode:<10} Shape: {m}x{n} -> {str(e)}")
        return False

def main():
    # 测试三种典型场景
    test_shapes = [
        (128, 128),   # 小规模对齐
        (8192, 8192), # 大规模对齐
        (8191, 8191), # 非对齐 (越界测试关键!)
        (32, 1024),   # 长方形
        (1024, 32)    # 宽方形
    ]
    
    modes = ["naive", "shared_p0", "shared_p1", "swizzle"]
    
    print("🚀 开始算子正确性验证...\n")
    for m, n in test_shapes:
        for mode in modes:
            verify(m, n, mode)
        print("-" * 40)

if __name__ == "__main__":
    main()