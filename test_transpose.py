# test_transpose.py
import torch
import my_lib  # 假设你的编译模块叫 my_lib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="naive")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    args = parser.parse_args()

    # 创建数据 (4090 上建议用 FP32 测带宽极限)
    input_tensor = torch.randn(args.m, args.n, device='cuda', dtype=torch.float32)
    
    # Warmup: 运行 10 次不计入统计
    for _ in range(10):
        if args.mode == "naive":
            output = my_lib.transpose_naive(input_tensor)
        elif args.mode == "shared_p0":
            output = my_lib.transpose_shared(input_tensor)
        elif args.mode == "shared_p1":
            output = my_lib.transpose_shared_padding(input_tensor)
        elif args.mode == "swizzle":
            output = my_lib.transpose_shared_swizzle(input_tensor)
    
    torch.cuda.synchronize()

    # 实际被 NCU 捕获的那一次
    if args.mode == "naive":
        output = my_lib.transpose_naive(input_tensor)
    elif args.mode == "shared_p0":
        output = my_lib.transpose_shared(input_tensor)
    elif args.mode == "shared_p1":
        output = my_lib.transpose_shared_padding(input_tensor)
    elif args.mode == "swizzle":
        output = my_lib.transpose_shared_swizzle(input_tensor)
        
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()