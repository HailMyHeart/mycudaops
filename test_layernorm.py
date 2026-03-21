import torch
import my_lib

def verify_layernorm():
    # 测试维度，例如 M个序列，N为特征维度
    M, D = 1024, 4096 
    eps = 1e-5

    # 随机生成测试数据
    input_tensor = torch.randn(M, D, device='cuda', dtype=torch.float32)
    gamma = torch.randn(D, device='cuda', dtype=torch.float32)
    beta = torch.randn(D, device='cuda', dtype=torch.float32)

    # 1. PyTorch 官方 LayerNorm
    # PyTorch 的 LayerNorm 直接支持 gamma(weight) 和 beta(bias)
    ln = torch.nn.LayerNorm(D, eps=eps, elementwise_affine=True).cuda()
    ln.weight.data = gamma
    ln.bias.data = beta
    with torch.no_grad():
        out_torch = ln(input_tensor)

    # 2. 自定义内核 LayerNorm
    out_custom = my_lib.layernorm(input_tensor, gamma, beta, eps)

    # 3. 对比结果
    diff = torch.abs(out_torch - out_custom).max().item()
    print(f"Max absolute difference: {diff}")
    
    if diff < 1e-4:
        print("LayerNorm Kernel PASSED!")
    else:
        print("LayerNorm Kernel FAILED!")

if __name__ == "__main__":
    verify_layernorm()