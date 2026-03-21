import torch
import my_lib
import sys

M = int(sys.argv[1])
D = int(sys.argv[2])

input_tensor = torch.randn(M, D, device='cuda', dtype=torch.float32)
gamma = torch.randn(D, device='cuda', dtype=torch.float32)
beta = torch.randn(D, device='cuda', dtype=torch.float32)

for _ in range(5):
    my_lib.layernorm(input_tensor, gamma, beta, 1e-5)

torch.cuda.synchronize()
my_lib.layernorm(input_tensor, gamma, beta, 1e-5)
torch.cuda.synchronize()
