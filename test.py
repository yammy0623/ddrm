import torch

H = torch.randn(1024, 1024).cuda()  # 測試大矩陣的 SVD
U, S, V = torch.svd(H, some=False)
print(U, S, V)
