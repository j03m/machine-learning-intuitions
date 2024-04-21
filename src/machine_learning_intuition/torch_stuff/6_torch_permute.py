import torch
A = torch.arange(2*3*4).reshape(2, 3, 4)  # Random tensor of shape [2, 3, 4]
print(A)
B = A.permute(1, 2, 0)    # Re-arrange to [3, 4, 2]
print(B)
