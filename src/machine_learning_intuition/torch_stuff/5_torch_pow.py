import torch

# generate a range of 0-10 stepping 2s, reshape it
# raise 2 to the power of each value in the tensor
x = torch.pow(2, torch.arange(0, 11, 2, dtype=torch.float32).reshape(-1, 2))
