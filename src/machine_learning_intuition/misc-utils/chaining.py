import itertools
import torch.nn as nn
import torch

# Example neural networks
class NetworkA(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(2, 2))

class NetworkB(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(2, 2))

# Instantiate networks
netA = NetworkA()
netB = NetworkB()

# Chain parameters from both networks
all_params = itertools.chain(netA.parameters(), netB.parameters())

# Example operation: zero gradients (common before a backpropagation step)
for param in all_params:
    print(param)
    param.grad = None  # Normally, you'd use param.grad.zero_() in actual code
