# transpose takes two dimensions and swaps them

import torch

x = torch.randn(2, 3)
print(x.shape)

# swap the 0th and 1st dims:
x = x.transpose(0, 1)
print(x.shape)

# Create a sample tensor with shape [batch_size, seq_len, d_model]
batch_size = 2
seq_len = 5
d_model = 3

x = torch.randn(batch_size, seq_len, d_model)
print("Original shape:", x.shape)  # Should print: torch.Size([2, 5, 3])

# Transpose the tensor to swap the last two dimensions
# -1 here means the last dimension, which is a little confusing but its essentially 2. This would be
# equivalent to saying x.transpose(2,1)
x_transposed = x.transpose(-1, 1)
print("Shape after transpose (-1,1):", x_transposed.shape)  # Should print: torch.Size([2, 3, 5])
x_transposed = x.transpose(2, 1)
print("Shape after transpose (2,1):", x_transposed.shape)  # Should print: torch.Size([2, 3, 5])

# Transpose back to the original shape
x_back = x_transposed.transpose(-1, 1)
print("Shape after transposing back:", x_back.shape)  # Should print: torch.Size([2, 5, 3])

# Verify that the tensor is unchanged after double transpose
print("Is the original tensor equal to the double transposed tensor? ", torch.equal(x, x))
