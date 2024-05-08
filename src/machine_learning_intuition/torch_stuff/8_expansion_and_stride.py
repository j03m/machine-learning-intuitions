import torch

'''
A tensor’s stride describes how many steps to take in memory to move to the next element along a particular dimension.

`tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])`

The shape of tensor is (2, 3), meaning it has 2 rows and 3 columns.
The stride is (3, 1), which means:
Moving to the next row requires stepping 3 positions in memory.
Moving to the next column requires stepping 1 position in memory.
'''

# Here, the stride is set to 0 for the expanded dimension,
# indicating that moving to the next position requires no additional
# steps in memory (as all positions point to the same value).
# Create a 1D tensor with shape (1,)
tensor = torch.tensor([1])
print("Original tensor shape:", tensor.shape)  # Shape: (1,)
print("Original tensor stride:", tensor.stride())  # Stride: (1,)
print(tensor)

# Expand the tensor to shape (4,)
expanded = tensor.expand(4)
print("Expanded tensor shape:", expanded.shape)  # Shape: (4,)
print("Expanded tensor stride:", expanded.stride())  # Stride: (0,)
print(expanded)  # All elements will appear as '1'


# Create a 2D tensor with a dimension of size 1
tensor = torch.tensor([[1], [2], [3]])
print("Original tensor shape:", tensor.shape)  # Shape: (3, 1)
print("Original tensor stride:", tensor.stride())  # Stride: (1, 1)

# Expand the second dimension
expanded = tensor.expand(3, 4)
print("Expanded tensor shape:", expanded.shape)  # Shape: (3, 4)
print("Expanded tensor stride:", expanded.stride())  # Stride: (1, 0)
print(expanded)


# Create two tensors with different shapes
tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([[1], [2], [3]])

print("a shape:", tensor_a.shape)
print("b shape:", tensor_b.shape)

# Expand tensor_b to match the shape of tensor_a
expanded_b = tensor_b.expand(3, 3)
print("Expanded tensor_b shape:", expanded_b.shape)  # Shape: (3, 3)
print("b post expand:", expanded_b)

# Broadcasting addition
result = tensor_a + expanded_b
print("Result after broadcasting addition:\n", result)

# unfortunately you can't just expand non-singleton dimensions - this will fail:
pe = torch.zeros(5000, 250)
try:
    pe.expand(5000, 11, 250)
except RuntimeError as e:
    print("nope:", e)

# but you can add dimensions with unsqueeze
pe = pe.unsqueeze(1)
pe = pe.expand(5000, 11, 250)
print("got it: ", pe.shape)
