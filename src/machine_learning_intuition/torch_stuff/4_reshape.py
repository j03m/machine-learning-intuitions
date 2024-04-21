import torch

# Create a tensor of shape (4, 4)
x = torch.arange(16).reshape(4, 4)
print("Original Tensor:\n", x)

# Reshape it to (2, 8)
y = x.reshape(2, 8)
print("Reshaped Tensor (2, 8):\n", y)

# Here, reshape(2, -1) tells PyTorch:
# "I want 2 rows, and you figure out how many columns are needed."
# Since there are 16 elements in total, PyTorch calculates that there should be 8 columns.
z = x.reshape(2, -1)
print("Reshaped Tensor with -1 (2, -1):\n", z)

# Create a tensor of shape (2, 3, 4)
x = torch.arange(24).reshape(2, 3, 4)
print("Original Tensor (2, 3, 4):\n", x)

# Reshape it to (6, -1)
# pyTorch calculates the number of rows to be 6 and automatically figures out that the
# columns should be 4 (since 6x4 = 24, the total number of elements).
y = x.reshape(6, -1)
print("Reshaped Tensor (6, -1):\n", y)

# Try another reshape
# calculates how many rows are needed to make sure each row has 2 columns.
# since there are 24 elements, there must be 12 row
z = x.reshape(-1, 2)
print("Reshaped Tensor (-1, 2):\n", z)

# only one dimension can be inferred! will crash
try:
    z1 = x.reshape(-1, 4, -1)
except Exception as e:
    print("Error: ", e)


z1 = x.reshape(2, 4, -1)
print("Inferred 3rd dim:\n", z1)