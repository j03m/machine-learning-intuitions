'''

This isn't really torch perse but I put it here anyway

broadcasting is what happens when we operate on matrices with different dimensions.

Any operator works (+-*/)

The rules are:

If the tensors have a different number of dimensions, prepend 1s to the shape of the smaller tensor until both have the same number of dimensions.
 * Starting from the last dimension and moving backward:
 * If the sizes are the same or one of the dimensions is 1, the operation is allowed.
 * Otherwise, the operation will fail.

'''


import numpy as np

# Arrays with compatible dimensions - note the 1's are replaced with the larger dimensions
a = np.ones((3, 1, 5))  # Shape (3, 1, 5)
b = np.ones((1, 4, 5))  # Shape (1, 4, 5)

# Broadcasting operation
result = a + b
print("Result shape:", result.shape)  # Should be (3, 4, 5)


# this one will fail
import numpy as np

# Arrays with incompatible dimensions
a = np.ones((2, 3))  # Shape (2, 3)
b = np.ones((3, 3))  # Shape (3, 3)

try:
    # This will raise an error
    result = a + b
except ValueError as e:
    print("Broadcasting error:", e)


# another error, subtle incompat:

a = np.ones((4, 1, 3))  # Shape (4, 1, 3)
b = np.ones((2, 4, 1))  # Shape (2, 4, 1)

try:
    # This will raise an error
    result = a * b
except ValueError as e:
    print("Broadcasting error:", e)

# change 4 to 2 and it will work tho:


a = np.ones((2, 1, 3))  # Shape (2, 1, 3)
b = np.ones((2, 4, 1))  # Shape (2, 4, 1)
result = a + b
print("All better: ", result.shape)
print("a:", a)
print("+ b:", b)
print("= c:", result)


# Arrays with dimensions where both have a "1"
a = np.ones((5, 1))     # Shape (5, 1)
b = np.ones((1, 6))     # Shape (1, 6)

# Broadcasting operation
result = a * b
print("Result shape:", result.shape)


# Arrays with dimensions where both have a "1"
a = np.ones((5, 6))     # Shape (5, 6)
b = np.ones((1, 6))     # Shape (1, 6)

# Broadcasting operation
result = a * b
print("Result shape:", result.shape)  # Should be (5, 6)


# pre-pending rule in action:

import torch

# Example tensors - when I first learned about this stuff I thought this would fail,
# but it doesn't because we prepend 1 - so it becomes (500, 1) * (1, 125)
tensor_a = torch.randn(500, 1)
tensor_b = torch.randn(125)

# Element-wise multiplication
result = tensor_a * tensor_b

# The resulting shape will be (500, 125)
print(result.shape)  # Output: torch.Size([500, 125])


# 10, 1, 5
a = torch.arange(50).reshape(10, 1, 5)
b = torch.arange(50).reshape(10, 5, 1)
c = a + b
print(c.shape)