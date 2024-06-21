import torch

# Define two vectors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# The letters here don't matter; they are simply labels. The order of what they represent matters.
# Here, we label the 1st dimension of both tensors as 'i'. Then we get the product expressed as a scalar sum.
out = torch.einsum('i,i->', a, b)
print(f"scalar:", out)

# Here, we label the 1st dimension of both tensors as 'i'. Then we get the element-wise product,
# keeping the same dimensions.
out = torch.einsum('i,i->i', a, b)
print(f"element wise multiplication to the same dims:", out)


# Attempting to perform einsum with mismatched dimensions will result in an error.
try:
    out = torch.einsum('ij,i->i', a, b)
except RuntimeError as e:
    print(f"invalid dimensions will crash: ", e)

# Define two 2D tensors
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])


# These are equivalent because 'i' and 'j' are just labels for the 0th and 1st dimensions.
# Both notations perform element-wise multiplication and sum over the 1st dimension (columns) for each 0th dimension
# (row).
result1 = torch.einsum('ij,ij->i', a, b)
result2 = torch.einsum('ji,ji->j', a, b)
print("Same output: ", result1, "==", result2, " in value")

# To operate on the columns, we can change the labeling. Here, we sum over the 1st dimension for each
# column.
result = torch.einsum('ij,ij->j', a, b)
print("second dimension dot product:", result)

# Standard matrix multiplication can also be expressed using einsum:
# Here, we say that the rows of a (i) and columns of a (k) should be
# multiplied by the rows of b (k) and the columns of b (j) to produce a final
# shape of i,j. In this case, the resulting shape would be (3, 2).
# Because there are 2 columns in a and 2 rows in b, this works.
a = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
b = torch.tensor([[7, 8], [9, 10]])         # Shape (2, 2)
result = torch.einsum('ik,kj->ij', a, b)
print("Matrix multiplication result:\n", result)

# Batch Matrix Multiplication
a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = torch.tensor([[[1, 0], [0, 1]], [[1, 2], [3, 4]]])
result = torch.einsum('bij,bjk->bik', a, b)
print("Batch matrix multiplication result:\n", result)

# Transpose of a Matrix
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = torch.einsum('ij->ji', a)
print("Transpose of the matrix:\n", result)

# Sum Across Specific Axes
a = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])
result = torch.einsum('ijk->', a)  # Sum all elements
print("Sum of all elements:", result)

result = torch.einsum('ijk->j', a)  # Sum over the first and third dimensions
print("Sum over first and third dimensions:", result)

# Diagonals of a Matrix
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.einsum('ii->i', a)
print("Diagonal elements of the matrix:", result)