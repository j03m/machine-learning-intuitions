import torch
mask = torch.arange(10)
print("0-10: ", mask, "shape: ", mask.shape)

# Using [None] - 1, row 10 columns
mask_1 = mask[None]
print("Using [None]: ", mask_1, "Shape: ", mask_1.shape)

# Using [None, :] - same thing - 1, row 10 columns
mask_2 = mask[None, :]
print("Using [None, :]: ", mask_2, "Shape: ", mask_2.shape)


# Using [:, None] - now we get 10 rows of 1 column
mask_3 = mask[:, None]
print("Using [:, None]: ", mask_3, "Shape: ", mask_3.shape)

