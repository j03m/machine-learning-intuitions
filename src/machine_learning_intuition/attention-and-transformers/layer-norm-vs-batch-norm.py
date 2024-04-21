import torch
import torch.nn as nn

# Create a tensor from a range of numbers and reshape it
import torch

# Assuming num_batches, num_rows, and num_features are predefined
num_batches = 4  # Example value
num_features = 3  # Example value
num_rows = 5  # Example value

# Create a tensor X
X = torch.arange(1, num_batches * num_rows * num_features + 1, dtype=torch.float32).reshape((num_batches, num_features, num_rows))
print("Original Tensor X:\n", X)


# Batch Normalization
# Apply along the (N, L) dimension for each feature in C from (N, L, C)
bn = nn.BatchNorm1d(num_features)
bn.eval()  # Use evaluation mode which uses running estimates for mean/variance
X_bn = bn(X)

# Layer Normalization
# Apply across each feature vector in each sample
ln = nn.LayerNorm(X.size()[1:])
ln.eval()  # Evaluation mode
X_ln = ln(X)

print("Batch Normalized X:\n", X_bn)
print("Layer Normalized X:\n", X_ln)

# Calculate means and std deviations for Batch Norm (across batches and samples for each feature)
mean_bn = torch.mean(X, dim=(0, 1), keepdim=True)
var_bn = torch.var(X, dim=(0, 1))

mean_ln = torch.mean(X, dim=2, keepdim=True)
var_ln = torch.var(X, dim=2)

print("batch mean and var:", mean_bn, var_bn)
print("layer mean and var:", mean_ln, var_ln)
