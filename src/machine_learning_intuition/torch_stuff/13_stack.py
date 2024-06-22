# torch.stack will append tensors along a specific dimensions. We see this used sometimes for
# multiheaded stuff in transformer impls

import torch

# Create two sample tensors of shape [32, 10, 16]
tensor1 = torch.randn(32, 10, 16)
tensor2 = torch.randn(32, 10, 16)

# Stack along dim=1
stacked_dim1 = torch.stack([tensor1, tensor2], dim=1)
print("Shape after stacking with dim=1:", stacked_dim1.shape)  # Expected shape: [32, 2, 10, 16]


# This will give us a grouping like:

'''
Batch 32  ┌────────────────────────────────────────────────────────────────────┐
          │    2 (New Dimension)                                               │
          │    ┌────────────────────────────────────────────────────────────┐  │
          │    │ Tensor 1 [10, 16]       Tensor 2 [10, 16]                  │  │
          │    └────────────────────────────────────────────────────────────┘  │
          └────────────────────────────────────────────────────────────────────┘
'''

# Stack along dim=-2
stacked_dim_neg2 = torch.stack([tensor1, tensor2], dim=-2)
print("Shape after stacking with dim=-2:", stacked_dim_neg2.shape)  # Expected shape: [32, 10, 2, 16]

'''
Here because we went with the second to the last dimension (-2) we get the new dimension in the 3rd position:

Batch 32 ┌──────────────────────────────────────────────────────────────────────┐
         │ Sequence Length 10                                                   │
         │ ┌──────────────────────────────────────────────────────────────────┐ │
         │ │ 2 (New Dimension)                                                │ │
         │ │ ┌──────────────────────────────┐   ┌────────────────────────────┐│ │
         │ │ │ Tensor 1 Features [16]       │   │ Tensor 2 Features [16]     ││ │
         │ │ └──────────────────────────────┘   └────────────────────────────┘│ │
         │ └──────────────────────────────────────────────────────────────────┘ │
         └──────────────────────────────────────────────────────────────────────┘



# For debugging and visualization
print("\nTensor shapes for debugging:")
print("tensor1 shape:", tensor1.shape)
print("tensor2 shape:", tensor2.shape)
print("stacked_dim1 shape:", stacked_dim1.shape)
print("stacked_dim_neg2 shape:", stacked_dim_neg2.shape)

