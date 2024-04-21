import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # Transpose the keys matrix to align for the dot product
        keys_transposed = keys.transpose(1, 2)
        # Calculate the raw attention scores
        scores = torch.bmm(queries, keys_transposed) / math.sqrt(d)
        # Apply softmax to get probability-like weights
        attention_weights = F.softmax(scores, dim=-1)
        # Optionally apply dropout
        attention_weights = self.dropout(attention_weights)
        # Weighted sum of values based on the attention weights
        output = torch.bmm(attention_weights, values)
        return output, attention_weights

# Example dimensions
batch_size = 2
num_queries = 3
num_key_value_pairs = 4
dimension = 5

# Create random data for queries, keys, values
queries = torch.randn(batch_size, num_queries, dimension)
keys = torch.randn(batch_size, num_key_value_pairs, dimension)
values = torch.randn(batch_size, num_key_value_pairs, dimension)

# Create the attention layer
attention = DotProductAttention(dropout=0.1)

# Forward pass
output, attention_weights = attention(queries, keys, values)

print("Output:", output)
print("Attention Weights:", attention_weights)
