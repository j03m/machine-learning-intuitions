import torch
import torch.nn.functional as F

# Example data
short_term_features = torch.tensor([  [100, 102, 101, 103, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112]]).float()
long_term_features = torch.tensor([[104, 108]]).float()

# Example dimensions
B, N, D = 1, 2, 7  # Batch size, Number of features, Feature dimension


class SimpleAttention(torch.nn.Module):
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()
        self.scale = dim ** -0.5
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = self.softmax(scores)
        attended_values = torch.matmul(attn_weights, value)
        return attended_values, attn_weights

# Create attention module
attention = SimpleAttention(dim=D)

# Reshape data to (B, N, D) for attention
short_term_features = short_term_features.view(B, -1, D)
long_term_features = long_term_features.view(B, -1, D)

# Apply attention: using short-term as query, long-term as key and value
attended_features, attn_weights = attention(short_term_features, long_term_features, long_term_features)

print("Attended Features:", attended_features)
print("Attention Weights:", attn_weights)
