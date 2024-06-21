# this is an exerpt from: https://github.com/gsyyysg/StockFormer.git
# with a sample application applied for study

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout, activation):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.activation = activation
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))

class MultiheadFeedForward(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, activation):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mhfw = nn.ModuleList([FeedForward(d_model=self.head_dim, ff_dim=ff_dim, dropout=dropout, activation=activation) for _ in range(self.n_heads)])

    def forward(self, x): # [bs, seq_len, d_model]
        bs = x.shape[0]
        input = x.reshape(bs, -1, self.n_heads, self.head_dim) # [bs, seq_len, n_heads, head_dim]
        outputs = []
        for i in range(self.n_heads):
            outputs.append(self.mhfw[i](input[:, :, i, :])) # [bs, seq_len, head_dim]
        outputs = torch.stack(outputs, dim=-2).reshape(bs, -1, self.d_model) # [bs, seq_len, n_heads, head_dim]
        return outputs

# Example usage:
d_model = 128
n_heads = 8
ff_dim = 256
dropout = 0.1
activation = nn.ReLU()

model = MultiheadFeedForward(d_model, n_heads, ff_dim, dropout, activation)
x = torch.randn(32, 10, d_model) # [batch_size, seq_len, d_model]
output = model(x)
print(output.shape)  # Should be [32, 10, d_model]
