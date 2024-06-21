import torch
max_len = 1000
model_size = 250
model_size = 250
features = 11

X = torch.arange(max_len).reshape(-1, 1)
div_tensor = torch.pow(10000, torch.arange(0, model_size, 2).float() / model_size)
X_ = X / div_tensor
P = torch.zeros(max_len, model_size)
P[:, 0::2] = torch.sin(X_)
P[:, 1::2] = torch.cos(X_)
P_ = P.unsqueeze(2)
P__ = P_.repeat(1, 1, features)
