# from http://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding
import torch

max_len = 5000
hidden = 250
features = 11
# we make 5000 rows of 1 column
# then we make expand this out by the pow tensor which is 125,
# which explodes out to 5000 rows and 125 columns
X = torch.arange(max_len).reshape(-1, 1)
div_tensor = torch.arange(0, hidden, 2)
div_tensor_dv = div_tensor / hidden
div_tensor_pow = torch.pow(10000, div_tensor_dv)
X_ = X/div_tensor_pow
print("final x would be: ", X_)

# meanwhile, P is 1, max_len, hiddens
P = torch.zeros(1, max_len, hidden)

# Then for the last dimension we select every other
P[:, :, 0::2] = torch.sin(X_)
P[:, :, 1::2] = torch.cos(X_)

# we can expand out for additional features as well
# but we shouldn't do this, because we can broadcast with 1 (I think?)
P_ = P.expand(features, -1, -1)
P_ = P_.reshape(max_len, hidden, features)
print("done")



