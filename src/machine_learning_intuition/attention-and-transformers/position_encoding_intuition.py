import torch

max_len = 10
num_hiddens = 4

# imagine input_embeddings are our words, P becomes a "tint" applied to each word to represent its position
input_embeddings = torch.arange(0, max_len * num_hiddens).reshape(1, max_len, num_hiddens)

# create the tint
P = torch.zeros((1, max_len, num_hiddens))
pos_index = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
pows = torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
print("original pos_index: ", pos_index)
print("pows: ", pows)
pos_index = pos_index / pows
print("final pos_index: ", pos_index)
print("original P: ", P)


# believe we use sin/cos because a linearly inceasing value may not scale (even if small)
# sin/cos cycle?
P[:, :, 0::2] = torch.sin(pos_index)
print("P post sin assignment: ", P)
P[:, :, 1::2] = torch.cos(pos_index)
print("P post cosin assignment: ", P)


# apply the tint
input_embeddings = input_embeddings + P
print("tinted input:", input_embeddings)