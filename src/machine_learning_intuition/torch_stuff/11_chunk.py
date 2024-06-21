import torch

# Create a tensor of shape (2, 6)
x = torch.tensor([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])

# Split the tensor into 3 chunks along dimension 1
chunks = torch.chunk(x, 3, dim=1)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n", chunk)


# Create a tensor of shape (2, 2, 6)
x = torch.tensor([[[1, 2, 3, 4, 5, 6],
                   [7, 8, 9, 10, 11, 12]],
                  [[13, 14, 15, 16, 17, 18],
                   [19, 20, 21, 22, 23, 24]]])

# Split the tensor into 3 chunks along dimension 2
chunks = torch.chunk(x, 3, dim=2)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n", chunk)
