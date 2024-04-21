import torch
import torch.nn.functional as F
import math

# Define the vocabulary and embeddings
vocab = {'<pad>': 0, 'The': 1, 'cat': 2, 'sat': 3, 'on': 4, 'the': 5, 'mat': 6}
vocab_size = len(vocab)
embed_dim = 6  # Small dimension for illustration

# Create an embedding layer
embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)

# Example sentence tokens
sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
indices = torch.tensor([vocab[word] for word in sentence])

# Get embeddings for the sentence
embeddings = embedding_layer(indices)

# Positional Encoding
positions = torch.arange(len(sentence)).unsqueeze(1)
div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
positional_encodings = torch.zeros(len(sentence), embed_dim)
positional_encodings[:, 0::2] = torch.sin(positions * div_term)
positional_encodings[:, 1::2] = torch.cos(positions * div_term)

# Add positional encodings to embeddings
embeddings += positional_encodings

# Self-Attention
keys = embeddings
queries = embeddings
values = embeddings

# Dot product of queries and keys
attn_scores = torch.matmul(queries, keys.transpose(0, 1))

# Scale scores
attn_scores /= math.sqrt(embed_dim)

# Softmax to get probabilities
attn_probs = F.softmax(attn_scores, dim=-1)

# Weighted sum of values
weighted_sum = torch.matmul(attn_probs, values)

print("Attention Weights:")
print(attn_probs)
print("\nWeighted Sum (Context Vectors):")
print(weighted_sum)
