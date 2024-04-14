import numpy as np

# Define keys and a query
keys = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])  # Shape: (3, 3), 3 keys of dimension 3
query = np.array([1, 2, 3])  # Shape: (3,), a single query of dimension 3

# Compute dot products between the query and each key
dot_products = np.dot(keys, query)  # Shape: (3,), dot product for each key-query pair

# Apply softmax to get attention weights
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

attention_weights = softmax(dot_products)

print("Dot Products:", dot_products)
print("Attention Weights:", attention_weights)
