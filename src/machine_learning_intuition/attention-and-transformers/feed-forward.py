import numpy as np

# Define dimensions
seq_length = 3  # Number of positions in the sequence
d_model = 4     # Number of features per position
d_ffn = 6       # Number of hidden units in the FFN

# Create random input matrix X with shape [seq_length, d_model]
X = np.random.randn(seq_length, d_model)

# Create weights for the first dense layer
W1 = np.random.randn(d_model, d_ffn)
b1 = np.random.randn(d_ffn)

# Create weights for the second dense layer
W2 = np.random.randn(d_ffn, d_model)
b2 = np.random.randn(d_model)

# Display initial setup
print("Input X:\n", X)
print("Weights W1:\n", W1)
print("Bias b1:\n", b1)
print("Weights W2:\n", W2)
print("Bias b2:\n", b2)

# Apply the first Dense layer (XW1 + b1)
Z1 = np.dot(X, W1) + b1  # Shape [seq_length, d_ffn]

# Apply ReLU activation
A1 = np.maximum(0, Z1)  # ReLU

# Apply the second Dense layer (A1W2 + b2)
outputs = np.dot(A1, W2) + b2  # Shape [seq_length, d_model]

# Print the outputs at each stage
print("Output of first Dense layer Z1:\n", Z1)
print("Output of ReLU A1:\n", A1)
print("Final Output:\n", outputs)


'''
Example of each "sequence" having weights applied independently:

Inputs X (shape [3, 2]):
[[x11, x12],
 [x21, x22],
 [x31, x32]]

Weights W1 (shape [2, 3]):
[[w11, w12, w13],
 [w21, w22, w23]]

Bias b1 (shape [3]):
[b1, b2, b3]

Result of np.dot(X, W1) (shape [3, 3]):
[[x11*w11 + x12*w21, x11*w12 + x12*w22, x11*w13 + x12*w23],
 [x21*w11 + x22*w21, x21*w12 + x22*w22, x21*w13 + x22*w23],
 [x31*w11 + x32*w21, x31*w12 + x32*w22, x31*w13 + x32*w23]]

Result after adding bias b1 (shape [3, 3]):
[[x11*w11 + x12*w21 + b1, x11*w12 + x12*w22 + b2, x11*w13 + x12*w23 + b3],
 [x21*w11 + x22*w21 + b1, x21*w12 + x22*w22 + b2, x21*w13 + x22*w23 + b3],
 [x31*w11 + x32*w21 + b1, x31*w12 + x32*w22 + b2, x31*w13 + x32*w23 + b3]]

'''