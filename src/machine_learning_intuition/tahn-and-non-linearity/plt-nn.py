import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the network structure
layer1 = nn.Linear(10, 64)  # First linear layer from 10-dim to 64-dim
layer2 = nn.Linear(64, 64)  # Second linear layer from 64-dim to 64-dim
tanh = nn.Tanh()            # Tanh activation function

# Random input
input_tensor = torch.randn(10)

# Forward pass through the network
output1 = layer1(input_tensor)  # Output of the first linear layer
output1_activated = tanh(output1)  # Output after applying tanh
output2 = layer2(output1_activated)  # Output of the second linear layer

# Converting tensor data to numpy for plotting
input_numpy = input_tensor.detach().numpy()
output1_numpy = output1.detach().numpy()
output1_activated_numpy = output1_activated.detach().numpy()
output2_numpy = output2.detach().numpy()

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 4, 1)
plt.bar(range(10), input_numpy)
plt.title('Input Layer')
plt.xlabel('Input Dimensions')
plt.ylabel('Values')

plt.subplot(1, 4, 2)
plt.bar(range(64), output1_numpy)
plt.title('After 1st Linear Layer')
plt.xlabel('Neuron Index')

plt.subplot(1, 4, 3)
plt.bar(range(64), output1_activated_numpy)
plt.title('After Tanh Activation')
plt.xlabel('Neuron Index')

plt.subplot(1, 4, 4)
plt.bar(range(64), output2_numpy)
plt.title('After 2nd Linear Layer')
plt.xlabel('Neuron Index')

plt.tight_layout()
plt.show()
