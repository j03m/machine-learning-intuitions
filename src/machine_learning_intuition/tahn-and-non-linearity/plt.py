import numpy as np
import matplotlib.pyplot as plt

# here tahn takes a linear range that is -inf to +inf and compresses
# it into a non-linear curve between -1 and 1
# This compression not only adds non-linearity but also
# helps in managing extreme values, ensuring that the output
# of the neurons remains in a more controlled and stable
# range, which is crucial for learning efficiently in
# neural networks.

# Define the input range
x = np.linspace(-3, 3, 400)
y1_linear = 2 * x + 3  # First linear function
y2_linear = -3 * x - 1 # Second linear function

# Apply tanh to the linear transformations
y1_tanh = np.tanh(y1_linear)
y2_tanh = np.tanh(y2_linear)

plt.figure(figsize=(12, 6))

# Plotting the linear functions
plt.subplot(1, 2, 1)
plt.plot(x, y1_linear, label='y = 2x + 3')
plt.plot(x, y2_linear, label='y = -3x - 1')
plt.title('Linear Transformations')
plt.xlabel('Input x')
plt.ylabel('Linear Output')
plt.grid(True)
plt.legend()

# Plotting the tanh of linear functions
plt.subplot(1, 2, 2)
plt.plot(x, y1_tanh, label='tanh(2x + 3)')
plt.plot(x, y2_tanh, label='tanh(-3x - 1)')
plt.title('Tanh of Linear Transformations')
plt.xlabel('Input x')
plt.ylabel('Tanh Output')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
