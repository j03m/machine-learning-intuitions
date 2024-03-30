import matplotlib.pyplot as plt
import numpy as np

# Setting up the figure for plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Original vector (that will not be an eigenvector)
v = np.array([2, 1])

# Example transformation matrix
A = np.array([[2, 1], [1, 2]])

# Transforming the original vector
v_transformed = A.dot(v)

# Plotting the original vector
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector')

# Plotting the transformed vector
ax.quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1, color='green', label='Transformed Vector (Not Eigenvector)')

# An example eigenvector of A
eigenvector = np.array([1, 1])

# Transforming the eigenvector
eigenvector_transformed = A.dot(eigenvector)

# Plotting the eigenvector before transformation
ax.quiver(0, 0, eigenvector[0], eigenvector[1], angles='xy', scale_units='xy', scale=1, color='red', label='Eigenvector')

# Plotting the transformed eigenvector
ax.quiver(0, 0, eigenvector_transformed[0], eigenvector_transformed[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Transformed Eigenvector (Same Direction)')

# Setting up the plot
plt.grid(True)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.legend()
plt.title('Eigenvector vs. Non-Eigenvector Transformation')
plt.show()
