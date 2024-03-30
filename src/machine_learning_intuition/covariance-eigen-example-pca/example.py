import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(0)
data = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Compute the covariance matrix
cov_mat = np.cov(data.T)

# Compute the eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Visualize the data along with the eigenvectors
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1])

# Origin of the data (mean of the data)
origin = data.mean(axis=0)

# Plot the eigenvectors as arrows
plt.quiver(*origin, *eig_vecs[:, 0] * 3, color='r', scale=3, scale_units='xy', width=0.005)
plt.quiver(*origin, *eig_vecs[:, 1] * 3, color='g', scale=3, scale_units='xy', width=0.005)

# Setting the plot labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('2D Dataset and PCA Principal Axes')
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio ensures that the scale is the same on both axes
plt.show()
