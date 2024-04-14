import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
x = np.linspace(0, 10, 100)  # 100 data points in the range [0, 10]
y = np.sin(x) + np.random.normal(0, 0.1, 100)  # Corresponding values with some noise

# Gaussian kernel function
def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

# Query point
query_x = 5

# Compute distances from the query point to all other points
distances = np.abs(x - query_x)

# Bandwidth for the Gaussian kernel
bandwidth = 0.5

# Compute weights for each point using the Gaussian kernel
weights = gaussian_kernel(distances, bandwidth)

# Perform a weighted sum to estimate the value at the query point
estimated_y = np.sum(weights * y) / np.sum(weights)

print(f"Estimated value at x={query_x}: {estimated_y}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.scatter(query_x, estimated_y, color='red', label='Estimated point')
plt.plot(x, np.sin(x), '--', color='green', label='True function')
plt.legend()
plt.show()
