# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generating synthetic data: size of the house and number of bedrooms
np.random.seed(0)
house_size = np.random.normal(3000, 750, 100)  # Mean size 3000 sq ft with some variance
num_bedrooms = (house_size / 1000) + np.random.normal(0, 0.5, 100)  # More bedrooms in larger houses, but with variance

# Stack them into a dataset
X = np.column_stack((house_size, num_bedrooms))

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting the original data and the first principal component
plt.figure(figsize=(12, 6))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.8)
plt.xlabel('House Size (standardized)')
plt.ylabel('Number of Bedrooms (standardized)')
plt.title('Original Data')
plt.grid(True)

# Add the PCA direction
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    color='red',
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# Plot data
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.8)
draw_vector([0, 0], [0, 3], ax=plt.gca())
draw_vector([0, 0], [3, 0], ax=plt.gca())
plt.xlabel('House Size (standardized)')
plt.ylabel('Number of Bedrooms (standardized)')
plt.title('Principal Component Direction')
plt.grid(True)

# Showing the direction of the first principal component
for v in pca.components_:
    draw_vector([0, 0], v * 3,  ax=plt.gca())
plt.axis('equal')  # Equal scaling by x and y axes

plt.tight_layout()
plt.show()
