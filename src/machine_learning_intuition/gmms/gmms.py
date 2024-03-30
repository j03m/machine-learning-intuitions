import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the Gaussian components
means = [0, 5, 10]
variances = [1, 2, 1.5]
coefficients = [0.5, 0.3, 0.2]  # Mixture coefficients must sum to 1

# Generate a range of values
x = np.linspace(-5, 15, 1000)

# Calculate the mixed Gaussian distribution
mixed_gaussian = sum(coeff * norm.pdf(x, mean, np.sqrt(variance))
                     for coeff, mean, variance in zip(coefficients, means, variances))

# Plot each Gaussian component
for mean, variance in zip(means, variances):
    plt.plot(x, norm.pdf(x, mean, np.sqrt(variance)), label=f'Mean: {mean}, Variance: {variance}')

# Plot the mixed Gaussian distribution
plt.plot(x, mixed_gaussian, color='k', linewidth=2, label='Mixed Gaussian')

plt.fill_between(x, mixed_gaussian, color='k', alpha=0.1)
plt.legend()
plt.title('Gaussian Mixture Model')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
