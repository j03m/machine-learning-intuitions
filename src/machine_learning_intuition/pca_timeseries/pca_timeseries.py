import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generating synthetic financial time series data
np.random.seed(42)
dates = pd.date_range('20230101', periods=60)
prices = np.cumsum(np.random.randn(60) + 0.5) + 100  # Simulate upward trend with noise
volume = np.random.rand(60) * 100 + 50  # Random trading volume

# Create a DataFrame
data = pd.DataFrame(data={'Price': prices, 'Volume': volume}, index=dates)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(data=pca_result, index=dates, columns=['PC1', 'PC2'])

# Plot original data and principal components
plt.figure(figsize=(14, 7))

# Plotting the original Price data
plt.subplot(1, 2, 1)
plt.plot(data.index, data['Price'], label='Price', color='blue')
plt.title('Original Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)

# Plotting the first principal component
plt.subplot(1, 2, 2)
plt.plot(pca_df.index, pca_df['PC1'], label='First Principal Component', color='red')
plt.title('First Principal Component Over Time')
plt.xlabel('Date')
plt.ylabel('PC1')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
