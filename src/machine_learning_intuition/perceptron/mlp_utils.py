import numpy as np


# Assuming X and y is between 0 and 1 otherwise use minmax_scale
def scale_data(x, y):
    global_min = min(0, np.min(y))
    global_max = max(1, np.max(y))
    x_scaled = minmax_scale(x, global_min, global_max)
    y_scaled = minmax_scale(y, global_min, global_max)
    return x_scaled, y_scaled


def minmax_scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def generate_data(num_samples=1000):
    X = np.random.rand(num_samples)
    y = X * 100
    return X, y


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def generate_apartment_data(num_samples=1000):
    # Initialize empty lists to hold our features and labels
    square_feet = []
    num_bedrooms = []
    num_bathrooms = []
    proximity_to_transit = []
    neighborhood_quality = []
    labels = []

    # Generate features and labels
    for _ in range(num_samples):
        square_feet.append(np.random.randint(500, 3001))
        num_bedrooms.append(np.random.randint(0, 5))
        num_bathrooms.append(np.random.randint(1, 4))
        proximity_to_transit.append(np.random.randint(1, 11))
        neighborhood_quality.append(np.random.randint(1, 11))

        # Calculate label (price) based on the features
        base_price = (square_feet[-1] * 1.5) + (num_bedrooms[-1] * 300) + (num_bathrooms[-1] * 200) + (
                proximity_to_transit[-1] * 40) + (neighborhood_quality[-1] * 50)

        # Add random fluctuation between 0-1%
        fluctuation = np.random.uniform(0, 0.01)
        final_price = base_price * (1 + fluctuation)

        labels.append(final_price)

    if num_samples > 1:
        # Scale each feature
        square_feet = minmax_scale(np.array(square_feet), np.min(square_feet), np.max(square_feet))
        num_bedrooms = minmax_scale(np.array(num_bedrooms), np.min(num_bedrooms),
                                    np.max(num_bedrooms))
        num_bathrooms = minmax_scale(np.array(num_bathrooms), np.min(num_bathrooms),
                                     np.max(num_bathrooms))
        proximity_to_transit = minmax_scale(np.array(proximity_to_transit),
                                            np.min(proximity_to_transit),
                                            np.max(proximity_to_transit))
        neighborhood_quality = minmax_scale(np.array(neighborhood_quality),
                                            np.min(neighborhood_quality),
                                            np.max(neighborhood_quality))
        labels = minmax_scale(labels, np.min(labels), np.max(labels))

    # Assemble features back into a single array
    features = np.column_stack((square_feet, num_bedrooms, num_bathrooms, proximity_to_transit, neighborhood_quality))

    return np.array(features), np.array(labels)
