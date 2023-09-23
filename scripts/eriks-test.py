import numpy as np

from machine_learning_intuition.neural_network import NeuralNetwork

np.random.seed(12346)


def generate_data(num_samples=1000):
    np.random.seed(42)
    X = np.random.rand(num_samples)
    y = X * 100
    return X, y


def minmax_scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def convert_samples_to_np_array(samples):
    samples_np = np.array(samples)
    return samples_np.reshape(-1, 1)


def flattened_diff(a, b):
    flat_a = flatten(a)
    flat_b = flatten(b)
    return np.abs(np.array(flat_a) - np.array(flat_b))


def is_numeric(n):
    return isinstance(n, (int, float, np.number))


def flatten(a):
    result = []
    for item in a:
        if is_numeric(item):
            result.append(item)
        elif isinstance(item, (list, np.ndarray)):
            result.extend(flatten(item))
    return result


def calculate_diffs(losses1, losses2):
    diffs = []
    for l1, l2 in zip(losses1, losses2):
        diff = np.abs(np.array(l1) - np.array(l2))
        diffs.append(diff.tolist())
    return diffs


x, y = generate_data()

global_min = min(0, np.min(x))
global_max = max(1, np.max(y))
x_scaled = minmax_scale(x, global_min, global_max)
y_scaled = minmax_scale(y, global_min, global_max)

x = convert_samples_to_np_array(x_scaled)
y = convert_samples_to_np_array(y_scaled)

for x_, y_ in zip(x, y):
    np.isclose(x_ * 100, y_, atol=1e-9)

rn = NeuralNetwork([1, 5, 5, 1])

rn.train(x, y, 1000)

print("me:", rn.predict(np.array([5])))
print("me:", rn.predict(np.array([3])))
