import numpy as np
import json
import os
from machine_learning_intuition.utils import mse

_assert_nan = bool(os.environ.get('ASSERT_NAN', False))
np.seterr(over='raise')


# Adding assertions to check for NaN in weights and biases
class MultiLevelPerceptron:
    def __init__(self, layers, activation="linear", init=True):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activation = activation
        if activation == "linear":
            self.activation_function = self.linear
            self.activation_derivative = self.linear_derivative
        elif activation == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise Exception("Unknown activation")
        if init:
            self._initialize_weights_and_biases()

    def linear(self, x):
        return x

    def linear_derivative(self, z):
        return 1

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, z):
        return 1 - (z ** 2)

    def save(self, filename):
        model_dict = {
            'layers': self.layers,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'activation': self.activation
        }
        with open(filename, 'w') as f:
            json.dump(model_dict, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            model_dict = json.load(f)

        layers = model_dict['layers']
        weights = [np.array(w) for w in model_dict['weights']]
        biases = [np.array(b) for b in model_dict['biases']]
        if 'activation' in model_dict:
            activation = model_dict['activation']
            model = MultiLevelPerceptron(layers, activation=activation, init=False)
        else:
            model = MultiLevelPerceptron(layers, init=False)

        model.weights = weights
        model.biases = biases
        return model

    def _initialize_weights_and_biases(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            # naive
            # weight_matrix = np.random.randn(self.layers[i + 1], self.layers[i])
            # He
            # weight_matrix = np.random.randn(self.layers[i + 1], self.layers[i]) * np.sqrt(2. / self.layers[i])
            # Xavier/Glordot
            limit = np.sqrt(6 / (self.layers[i] + self.layers[i + 1]))
            weight_matrix = np.random.uniform(-limit, limit, (self.layers[i + 1], self.layers[i]))
            self.weights.append(weight_matrix)
            bias_matrix = np.zeros((self.layers[i + 1], 1))
            self.biases.append(bias_matrix)

    def predict(self, input_):
        x = np.array(input_)

        # Check if x is a scalar
        if x.shape == ():
            x = x.reshape(1, 1)
        elif len(x.shape) == 2:  # If the input is a 2D array
            if x.shape[0] == 1:  # If there's only one sample
                x = x.T  # Transpose to make it a column vector
        else:  # 1D array
            x = x.reshape(-1, 1)  # Reshape to a column vector

        activations = [x]
        zs = []

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, x) + bias
            zs.append(z)
            x = self.activation_function(z)
            activations.append(x)
            self.assert_nan(x, activations, zs)

        return x, activations, zs

    def backward_propagation(selrf, y_true, y_pred, activations, zs):
        grad_weights = [np.zeros(w.shape) for w in self.weights]
        grad_biases = [np.zeros(b.shape) for b in self.biases]

        delta = 2 * (y_pred - y_true) * self.activation_derivative(zs[-1])
        # delta = (y_pred - y_true)
        grad_biases[-1] = delta
        grad_weights[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.layers)):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_derivative(zs[-l])
            grad_biases[-l] = delta
            grad_weights[-l] = np.dot(delta, activations[-l - 1].T)

        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, grad_weights)]
        self.biases = [b - learning_rate * gb for b, gb in zip(self.biases, grad_biases)]

    def assert_nan(self, x, activations, zs):
        if _assert_nan:
            # Assertions to check for NaN in weights and biases
            for i, w in enumerate(self.weights):
                assert not np.isnan(w).any(), f'NaN found in weights at index {i}'
            for i, b in enumerate(self.biases):
                assert not np.isnan(b).any(), f'NaN found in biases at index {i}'
            for i, b in enumerate(x):
                assert not np.isnan(b).any(), f'NaN found in x at index {i}'
            for i, b in enumerate(activations):
                assert not np.isnan(b).any(), f'NaN found in activations at index {i}'
            for i, b in enumerate(zs):
                assert not np.isnan(b).any(), f'NaN found in zs at index {i}'

    def train(self, X, y, epochs=1000, learning_rate=0.01, patience_limit=500, warm_up_epochs=500):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):

            for x_val, y_true in zip(X, y):
                # sometimes storing the zs is useful for backpropagation. So, predict returns it
                # but we don't need it here
                y_pred, activations, zs = self.predict(x_val)
                loss = mse(y_true, y_pred)
                grad_weights, grad_biases = self.backward_propagation(y_true.reshape(-1, 1), y_pred, activations, zs)
                self.update_parameters(grad_weights, grad_biases, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            if epoch >= warm_up_epochs:

                if loss < best_val_loss:
                    best_val_loss = loss
                    patience_counter = 0  # Reset counter
                else:
                    patience_counter += 1  # Increment counter

                if patience_counter >= patience_limit:
                    print("Early stopping due to lack of improvement.")
                    break
