import numpy as np
from numpy import ndarray, float64
from machine_learning_intuition.types import NpArray
from .activation_function import ActivationFunction
from typing import Optional, Tuple
import os

gbl_assert_nan = bool(os.environ.get('ASSERT_NAN', False))


class Layer():
    def __init__(self, units: int, next_units: int, learning_rate: float, activation_function: ActivationFunction):
        self.units = units
        self.next_units = next_units
        self.weights = self.init_weights()
        self.bias = self.init_bias()
        self.learning_rate = learning_rate
        self.last_input: Optional[NpArray] = None
        self.last_output: Optional[NpArray] = None
        self.activation_function = activation_function

    def init_weights(self) -> NpArray:
        # Xavier/Glordot
        limit = np.sqrt(6 / (self.units + self.next_units))
        weights_matrix = np.random.uniform(-limit, limit, (self.next_units, self.units))
        return weights_matrix

    def init_bias(self) -> NpArray:
        bias_matrix = np.zeros((self.next_units, 1))
        return bias_matrix

    def forward_pass(self, x: NpArray) -> NpArray:
        self.last_input = x
        z = x.dot(self.weights) + self.bias
        self.assert_nan(x, z)
        self.last_output = self.activation_function(z)
        return self.last_output

    def backwards_pass(self, gradient: NpArray) -> NpArray:
        assert self.last_input is not None
        assert self.last_output is not None
        delta = gradient * self.activation_function.derivative(self.last_output)
        weights_gradient = self.last_input.T.dot(delta)
        bias_gradient = np.sum(delta, axis=0)
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient
        return NpArray(delta.dot(self.weights.T))

    def assert_nan(self, x: NpArray, z: NpArray) -> None:
        if gbl_assert_nan:
            # Assertions to check for NaN in weights and biases
            assert not np.isnan(self.weights).any()
            assert not np.isnan(self.bias).any()
            assert not np.isnan(x).any()
            assert not np.isnan(z).any()
