import numpy as np
from machine_learning_intuition.types import NpArray
from .activation_function import ActivationFunction, Linear
from .initialization_function import InitFunction, He
from typing import Optional, Union
import os
import math
gbl_assert_nan = bool(os.environ.get('ASSERT_NAN', False))


class Layer():
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 learning_rate: float = 0.01,
                 activation_function: ActivationFunction = Linear(),
                 init_function: InitFunction = He(),
                 clipping: Union[int, None] = None):
        '''
        :param units: number of neurons
        :param input_shape: default is none set based on previous layer
        :param learning_rate:
        :param activation_function:
        '''
        self.input_units = input_units
        self.output_units = output_units
        self.learning_rate = learning_rate
        self.last_input: Optional[NpArray] = None
        self.activation_function = activation_function
        self.init_function = init_function
        self.clipping = clipping
        self.weights = self.init_weights()
        self.bias = self.init_bias()

    def init_weights(self) -> NpArray:
        return self.init_function(self.input_units, self.output_units)

    def init_bias(self) -> NpArray:
        bias_matrix = np.zeros((1,self.output_units))
        return bias_matrix

    def forward_pass(self, x: NpArray) -> NpArray:
        self.last_input = x
        return x.dot(self.weights) + self.bias

    def backward_pass(self, gradient: NpArray) -> NpArray:
        assert self.last_input is not None
        weights_gradient = self.last_input.T.dot(gradient)
        bias_gradient = np.sum(gradient, axis=0, keepdims=True)

        # I cannot express enough the importance pf this line
        # Here we capture the weights and use these weights our return operation
        # against the gradient, NOT the adjusted weights
        W = self.weights

        self.weights = self.weights - self.learning_rate * weights_gradient
        self.bias = self.bias - self.learning_rate * bias_gradient
        return gradient.dot(W.T)

    def assert_nan(self, x: NpArray, z: NpArray) -> None:
        if gbl_assert_nan:
            # Assertions to check for NaN in weights and biases
            assert not np.isnan(x).any()
            assert not np.isnan(z).any()
            assert not np.isinf(x).any()
            assert not np.isinf(z).any()
            self.assert_members_nan()

    def assert_members_nan(self):
        if gbl_assert_nan:
            assert not np.isnan(self.weights).any()
            assert not np.isnan(self.bias).any()
            assert not np.isinf(self.weights).any()
            assert not np.isinf(self.bias).any()