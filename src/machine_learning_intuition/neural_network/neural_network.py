from .layer import Layer
from .activation_function import LeakyReLU, ActivationFunction, all_activation_functions
from .initialization_function import InitFunction, XavierGlordotSimple, all_init_functions
from .loss_function import MSE, Loss, all_loss_functions
from typing import List, Union, Any, Dict
from machine_learning_intuition.types import NpArray
from numpy import float64
import numpy as np
import json
import os

gbl_record_debug_data = bool(os.environ.get('RECORD_DEBUG', False))


class NeuralNetwork:

    def __init__(self,
                 layers: List[int],
                 activation_functions: Union[List[ActivationFunction], None] = None,
                 loss_function: Loss = MSE(),
                 learning_rate: float = 0.01,
                 init_functions: Union[List[InitFunction], None] = None,
                 clipping: Union[int, None] = None):

        self.loss_function = loss_function
        self.clipping = clipping
        self.debug_values: Dict[int, List[Any]] = {}
        self.current_epoch = -1

        # this is dumb, move to .add_layer pattern
        if activation_functions is None:
            activation_functions = [LeakyReLU()] * (len(layers) - 1)

        if init_functions is None:
            init_functions = [XavierGlordotSimple()] * (len(layers) - 1)

        self.layers: List[Layer] = []
        self.layer_spec = layers
        self.activation_spec = [act_func.name for act_func in activation_functions]
        self.init_spec = [init_func.name for init_func in init_functions]
        for i in range(len(layers) - 1):
            input_spec = layers[i]
            output_spec = layers[i + 1]
            activation_function = activation_functions[i]
            init_function = init_functions[i]
            layer = Layer(input_spec, output_spec, learning_rate, activation_function, init_function,
                          clipping=self.clipping)
            self.layers.append(layer)
        self.learning_rate = learning_rate

    def predict(self, input_: NpArray):
        layer_output = input_
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output)
        return layer_output

    def transfer_weights(self, mlp: Any):
        print(mlp.weights)

    def collect_weights(self):
        weights = []
        bias = []
        for layer in reversed(self.layers):
            weights.append(layer.W)
            bias.append(layer.w0)
        return weights, bias

    def train(self, x: NpArray, y: NpArray, epochs: int = 1000, patience_limit: int = 50, warm_up_epochs: int = 500,
              verbosity: int = 1):

        best_val_loss = float64('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch
            for x_val, y_true, index in zip(x, y, range(0, len(x))):
                x_val = x_val.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)
                y_pred = self.predict(x_val)
                loss = self.loss_function(y_true, y_pred)
                initial_gradient = self.loss_function.derivative(y_true, y_pred)
                record = False
                if index == len(x) - 1 and gbl_record_debug_data:
                    record = True

                self.backwards_propagate(initial_gradient, record)

            if verbosity > 1:
                print(f"Epoch {epoch}, Loss: {loss}")

            if verbosity == 1 and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            if epoch >= warm_up_epochs:
                if loss < best_val_loss:
                    best_val_loss = loss
                    patience_counter = 0  # Reset counter
                else:
                    patience_counter += 1  # Increment counter

                if patience_counter >= patience_limit:
                    if verbosity != 0:
                        print("Early stopping due to lack of improvement.")
                    break

    def backwards_propagate(self, gradient: NpArray, record=False):
        layer_number = 0
        loss_gradient = gradient
        for layer in reversed(self.layers):
            loss_gradient = layer.backward_pass(loss_gradient)
            if record:
                pass
                # self.record(layer_number, loss_gradient, layer.weights, layer.bias)
            layer_number += 1

    def record(self, layer_number: int, loss_gradient: NpArray, weights: NpArray, bias: NpArray):
        if self.current_epoch not in self.debug_values:
            self.debug_values[self.current_epoch] = []

        self.debug_values[self.current_epoch].append(
            {
                "layer": layer_number,
                "gradient": np.copy(loss_gradient),
                "weights": np.copy(weights),
                "bias": np.copy(bias)
            }
        )

    def save(self, filename):
        # save the layer specs
        # save the activation functions names
        # save the loss function name
        # save each layer weight/bias
        save_data = {
            "layers": self.layer_spec,
            "activation_functions": self.activation_spec,
            "loss": self.loss_function.name,
            "weights": [layer.weights.to_list() for layer in self.layers],
            "learning_rate": self.learning_rate,
            "biases": [layer.bias.to_list() for layer in self.layers]
        }
        with open(filename, 'w') as f:
            json.dump(save_data, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            save_data = json.load(f)

        loss_function = all_loss_functions[save_data["loss"]]
        activation_functions = [all_activation_functions[name]() for name in save_data["activation_functions"]]
        init_functions = [all_init_functions[name]() for name in save_data["init_functions"]]
        layer_spec = save_data["layers"]
        learning_rate = save_data["learning_rate"]
        weights = save_data["weights"]
        biases = save_data["biases"]

        nn = NeuralNetwork(layer_spec,
                           loss_function=loss_function,
                           activation_functions=activation_functions,
                           init_functions=init_functions,
                           learning_rate=learning_rate)

        for layer, weight, bias in zip(nn.layers, weights, biases):
            layer.weights = np.array(weight)
            layer.bias = np.array(bias)

        return nn
