from .layer import Layer
from .activation_function import Linear, ActivationFunction, all_activations
from .loss_function import MSE, Loss, all_loss_functions
from typing import List
from machine_learning_intuition.types import NpArray
from numpy import float64
import json


class NeuralNetwork:

    def __int__(self,
                layers: List[int],
                activation_functions: List[ActivationFunction] = None,
                loss_function: Loss = MSE,
                learning_rate: float = 0.01):

        self.loss_function = loss_function

        if activation_functions is None:
            activation_functions = [Linear] * len(layers)

        self.layers: List[Layer] = []
        self.layer_spec = layers
        self.activation_spec = [act_func.name for act_func in activation_functions]
        for i in range(len(layers) - 1):
            input_spec = layers[i]
            output_spec = layers[i + 1]
            activation_function = activation_functions[i]
            layer = Layer(input_spec, output_spec, learning_rate, activation_function)
            self.layers.append(layer)

    def predict(self, input_: NpArray):
        layer_output = input_
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output)
        return layer_output

    def train(self, x: NpArray, y: NpArray, epochs: int, patience_limit: int = 500, warm_up_epochs: int = 500,
              verbose=True):
        # todo: should this be LAST value loss instead?
        best_val_loss = float64('inf')
        patience_counter = 0

        for epoch in range(epochs):
            for x_val, y_true in zip(x, y):
                y_pred = self.predict(x_val)
                loss = self.loss_function(y_true, y_pred)
                loss_gradient = self.loss_function.derivative(y_true, y_pred)
                self.backwards_propagate(loss_gradient)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            if epoch >= warm_up_epochs:
                if loss < best_val_loss:
                    best_val_loss = loss
                    patience_counter = 0  # Reset counter
                else:
                    patience_counter += 1  # Increment counter

                if patience_counter >= patience_limit:
                    if verbose:
                        print("Early stopping due to lack of improvement.")
                    break

    def backwards_propagate(self, gradient: NpArray):
        loss_gradient = gradient
        for layer in reversed(self.layers):
            loss_gradient = layer.backwards_pass(loss_gradient)

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
        activation_functions = [all_activations[name] for name in save_data["activation_functions"]]
        layer_spec = save_data["layers"]
        learning_rate = save_data["learning_rate"]
        weights = save_data["weights"]
        biases = save_data["biases"]

        nn = NeuralNetwork(layer_spec,
                           loss_function=loss_function,
                           activation_functions=activation_functions,
                           learning_rate=learning_rate)

        for layer, weight, bias in zip(nn.layers, weights, biases):
            layer.weights = weight
            layer.bias = bias
