import numpy as np
from machine_learning_intuition.neural_network import NeuralNetwork, Linear


def test_neural_network_init():
    nn = NeuralNetwork([2, 3, 1])
    assert len(nn.layers) == 2
    assert nn.layers[0].input_units == 2
    assert nn.layers[0].output_units == 3
    assert nn.layers[1].input_units == 3
    assert nn.layers[1].output_units == 1


def test_forward_propagation():
    nn = NeuralNetwork([2, 3, 1])
    nn.layers[0].weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    nn.layers[1].weights = np.array([[0.7], [0.8], [0.9]])
    input_ = np.array([1, 1]).reshape(1, -1)
    output = nn.predict(input_)
    expected_output = np.array([[0.5, 0.7, 0.9]]).dot(np.array([[0.7], [0.8], [0.9]]))
    assert np.allclose(output, expected_output, atol=1e-8)


def test_backwards_propagation():
    # Initialize Neural Network with specific weights and biases
    nn = NeuralNetwork([2, 3, 1])
    nn.layers[0].weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    nn.layers[1].weights = np.array([[0.7], [0.8], [0.9]])
    nn.layers[0].last_input = np.array([[1, 1]])
    nn.layers[0].last_output = np.array([[0.5, 0.6, 0.7]])
    nn.layers[1].last_input = np.array([[0.5, 0.6, 0.7]])
    nn.layers[1].last_output = np.array([[1.0]])

    # Manually set the gradient from the loss function
    loss_gradient = np.array([[-1.66]])

    # Perform backward propagation
    nn.backwards_propagate(loss_gradient)

    # Expected updated weights and biases
    expected_weights_layer_1 = np.array(
        [[0.11176778, 0.213447956, 0.315133282], [0.41176778, 0.513447956, 0.615133282]])
    expected_weights_layer_2 = np.array([[0.7083],[0.80996],[0.91162]])

    # Check if weights and biases are updated correctly
    np.testing.assert_almost_equal(nn.layers[0].weights, expected_weights_layer_1, decimal=4)
    np.testing.assert_almost_equal(nn.layers[1].weights, expected_weights_layer_2, decimal=4)


