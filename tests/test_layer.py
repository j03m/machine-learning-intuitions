from machine_learning_intuition import neural_network

import numpy as np


def test_layer_init():
    layer = neural_network.Layer(5, 3)
    assert layer.input_units == 5
    assert layer.output_units == 3
    assert layer.weights.shape == (5, 3)
    assert layer.bias.shape == (3,)


def test_forward_pass():
    layer = neural_network.Layer(2, 3)
    control_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    layer.weights = control_values
    layer.bias = np.array([0.0, 0.0, 0.0])
    _input = np.array([1, 1]).reshape(1, -1)
    output = layer.forward_pass(_input)
    expected_output = np.array([[0.5, 0.7, 0.9]])
    assert np.allclose(output, expected_output, atol=1e-8)


def test_batch_forward_pass():
    layer = neural_network.Layer(5, 3)
    input_ = np.random.uniform(-1, 1, (1, 5))
    output = layer.forward_pass(input_)
    assert output.shape == (1, 3)
    input_ = np.random.uniform(-1, 1, (100, 5))
    output = layer.forward_pass(input_)
    assert output.shape == (100, 3)



'''
1. **Initial Setup**: Choose initial weights, biases, input, and gradient. 
    - `initial_weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]` (shape: 3x2)
    - `initial_bias = [0, 0, 0]` (shape: 3)
    - `input = [1, 1]` (shape: 1x2)
    - `gradient = [0.1, 0.2, 0.3]` (shape: 3)
    - `learning_rate = 0.01`

2. **Forward Pass**: 
    - `output = input.dot(initial_weights) + initial_bias`
    - `output = [1*0.1 + 1*0.2, 1*0.3 + 1*0.4, 1*0.5 + 1*0.6] = [0.3, 0.7, 1.1]`

3. **Calculate Delta**: 
    - `delta = gradient * 1 = [0.1, 0.2, 0.3]`

4. **Calculate Weights Gradient and Bias Gradient**:
    - `weights_gradient = input.T.dot(delta) = [1, 1].T.dot([0.1, 0.2, 0.3]) = [0.1+0.2+0.3, 0.1+0.2+0.3] = [0.6, 0.6]`
    - `bias_gradient = sum(delta, axis=0) = 0.1 + 0.2 + 0.3 = 0.6`

5. **Calculate New Weights and Bias**:
    - `new_weights = initial_weights - learning_rate * weights_gradient`
    - `new_weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] - 0.01 * [0.6, 0.6]`
    - `new_weights = [[0.1-0.006, 0.2-0.006], [0.3-0.006, 0.4-0.006], [0.5-0.006, 0.6-0.006]]`
    - `new_weights = [[0.094, 0.194], [0.294, 0.394], [0.494, 0.594]]`
    - `new_bias = initial_bias - learning_rate * bias_gradient = [0, 0, 0] - 0.01 * 0.6 = [-0.006, -0.006, -0.006]`
'''


def test_backwards_pass():
    layer = neural_network.Layer(2, 3)
    control_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    layer.weights = control_values
    layer.bias = np.array([0.0, 0.0, 0.0])
    _input = np.array([1, 1]).reshape(1, -1)
    output = layer.forward_pass(_input)
    expected_output = np.array([[0.5, 0.7, 0.9]])
    assert np.allclose(output, expected_output, atol=1e-8)

    gradient = np.array([0.1, 0.2, 0.3]).reshape(1, -1)
    layer.backwards_pass(gradient)

    expected_weights = np.array([[0.099, 0.198, 0.297], [0.399, 0.498, 0.597]])
    expected_bias = np.array([-0.001, -0.002, -0.003])

    assert np.allclose(layer.weights, expected_weights, atol=1e-8)
    assert np.allclose(layer.bias, expected_bias, atol=1e-8)
