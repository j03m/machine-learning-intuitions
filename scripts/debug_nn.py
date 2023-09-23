from machine_learning_intuition import MultiLevelPerceptron
from machine_learning_intuition.utils import generate_data, scale_data
from machine_learning_intuition.neural_network import NeuralNetwork, ReLU, Linear

X, y = generate_data()

X, y = scale_data(X, y)

# Initialize and train the neural network
mlp = MultiLevelPerceptron([1, 4, 3, 2, 1])
mlp.activation_function = mlp.linear
mlp.activation_derivative = mlp.linear_derivative

nn = NeuralNetwork([1, 4, 3, 2, 1])
nn.transfer_weights(mlp)