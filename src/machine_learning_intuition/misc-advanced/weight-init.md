## Weight + Bias initialization

Let's dive into these weight initialization techniques.

### 1. He Initialization
The line of code:
```python
weight_matrix = np.random.randn(self.layers[i + 1], self.layers[i]) * np.sqrt(2. / self.layers[i])
```
is an example of He Initialization. It was designed for ReLU (Rectified Linear Unit) activations and aims to keep the variance of the activations roughly the same across different layers in a deep neural network. The idea is to help mitigate the vanishing/exploding gradients problem in deep networks.

Here, \( \text{np.random.randn} \) generates samples from a standard normal distribution with a mean of 0 and a standard deviation of 1. These samples are then scaled by \( \sqrt{\frac{2}{\text{self.layers}[i]}} \).

In the formula \( \sqrt{\frac{2}{\text{self.layers}[i]}} \), 2 is a hyperparameter and \( \text{self.layers}[i] \) represents the number of units in the layer from which the output is coming.

### 2. Xavier/Glorot Initialization
The lines of code:
```python
limit = np.sqrt(6 / (self.layers[i] + self.layers[i + 1]))
weight_matrix = np.random.uniform(-limit, limit, (self.layers[i + 1], self.layers[i]))
```
implement Xavier/Glorot Initialization. This technique was designed for Sigmoid and hyperbolic tangent (tanh) activation functions and aims to keep the variance of the activations roughly the same across different layers in a deep network.

The formula \( \sqrt{\frac{6}{\text{self.layers}[i] + \text{self.layers}[i+1]}} \) is derived from keeping the variance of the weights the same for both forward and backward passes. \( \text{self.layers}[i] \) and \( \text{self.layers}[i + 1] \) represent the number of units in the layers connected by the weight.

Both of these methods are designed to give you a good starting point, reducing the risk of encountering the vanishing or exploding gradient problems, especially for deeper networks.

In summary, these initializations try to set the weights in such a way that the variance remains the same for \(x\) and \(y\) through the layer, to avoid the vanishing and exploding gradient problems. The Xavier/Glorot initialization assumes the activation function is roughly linear (or is linear around 0, like tanh or sigmoid), while He initialization is designed for ReLU activations.


### He Initialization
The formula for He Initialization is:

\[
\text{weight} = \sqrt{\frac{2}{n}} \cdot \text{Random Normal Value}
\]

The \( \sqrt{\frac{2}{n}} \) term is used to scale the random normal values generated for the weights. The factor \( n \) is the number of input units in the layer for which the weights are being initialized. This scaling factor is derived from maintaining the variance of the activations in a network with ReLU activations.

When using the ReLU activation function, the output is zero for half of the input domain (i.e., for negative inputs). To compensate for this "dying" feature of the ReLU, the scaling factor \( \sqrt{2} \) is used to maintain the variance.

### Xavier/Glorot Initialization
The formula for Xavier Initialization is:

\[
\text{limit} = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
\]
\[
\text{weight} = \text{Random Uniform Value between } -\text{limit and limit}
\]

Here \( n_{\text{in}} \) and \( n_{\text{out}} \) are the number of input and output units for the layer. The random values for the weights are uniformly chosen from a range \([- \text{limit}, \text{limit}]\).

For linear and near-linear functions like the sigmoid and hyperbolic tangent, it's important to ensure that the input variance equals the output variance. The scaling factor \( \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} \) is derived to maintain the variance for both forward and backward passes in the network. The 6 in the numerator is actually 2 times 3, where 2 is for the uniform distribution variance and 3 is an adjustable parameter. 

In simpler terms, for Xavier initialization, you want the variance of the activations to be the same for both the input and output of each layer. Doing so helps keep the gradients from vanishing too quickly.

### Summary
Both initializations aim to balance the scales such that each neuron operates in a regime where gradients are neither too small nor too large, both during forward and backward passes. This makes the training of deep architectures more stable and faster.

