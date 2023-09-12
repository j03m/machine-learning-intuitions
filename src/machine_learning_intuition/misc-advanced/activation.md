The choice of activation function can indeed play a critical role in the learning capabilities of a neural network. Each type of activation function has its pros and cons, and the choice often depends on the specific application. Let's explore some of the commonly used activation functions:

### Sigmoid
- **Pros**: Smooth gradient, preventing "jumps" in output values. Outputs a probability value between 0 and 1.
- **Cons**: Susceptible to vanishing gradient problem, especially in deep networks. Also, it squashes all the output values between 0 and 1, which might not be desirable for all types of problems.

### ReLU (Rectified Linear Unit)
- **Pros**: Computationally efficient. Helps mitigate the vanishing gradient problem to some extent.
- **Cons**: The output is unbounded, so it can blow up. The gradient is zero for negative values, which means during backpropagation, weights will not get adjusted for those neurons, leading to dead neurons.

### Tanh
- **Pros**: Output is zero-centered because its range is -1 to 1, which can make learning easier for the next layer.
- **Cons**: Like the sigmoid function, the tanh function is also susceptible to the vanishing gradient problem.

### Intuition & Practical Consideration
- **Sigmoid and Tanh**: These are good for output layers of binary and multi-class classification problems, respectively, where you want the output to fall in a normalized range. 
- **ReLU**: It's commonly used in the hidden layers of deep networks because of its computational efficiency. However, you have to be cautious of the potential for exploding gradients, especially when your network architecture is deep.

Given your specific case of sorting numbers, and since you're concerned about clipping, you might want to try ReLU or its variants like Leaky ReLU or Parametric ReLU, which allow a small gradient when the unit is not active and could potentially mitigate the dying ReLU problem. 

Remember that each has its place depending on the problem you're trying to solve and the architecture you're using. For example, if your output naturally should fall between 0 and 1, then Sigmoid could be a good choice for the output layer. On the other hand, if your network is deep and you're worried about vanishing gradients, ReLU or its variants might be more appropriate for the hidden layers.
