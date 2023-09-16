2014 paper: https://arxiv.org/pdf/1409.3215.pdf

gpt from scratch: https://jaykmody.com/blog/gpt-from-scratch/

https://atcold.github.io/NYU-DLSP21/en/week08/08-3/

https://github.com/eriklindernoren/ML-From-Scratch <- should be helpful

https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html

### What is an Encoder-Decoder Architecture?

An encoder-decoder architecture is a type of neural network design pattern that is commonly used in tasks like machine translation, sequence-to-sequence prediction, and many others. The architecture consists of two main parts:

1. **Encoder**: This part takes the input sequence (like a sentence in English) and compresses the information into a fixed-length "context vector". 
2. **Decoder**: This takes the context vector and produces the output sequence (like the translated sentence in French).

The encoder "encodes" the input data as an internal fixed-size representation in reduced dimensionality and the decoder then "decodes" it back to some useful representation of output data.

#### Real-World Example

Consider Google Translate translating an English sentence to French. The English sentence is passed through an encoder, turned into a context vector, which is then fed to a decoder to produce the French sentence.

### The Need for Encoder-Decoder

In many sequence-to-sequence tasks, the length of the input sequence can vary and may not be the same as the length of the output sequence. Traditional neural networks are generally not well-suited for handling varying-length sequences as inputs or outputs. Encoder-decoder architectures solve this problem.

### Components of Encoder-Decoder

1. **Input Sequence**: List of symbols represented numerically.
2. **Output Sequence**: Another list of symbols (can be of different length from the input) represented numerically.
3. **Context Vector**: The fixed-length vector that the encoder produces after seeing the entire input sequence.
4. **Hidden States**: Internal states of the RNN/LSTM/GRU (or any other model you're using) in both the encoder and decoder.

### Encoder Forward Pass

1. `input_seq`: This is the input sequence that we want to encode. In our example, it's a NumPy array of 3 random numbers.
2. `weights`: These are the weights of the encoder, represented as a 2x3 NumPy array.
3. `bias`: This is the bias of the encoder, represented as a NumPy array of 2 random numbers.
4. `activation_func`: This is the activation function we use, which in our case is the sigmoid function.

The function performs the following operations:

### Dot Product
First, it calculates the dot product of `weights` and `input_seq`. In linear algebra, the dot product is a way of multiplying each element of one array by its corresponding element in another array and then summing those products.

Mathematically, if our `input_seq` is \([a, b, c]\) and `weights` is a 2x3 matrix:

\[
\begin{bmatrix}
    w1 & w2 & w3 \\
    w4 & w5 & w6
\end{bmatrix}
\]

The dot product would be:

\[
\begin{bmatrix}
    w1 \times a + w2 \times b + w3 \times c \\
    w4 \times a + w5 \times b + w6 \times c
\end{bmatrix}
\]

### Adding Bias
After the dot product, the function adds the `bias` term to the result. Bias allows the model to have some flexibility, enabling it to fit the data better. Adding bias changes the range of the weighted sum before the activation.

### Activation Function
Finally, the activation function is applied to the result. In our case, the sigmoid function is used as the activation function. The sigmoid function maps any input into a value between 0 and 1, which can be useful for binary classification problems.

So, the whole operation can be summarized as:

\[
\text{output} = \text{activation_func}(\text{weights} \cdot \text{input_seq} + \text{bias})
\]

The output of this function will serve as the "context vector" for our simplified encoder. In a more complex scenario like an LSTM-based encoder, this would be the final hidden state of the LSTM after processing the entire input sequence.

### Shapes

The dimensions of the weights in the `encoder_forward_pass` and `decoder_forward_pass` functions are chosen based on the architecture we're aiming to create, specifically the size of the input sequence and the size of the context vector.

### Encoder Weights:

1. **Input Sequence Size**: In our example, the encoder takes an input sequence of size 3. This is represented by the number of columns in the weight matrix.
2. **Context Vector Size**: We decided to use a context vector of size 2 for simplicity. This is represented by the number of rows in the weight matrix.

So, the dimensions of the encoder's weight matrix become `(2, 3)`.

### Decoder Weights:

1. **Context Vector Size**: The decoder takes a context vector of size 2 as input, represented by the number of columns in the weight matrix.
2. **Output Sequence Size**: The decoder's goal is to produce an output sequence of size 3. This is represented by the number of rows in the weight matrix.

So, the dimensions of the decoder's weight matrix become `(3, 2)`.

### Summary:

- Encoder: Transforms an input of size 3 into a context vector of size 2. Hence, its weight dimensions are `(2, 3)`.
- Decoder: Transforms a context vector of size 2 into an output of size 3. Hence, its weight dimensions are `(3, 2)`.

The size of the context vector in an encoder-decoder architecture is a design choice that can depend on various factors such as the complexity of the task, the dimensionality of the input and output sequences, and computational constraints. 

### Why Size 2 for the Context Vector?

In this simple example, the size of 2 for the context vector was an arbitrary choice for demonstration purposes. There's no specific relationship to the input or output size that mandates the context vector to be of size 2. The key idea was to show a "compression" of information: how a sequence of length 3 gets compressed into a smaller representation (length 2 in this case) and then is expanded back to a sequence of length 3.

### Factors Influencing Context Vector Size:

1. **Task Complexity**: More complex tasks may require larger context vectors to capture all the necessary information.
  
2. **Data Dimensionality**: If the input and output sequences have high dimensionality, you might also need a higher-dimensional context vector to adequately capture the relevant information.
  
3. **Computational Constraints**: Larger context vectors will require more computational power and memory.

4. **Overfitting Risks**: A context vector that is too large might lead to overfitting, especially if you have limited data.

### Real-world Example:

In machine translation tasks where the input could be a sentence with tens or even hundreds of words, the context vector often has a much higher dimensionality (e.g., 256, 512, or even more). This is because translating a sentence accurately requires understanding various nuances like context, tone, and semantics, which can be complex to capture.

So, there's no strict rule for choosing the size of the context vector; it's often determined empirically based on the problem you're trying to solve and the data you have.




# Notes:

In an Encoder-Decoder architecture, each part of the network has a different role. The Encoder's job is to encode the input sequence into a "context vector," which captures the essential features of the input. The Decoder's job is to decode this context vector into an output sequence. When training such a network, you need to consider the gradients flowing back from the Decoder's output all the way to the Encoder's input.

Let's break down the backpropagation steps:

1. **Decoder Backpropagation:** The first step is to perform backpropagation on the Decoder network using the actual target sequence (`target_seq`) and the predicted output sequence (`output_seq`). This step computes the gradients for the Decoder's weights and biases (`decoder_grad_weights` and `decoder_grad_biases`).

    \[
    \text{decoder\_grad\_weights, decoder\_grad\_biases} = \text{self.decoder.backward\_propagation}( \text{target\_seq, output\_seq, decoder\_activations, zs\_decoder} )
    \]

2. **Context Delta:** Next, you need to compute how much the loss function is sensitive to changes in the context vector. This sensitivity is captured in `context_delta`. The `context_delta` is calculated by taking the dot product of the transpose of the first set of weights of the Decoder (`self.decoder.weights[0].T`) and the first set of biases gradients from the Decoder (`decoder_grad_biases[0]`).

    \[
    \text{context\_delta} = \text{np.dot}( \text{self.decoder.weights}[0].\text{T}, \text{decoder\_grad\_biases}[0] )
    \]

   Intuition: You're essentially saying, "Given how the Decoder's output would change (in terms of loss) if I tweaked the Decoder's biases, how would that change propagate back to the context vector?"

3. **Encoder Backpropagation:** Finally, you perform backpropagation on the Encoder network, treating `context_delta` as the target. This step computes the gradients for the Encoder's weights and biases (`encoder_grad_weights` and `encoder_grad_biases`).

    \[
    \text{encoder\_grad\_weights, encoder\_grad\_biases} = \text{self.encoder.backward\_propagation}( \text{context\_delta, encoder\_activations}[-1], \text{encoder\_activations, zs\_encoder} )
    \]

### Why is `context_delta` the "right" context vector?

It's not that `context_delta` is the "right" context vector; rather, it captures how the loss function would change if the Encoder's output (context vector) were tweaked slightly. It's a measure of the sensitivity of the loss function to the context vector. By feeding this back into the Encoder during backpropagation, you're helping the Encoder adjust its weights and biases in a way that would minimize the loss in the future.

In essence, `context_delta` serves as a signal that tells the Encoder, "Here's how you should update your parameters to reduce the error next time you encode a sequence."

### Intuition Behind Encoder-Decoder vs Standard Multilayer Network

#### Encoder-Decoder Model:

An encoder-decoder model is designed to learn a compressed, lower-dimensional representation of the input data (encoding) and then reconstruct the original data from this compressed form (decoding). This architecture is particularly useful for tasks like dimensionality reduction, data denoising, and generative modeling.

#### Standard Multilayer Network:

A standard multilayer neural network (also known as a feedforward neural network) is designed for tasks like classification or regression. It takes input data and maps it directly to an output, such as a class label or a continuous value, without explicitly learning a compressed representation of the data.

### Why Encoder-Decoder Might Be Better

1. **Dimensionality Reduction**: Encoder-decoder models are explicitly designed for learning a lower-dimensional representation of the data. This is useful for visualizing high-dimensional data or for preprocessing data for other machine learning tasks.

2. **Data Reconstruction**: Encoder-decoder models can reconstruct input data, which can be useful for tasks like data denoising or inpainting. Standard multilayer networks are not designed for this.

3. **Feature Learning**: The encoder part of the model learns a set of features that capture the underlying structure of the data. These features can be used for other tasks, offering a form of transfer learning.

4. **Generative Capabilities**: While not the focus of your question, it's worth noting that encoder-decoder architectures can be extended to generative models like Variational Autoencoders (VAEs) or Generative Adversarial Networks (GANs).

5. **Flexibility**: Encoder-decoder models can handle a variety of input and output types (e.g., sequences, images, etc.), making them more versatile for different kinds of data.

