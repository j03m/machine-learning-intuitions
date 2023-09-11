from ..perceptron import MultiLevelPerceptron, mlp_utils
import numpy as np

mse = mlp_utils.mse


class EncoderDecoder:
    def __init__(self, encoder_layers, decoder_layers):
        # initialize two perceptrons, one for the encoder, one for the decoder
        # Layer configs will need to be set up so that they are able to exchange data
        self.encoder = MultiLevelPerceptron(encoder_layers)
        self.decoder = MultiLevelPerceptron(decoder_layers)

    def forward_pass(self, input_seq):
        # supply the input sequence to the encoder mlp, produce a context vector
        context_vector, encoder_activations, zs_encoder = self.encoder.predict(input_seq)

        # supply the context vector to the decode, produce the output
        output_seq, decoder_activations, zs_decoder = self.decoder.predict(context_vector)

        # return all relevant data for backward propagation and training
        return context_vector, output_seq, encoder_activations, decoder_activations, zs_encoder, zs_decoder

    def backward_propagation(self, target_seq, output_seq, encoder_activations, decoder_activations, zs_encoder, zs_decoder):
        # data moves from encoder -> decoder so the decoder is output is last.
        # this augments the decode network first
        decoder_grad_weights, decoder_grad_biases = self.decoder.backward_propagation(
            target_seq, output_seq, decoder_activations, zs_decoder
        )

        # Compute the gradient of the loss with respect to the context vector (output of the encoder)
        # This will be the delta propagated back into the encoder.
        context_delta = np.dot(self.decoder.weights[-1].T, decoder_grad_biases[-1])

        # Now backpropagate through the encoder
        encoder_grad_weights, encoder_grad_biases = self.encoder.backward_propagation(
            context_delta, encoder_activations[-1], encoder_activations, zs_encoder
        )

        return encoder_grad_weights, encoder_grad_biases, decoder_grad_weights, decoder_grad_biases

    def update_parameters(self, decoder_grad_weights, decoder_grad_biases, encoder_grad_weights, encoder_grad_biases, learning_rate):
        self.decoder.update_parameters(decoder_grad_weights, decoder_grad_biases, learning_rate)
        self.encoder.update_parameters(encoder_grad_weights, encoder_grad_biases, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for input_seq, target_seq in zip(X, y):
                context_vector, output_seq, encoder_activations, decoder_activations = self.forward_pass(input_seq)
                loss = mse(target_seq, output_seq)

                (encoder_grad_weights,
                 encoder_grad_biases,
                 decoder_grad_weights,
                 decoder_grad_biases) = self.backward_propagation(
                    target_seq, output_seq, encoder_activations, decoder_activations
                )
                self.update_parameters(decoder_grad_weights, decoder_grad_biases, encoder_grad_weights,
                                       encoder_grad_biases, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
