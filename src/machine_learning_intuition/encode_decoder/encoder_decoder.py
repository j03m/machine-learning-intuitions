from ..perceptron import MultiLevelPerceptron, mlp_utils
import numpy as np

mse = mlp_utils.mse


class EncoderDecoder:
    def __init__(self, encoder_layers, decoder_layers):
        # initialize two perceptrons, one for the encoder, one for the decoder
        # Layer configs will need to be set up so that they are able to exchange data
        self.encoder = MultiLevelPerceptron(encoder_layers)
        self.decoder = MultiLevelPerceptron(decoder_layers)

    def predict(self, input_seq):
        # supply the input sequence to the encoder mlp, produce a context vector
        context_vector, encoder_activations, zs_encoder = self.encoder.predict(input_seq)

        # supply the context vector to the decode, produce the output
        output_seq, decoder_activations, zs_decoder = self.decoder.predict(context_vector)

        # return all relevant data for backward propagation and training
        return context_vector, output_seq, encoder_activations, decoder_activations, zs_encoder, zs_decoder

    def backward_propagation(self, target_seq, output_seq, encoder_activations, decoder_activations, zs_encoder,
                             zs_decoder):
        # data moves from encoder -> decoder so the decoder is output is last.
        # this augments the decode network first
        decoder_grad_weights, decoder_grad_biases = self.decoder.backward_propagation(
            target_seq, output_seq, decoder_activations, zs_decoder
        )

        # Compute the gradient of the loss with respect to the context vector (output of the encoder)
        # This will be the delta propagated back into the encoder.
        context_delta = np.dot(self.decoder.weights[0].T, decoder_grad_biases[0])

        # Now back propagate through the encoder
        encoder_grad_weights, encoder_grad_biases = self.encoder.backward_propagation(
            context_delta, encoder_activations[-1], encoder_activations, zs_encoder
        )

        return encoder_grad_weights, encoder_grad_biases, decoder_grad_weights, decoder_grad_biases

    def update_parameters(self, decoder_grad_weights, decoder_grad_biases, encoder_grad_weights, encoder_grad_biases,
                          learning_rate):
        self.decoder.update_parameters(decoder_grad_weights, decoder_grad_biases, learning_rate)
        self.encoder.update_parameters(encoder_grad_weights, encoder_grad_biases, learning_rate)

    def save(self, prefix):
        self.encoder.save(f"{prefix}_encoder.json")
        self.decoder.save(f"{prefix}_decoder.json")

    @staticmethod
    def load(prefix):
        # layers will be overwritten
        model = EncoderDecoder([1, 1, 1], [1, 1, 1])
        model.encoder = MultiLevelPerceptron.load(f"{prefix}_encoder.json")
        model.decoder = MultiLevelPerceptron.load(f"{prefix}_decoder.json")
        return model

    def train(self, X, y, epochs=1000, learning_rate=0.01, patience_limit=500, warm_up_epochs=500):
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            for input_seq, y_true in zip(X, y):
                (context_vector,
                 y_pred,
                 encoder_activations,
                 decoder_activations,
                 zs_encoder,
                 zs_decoder) = self.predict(input_seq)

                y_true = y_true.reshape(-1, 1)
                loss = mse(y_true, y_pred)

                (encoder_grad_weights,
                 encoder_grad_biases,
                 decoder_grad_weights,
                 decoder_grad_biases) = self.backward_propagation(
                    y_true, y_pred, encoder_activations, decoder_activations, zs_encoder, zs_decoder
                )
                self.update_parameters(decoder_grad_weights, decoder_grad_biases, encoder_grad_weights,
                                       encoder_grad_biases, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            if epoch >= warm_up_epochs:
                if loss < best_val_loss:
                    best_val_loss = loss
                    patience_counter = 0  # Reset counter
                else:
                    patience_counter += 1  # Increment counter

                if patience_counter >= patience_limit:
                    print("Early stopping due to lack of improvement.")
                    break
