from ..perceptron import MultiLevelPerceptron
from .. import utils
import numpy as np

mse = utils.mse

# todo test me!
class EncoderDecoder():
    def __init__(self, encoder_layers, decoder_layers, init=True):
        # initialize two perceptrons, one for the encoder, one for the decoder
        # then initialize the autoencoder as one mlp. This allows us to train in union, but will still give us
        # direct predict access on both the encoder and decoder
        if init:
            self.encoder = MultiLevelPerceptron(encoder_layers)
            self.decoder = MultiLevelPerceptron(decoder_layers)
            self.autoencoder = MultiLevelPerceptron(encoder_layers + decoder_layers, init=False)
            self.autoencoder.weights = self.encoder.weights + self.decoder.weights
            self.autoencoder.biases = self.encoder.biases + self.decoder.biases

    def predict(self, input_seq):
        # supply the input sequence to the encoder mlp, produce a context vector
        return self.autoencoder.predict(input_seq)

    def save(self, prefix):
        self.autoencoder.save(f"{prefix}_autoencoder.json")

    @staticmethod
    def load(prefix):
        # layers will be overwritten
        model = EncoderDecoder([], [], init=False)
        model.autoencoder = MultiLevelPerceptron.load(f"{prefix}_autoencoder.json")

        encoder_length = len(model.autoencoder.layers) // 2   # Assuming encoder and decoder are of equal length
        model.encoder = MultiLevelPerceptron(model.autoencoder.layers[:encoder_length], init=False)
        model.decoder = MultiLevelPerceptron(model.autoencoder.layers[encoder_length:], init=False)

        model.encoder.weights = model.autoencoder.weights[:encoder_length]
        model.encoder.biases = model.autoencoder.biases[:encoder_length]
        model.decoder.weights = model.autoencoder.weights[encoder_length:]
        model.decoder.biases = model.autoencoder.biases[encoder_length:]
        return model

    def train(self, X, y, epochs=1000, learning_rate=0.01, patience_limit=500, warm_up_epochs=500):
        self.autoencoder.train(X, y, epochs, learning_rate, patience_limit, warm_up_epochs)
