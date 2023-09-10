from ..perceptron import MultiLevelPerceptron

class EncodedDecoder:
    def __init__(self, encoder_layers, decode_layers):
        self.encoder = MultiLevelPerceptron(encoder_layers)
        self.decoder = MultiLevelPerceptron(decode_layers)