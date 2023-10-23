import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_units, activation=torch.relu):
        super(Encoder, self).__init__()
        self.input_units = input_units
        self.activation = activation
        self.layers = nn.ModuleList()

    def add(self, units):
        if len(self.layers) == 0:
            self.layers.append(nn.Linear(self.input_units, units))
        else:
            prev_layer_output = self.layers[-1].out_features
            self.layers.append(nn.Linear(prev_layer_output, units))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # no activation for the last layer
        return x
class Decoder(nn.Module):
    def __init__(self, input_units, activation=torch.relu):
        super(Decoder, self).__init__()
        self.activation = activation
        self.input_units = input_units
        self.layers = nn.ModuleList()

    def add(self, units):
        if len(self.layers) == 0:
            self.layers.append(nn.Linear(self.input_units, units))
        else:
            prev_layer_output = self.layers[-1].out_features
            self.layers.append(nn.Linear(prev_layer_output, units))


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, activation=torch.relu):
        super(EncoderDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.encoder = encoder
        self.decoder = decoder
        self.layers.extend(encoder.layers)
        self.layers.extend(decoder.layers)
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # no activation for the last layer
        return x


