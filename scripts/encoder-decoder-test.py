import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

from machine_learning_intuition.encode_decoder import Encoder, Decoder, EncoderDecoder

import argparse

import matplotlib.pyplot as plt

import numpy as np

import random

g_use_cpu = bool(os.environ.get('USE_CPU', False))

def get_device():
    device = torch.device("cpu")
    if g_use_cpu:
        return device

    # Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        assert torch.backends.mps.is_built()
        device = torch.device("mps")
    else:
        print("WARNING: USING CPU FOR COMPUTATIONS. THIS WILL BE SLOW.")

    # Display the device that will be used
    print(f"Running on device: {device}")

    return device


def plot(model, x_test, y_test, rows, cols, tile, device):
    encoder_model = model.encoder
    x_test = x_test.to(device)
    with torch.no_grad():
        encoder_output = encoder_model(x_test).cpu().numpy()

    plt.subplot(rows, cols, tile)
    plt.scatter(encoder_output[:, 0], encoder_output[:, 1],
                s=20, alpha=0.8, cmap='Set1', c=y_test[0:x_test.shape[0]])
    plt.xlim(-9, 9)
    plt.ylim(-9, 9)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')

    return plt


def get_model_and_device(args):
    encoder = Encoder(784)  # 28 x 28
    encoder.add(192)
    encoder.add(64)
    encoder.add(32)
    encoder.add(2)  # latent dim

    decoder = Decoder(2)  # latent dim
    decoder.add(64)
    decoder.add(128)
    decoder.add(784)  # 28 x 28

    autoencoder = EncoderDecoder(encoder, decoder)

    if os.path.exists(args.model):
        autoencoder.load_state_dict(torch.load(args.model))

    device = get_device()
    autoencoder = autoencoder.to(device)
    return autoencoder, device


def main(args):
    if args.type == "mnist":
        epochs = int(args.epochs)
        mnist_train = MNIST(root='./data',transform=ToTensor(), download=True)
        mnist_test = MNIST(root='./data', train=False, transform=ToTensor())

        x_train = mnist_train.data.float() / 255.
        x_train = x_train.view(x_train.size(0), -1)
        x_test = mnist_test.data.float() / 255.
        x_test = x_test.view(x_test.size(0), -1)

        y_train = mnist_train.targets
        y_test = mnist_test.targets

        train_loader = DataLoader(x_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(x_test, batch_size=64)

        autoencoder, device = get_model_and_device(args)

        loss_fn = nn.MSELoss()
        autoencoder.train()
        optimizer = optim.SGD(autoencoder.parameters(), lr=0.01)
        plot_count = 1
        best_loss = np.inf
        for epoch in range(epochs):
            new_best = False
            for batch in train_loader:
                data = batch
                data = data.to(device)
                optimizer.zero_grad()
                y_pred = autoencoder(data)
                loss = loss_fn(y_pred, data)
                loss.backward()
                optimizer.step()

            if best_loss > loss:
                best_loss = loss
                patience = 0
            else:
                patience += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
                plot(autoencoder, x_test[0:500], y_test[0:500], int(epochs/10), 10, plot_count, device)
                plot_count += 1

            if patience > 100:
                print("Learning seems to have halted")
                break

        plt.savefig("./data/classes.png")
        plt.close()
        torch.save(autoencoder.state_dict(), args.model)
    elif args.type == "render":
        mnist_train = MNIST(root='./data', transform=ToTensor(), download=True)
        mnist_test = MNIST(root='./data', train=False, transform=ToTensor())

        # Normalize pixel values to [0., 1.]
        x_train = mnist_train.data.float() / 255.
        x_test = mnist_test.data.float() / 255.

        # Take a look at the dataset
        n_samples = 10
        idx = random.sample(range(x_train.shape[0]), n_samples)
        plt.figure(figsize=(15, 4))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(x_train[idx[i]].squeeze());
            plt.xticks([], [])
            plt.yticks([], [])
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default="mnist",
                        help="run our auto encoder mnist example")
    parser.add_argument("-e", "--epochs", default=1000,
                        help="number of epochs")
    parser.add_argument("-m", "--model", default="./data/autoencoder.pth", help="path to model")
    args = parser.parse_args()

    main(args)
