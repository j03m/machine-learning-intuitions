import numpy as np


def conv1d(input_signal, kernel, stride=1, padding=0):
    # Pad the input signal
    input_signal = np.pad(input_signal, (padding, padding), mode='constant')

    # Calculate the size of the output
    output_size = (len(input_signal) - len(kernel)) // stride + 1
    output_signal = np.zeros(output_size)

    # Perform the convolution
    for i in range(output_size):
        output_signal[i] = np.sum(input_signal[i * stride:i * stride + len(kernel)] * kernel)

    return output_signal


# Example input signal and kernel
input_signal = np.array([1, 2, 3, 4, 5, 6])
kernel = np.array([0.2, 0.5, 0.2])

# Perform convolution
output_signal = conv1d(input_signal, kernel, stride=1, padding=1)
print("Input Signal: ", input_signal)
print("Kernel: ", kernel)
print("Output Signal: ", output_signal)
