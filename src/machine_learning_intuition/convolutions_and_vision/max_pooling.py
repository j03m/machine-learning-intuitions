import numpy as np
def max_pooling1d(input_signal, pool_size, stride):
    output_size = (len(input_signal) - pool_size) // stride + 1
    output_signal = np.zeros(output_size)

    for i in range(output_size):
        output_signal[i] = np.max(input_signal[i * stride:i * stride + pool_size])

    return output_signal


# Example input signal
input_signal = np.array([1, 3, 2, 4, 6, 5])

# Perform max pooling
output_signal = max_pooling1d(input_signal, pool_size=2, stride=2)
print("Input Signal: ", input_signal)
print("Output Signal: ", output_signal)

