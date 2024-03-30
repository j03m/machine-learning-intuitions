import numpy as np

# Example RNN output probabilities for a 3-word sentence
# Rows correspond to time steps (words in the sentence)
# Columns correspond to probabilities for each word in the vocabulary (4 words in vocab)
o = np.array([
    [0.1, 0.2, 0.6, 0.1],  # Probabilities for the first word
    [0.3, 0.4, 0.2, 0.1],  # Probabilities for the second word
    [0.2, 0.2, 0.3, 0.3]   # Probabilities for the third word
])

# True labels for each word in the sentence (indices of the actual words)
y_i = np.array([2, 1, 0])  # True word indices for the 3-word sentence

# Generating time step indices
time_step_indices = np.arange(len(y_i))

# Extracting the predicted probabilities corresponding to the true words
predicted_probabilities = o[time_step_indices, y_i]

print(predicted_probabilities, time_step_indices, y_i)

