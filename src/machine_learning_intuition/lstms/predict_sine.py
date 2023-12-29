import numpy as np
import torch
import torch.nn as nn

# Set the random seed for reproducibility
torch.manual_seed(0)

# Generate sine wave data
timesteps = 100  # Number of timesteps we're looking into the future.
x = np.linspace(0, 2 * np.pi, timesteps)
data = np.sin(x)

# Convert data to PyTorch tensors and reshape for LSTM input
data = torch.FloatTensor(data).view(-1, 1, 1)


# Define the LSTM model
class SineWaveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]  # We only care about the last prediction


# Instantiate the model, define the loss function and the optimizer
model = SineWaveLSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the LSTM
epochs = 150
for i in range(epochs):
    for seq in range(timesteps - 1):
        # Prepare the sequence and target
        seq_input = data[seq:seq + 1]  # Input sequence
        target = data[seq + 1]  # Target sequence

        # Reset the hidden cell state for each sequence
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        # Zero out the gradient
        optimizer.zero_grad()

        # Get the model's prediction and compute the loss
        y_pred = model(seq_input)
        loss = loss_function(y_pred, target)

        # Backpropagate the loss, compute the gradient, and update the weights
        loss.backward()
        optimizer.step()

    # Print out the loss periodically
    if i % 25 == 0:
        print(f'Epoch {i} Loss: {loss.item()}')

# Test the LSTM's predictions
with torch.no_grad():
    seq_input = data[:1]  # Start with the first input point
    preds = []

    for _ in range(timesteps):
        pred = model(seq_input)
        preds.append(pred.item())
        seq_input = torch.cat((seq_input[1:], pred.unsqueeze(0)))

# Plot the results
import matplotlib.pyplot as plt

plt.plot(x, data.view(-1).numpy(), label='Actual Sine Wave')
plt.plot(x, preds, label='LSTM Predictions')
plt.legend()
plt.show()
