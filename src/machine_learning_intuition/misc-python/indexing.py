import numpy as np

# Simulate the output of a network for a batch of 10 states, each with Q-values for 2 actions (0 and 1)
batch_size = 10
action_dim = 2  # Two possible actions

# Fake "network output": Random Q-values for demonstration
# Each row corresponds to a state, and each column to an action's Q-value
np.random.seed(42)  # For reproducible results
fake_network_output = np.random.rand(batch_size, action_dim)

print("Fake network output (Q-values for each state and action):")
print(fake_network_output)

# Simulate random actions taken in each state
# For simplicity, these are just random 0s and 1s, indicating the action taken in each state
actions_taken = np.random.randint(0, action_dim, size=batch_size)

print("\nActions taken (for each state):")
print(actions_taken)

# Use np.arange to create an array of state indices
state_indices = np.arange(0, batch_size)

# Index into the fake network output to select the Q-value for the action taken in each state
selected_q_values = fake_network_output[state_indices, actions_taken]

print("\nSelected Q-values (for the action taken in each state):")
print(selected_q_values)