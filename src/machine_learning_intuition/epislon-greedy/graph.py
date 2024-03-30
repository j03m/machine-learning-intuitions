import matplotlib.pyplot as plt
import numpy as np

# Define the parameters for the epsilon decay function
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps = np.arange(1000)  # Let's plot for 1000 steps

# Calculate eps_threshold for each step
eps_thresholds = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps / EPS_DECAY)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, eps_thresholds, label='Epsilon Threshold')
plt.title('Epsilon Threshold Over Time')
plt.xlabel('Steps')
plt.ylabel('Epsilon Threshold')
plt.grid(True)
plt.legend()
plt.show()
