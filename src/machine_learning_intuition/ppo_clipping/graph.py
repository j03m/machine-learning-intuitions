import numpy as np
import matplotlib.pyplot as plt

# Parameters
epsilon = 0.2
ratios = np.linspace(0, 2, 100)
advantage = 1  # Positive advantage for simplicity

# PPO objective components
unclipped = ratios * advantage
clipped = np.clip(ratios, 1-epsilon, 1+epsilon) * advantage
objective = np.minimum(unclipped, clipped)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ratios, unclipped, label='Unclipped Objective')
plt.plot(ratios, clipped, label='Clipped Objective')
plt.plot(ratios, objective, label='PPO Objective', linestyle='--')
plt.xlabel('Ratio of new/old policy probabilities')
plt.ylabel('Objective value')
plt.title('PPO Clipping Mechanism')
plt.legend()
plt.grid(True)
plt.show()
