import matplotlib.pyplot as plt
import numpy as np

def update_weights(initial_weight, target_weight, tau, iterations):
    weights = [initial_weight]
    for _ in range(iterations):
        initial_weight = tau * target_weight + (1 - tau) * initial_weight
        weights.append(initial_weight)
    return weights

# Parameters
initial_weight = 0  # Starting weight of the target network
target_weight = 1  # Fixed weight of the policy network to simulate convergence
iterations = 50  # Number of iterations to simulate
taus = [0.1, 0.01, 0.001]  # Different values of tau to illustrate

# Plotting
plt.figure(figsize=(10, 6))
for tau in taus:
    weights = update_weights(initial_weight, target_weight, tau, iterations)
    plt.plot(weights, label=f'Tau = {tau}')

plt.title('Target Network Weight Update Rates for Different Tau Values')
plt.xlabel('Iteration')
plt.ylabel('Target Network Weight')
plt.legend()
plt.grid(True)
plt.show()
