import numpy as np

# Initialize state-action values
Q = np.zeros((3, 2))  # 3 states and 2 actions

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.2  # Exploration rate


# Simple policy that chooses action based on epsilon-greedy strategy from Q
def policy(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice([0, 1])  # Explore: Random action
    else:
        return np.argmax(Q[state])  # Exploit: Best action from Q


# Learning from a single episode
state = 0
action = policy(state, Q, epsilon)
while state != 2:
    # Take action, observe new state and reward
    next_state = state + action
    reward = 1 if next_state == 2 else 0

    # Next action from the current policy
    next_action = policy(next_state, Q, epsilon)

    # SARSA Update formula
    Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

    state, action = next_state, next_action

# Print updated Q-values
print("Updated Q-values with SARSA (On-Policy):")
print(Q)

# Initialize state-action values
Q = np.zeros((3, 2))  # 3 states and 2 actions

# on policy
state = 0
while state != 2:
    # Choose action based on current policy (could be different from Q-learning's greedy policy)
    action = policy(state, Q, epsilon)  # Using the same policy function for consistency

    # Take action, observe new state and reward
    next_state = state + action
    reward = 1 if next_state == 2 else 0

    # Q-Learning Update formula
    best_next_action = np.argmax(Q[next_state])  # Best future action (greedy)
    Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

    state = next_state

# Print updated Q-values
print("Updated Q-values with Q-Learning (Off-Policy):")
print(Q)

# Initialize state-action values
Q = np.zeros((3, 2))  # 3 states and 2 actions
