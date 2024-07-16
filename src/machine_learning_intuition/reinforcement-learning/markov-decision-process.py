import numpy as np

# Define the states and actions
states = list(range(9))  # 0 to 8 representing a 3x3 grid
actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
action_effects = {
    actions['up']: -3,
    actions['down']: 3,
    actions['left']: -1,
    actions['right']: 1
}

# Transition function
def transition(state, action):
    if (action == actions['up'] and state < 3) or \
       (action == actions['down'] and state > 5) or \
       (action == actions['left'] and state % 3 == 0) or \
       (action == actions['right'] and state % 3 == 2):
        return state  # No change if moves are out of bounds
    return state + action_effects[action]

# Reward function
def reward_function(state, action, next_state):
    if next_state == 8:
        return 10  # Goal state with positive reward
    elif state == next_state:
        return -1  # Penalty for hitting walls
    return 0  # No reward otherwise

# Initial state distribution: start at top-left corner of the grid
initial_state_prob = np.zeros(len(states))
initial_state_prob[0] = 1  # 100% chance of starting in state 0

# Example of using the MDP
current_state = np.random.choice(states, p=initial_state_prob)
action = actions['right']  # Choose to move right
next_state = transition(current_state, action)
reward = reward_function(current_state, action, next_state)

print(f"Current State: {current_state}, Action: 'right', Next State: {next_state}, Reward: {reward}")
