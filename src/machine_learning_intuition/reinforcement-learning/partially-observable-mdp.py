import numpy as np

# Define potential grid sizes and initial beliefs about the grid size
grid_sizes = [(3, 3), (4, 4), (5, 5)]
initial_beliefs = np.array([1 / 3, 1 / 3, 1 / 3])  # Equal belief in each size

# Start in the middle of the largest potential grid
initial_position = (2, 2)
state = {'position': initial_position, 'grid_size': None}

# Actions and their effects
actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def transition(state, action):
    pos = state['position']
    new_pos = (pos[0] + actions[action][0], pos[1] + actions[action][1])

    # Test the new position against all possible grid sizes
    valid_positions = [new_pos[0] < size[0] and new_pos[1] < size[1] and new_pos[0] >= 0 and new_pos[1] >= 0 for size in
                       grid_sizes]

    if any(valid_positions):
        return {'position': new_pos, 'grid_size': state['grid_size']}  # Stay in bounds
    else:
        return state  # Hit boundary, stay in place


def observation_function(state):
    # Simple observation: just the position (no direct info about grid size)
    return state['position']


# Simulate one step
current_state = state
action = 'right'  # Choose an action
next_state = transition(current_state, action)
observation = observation_function(next_state)

print("Action:", action)
print("New Position:", next_state['position'])
print("Observation:", observation)
