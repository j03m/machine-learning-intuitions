import numpy as np

# Rewards for each timestep in an episode
rewards = np.array([1, -1, 2, 0, 1])

# Total return for the entire episode
total_return = np.sum(rewards)

# Reward-to-go for each timestep
reward_to_go = np.array([np.sum(rewards[t:]) for t in range(len(rewards))])

print("Total Return (R(τ)):", total_return)
print("Reward-to-Go (R̂_t) for each timestep:", reward_to_go)
