Yes, you've got it right! In the context of actor-critic architectures like DDPG (Deep Deterministic Policy Gradient), there are two main components: the actor (often denoted as \( \pi \)) and the critic (often denoted as \( Q \)). Both components are typically implemented using multi-layer perceptrons (MLPs), but they have different roles and outputs:

### Actor (\( \pi \))
- **Role**: The actor's job is to directly determine the action to take given the current state of the environment. It essentially maps states to actions.
- **Output**: The output of the actor is a specific action, which is a continuous value (or values) in the case of environments with continuous action spaces.
- **Implementation**: As an MLP, the actor network takes the current state as input and produces the action as output. The final activation function (often a `tanh` scaled by the action limit) ensures the action values are within the valid range for the environment.

### Critic (\( Q \))
- **Role**: The critic evaluates the action taken by the actor by estimating the Q-value, which is the expected return (sum of discounted future rewards) starting from the current state and taking a particular action. This value is used to guide the training of the actor.
- **Output**: The output of the critic is a scalar value representing the Q-value of the state-action pair. It does not produce actions but evaluates their expected effectiveness.
- **Implementation**: The critic as an MLP takes both the current state and the action as inputs (often concatenated) and outputs a single scalar Q-value. This network helps assess how good the chosen action is from the current state.

### How They Work Together in DDPG

1. **Action Selection**: During each step of interaction with the environment, the actor network is used to decide the best action to take based on the current state.

2. **Policy Evaluation**: The critic network evaluates this action by estimating the Q-value of the resulting state-action pair. 

3. **Training the Critic**: The critic is trained to minimize the difference between its predicted Q-values and the target Q-values derived from the Bellman equation. The target Q-values are computed using rewards obtained from the environment and the Q-values estimated for the next state, which are provided by a target critic network for stability.

4. **Training the Actor**: The actor is trained to maximize the expected returns as estimated by the critic. Specifically, the actor’s policy is updated to produce actions that maximize the Q-values as estimated by the critic, effectively learning to choose better actions over time.

### Conclusion

In summary, while both the actor and the critic are implemented using MLPs, their roles are distinct:
- The **actor** decides **what action to take**.
- The **critic** estimates **how good that action is**.

This separation allows the system to effectively learn both how to act and how to evaluate actions, facilitating more robust and effective learning, especially in complex environments with continuous action spaces.