* Create an actor critic (ac)
  * Like td3 ac has a pi and two q's
  * However, pi is a SquashedGaussianMLPActor
    * Like td3 it has a linear network (net)
    * But then it also two additional separate networks
      * mu - (hidden_sizes[-1], act_dim)
      * log_std - (hidden_sizes[-1], act_dim)
    * The idea here is that working in unison, the observations are still processed them to predict the best possible actions. However, instead of directly predicting a single "best" action, it predicts parameters (mean, µ, and standard deviation, σ) that describe a range of possible actions.
    * Why a Distribution?: In real life, there's often not a single correct answer. For example, if you're trying to avoid an obstacle, you might have several viable paths. By predicting a distribution (a Normal distribution in this case), the robot isn't limited to one predetermined path; instead, it can explore multiple paths and learn which ones lead to better outcomes. 
    * The network outputs two key parameters: µ (mu) and σ (sigma). µ represents the average or most likely action the robot thinks is best based on its current observations. σ represents how uncertain the robot is about this action. A large σ means the robot is unsure, so it should consider a broader range of actions around µ.
    * We still need to return an actual action to learn, so Sampling: Based on the predicted µ and σ, the robot then randomly picks an action. This randomness helps the robot explore different strategies rather than sticking to what it thinks is best.
    * Squashing: Actions in real-world scenarios often need to be within certain limits (e.g., a steering angle between -30 and 30 degrees). The robot uses the tanh function to squash the sampled actions to ensure they stay within these practical limits.
* Clone the actor critic (ac_targ)
*  
