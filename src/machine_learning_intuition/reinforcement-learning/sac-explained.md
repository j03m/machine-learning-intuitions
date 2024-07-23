* Create an actor critic (ac)
  * Like td3 ac has a pi and two q's
  * However, pi is a SquashedGaussianMLPActor
    * Like td3 it has a linear network (net)
    * But then it also two additional separate networks
      * mu - (hidden_sizes[-1], act_dim)
      * log_std - (hidden_sizes[-1], act_dim)
* Clone the actor critic (ac_targ)
* 
