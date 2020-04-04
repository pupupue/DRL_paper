# DRL_paper




DDPG learning to addapt a drl algorythm from a paper
created by janis laurins
author is machinelearningguy or dude on youtube
absolute beastly lovely work from him!!!

Required stuff for working algorythm: 

> tf 1.0

tf 2.0+ breaks and to fix you need to use depreciated methods

Need a replay buffer class ## going to store replay data for uniform update selection
> also its going to be off policy so it can be large

Need a class for a target Q network (funtion of s, a)
We will use batch normalization (to have same unit mean and variance)
We have two actor and two critic networks, a target for each.
Updates are soft, according to theta' = tau*theta + (1-tau)*theta', with tau << 1
add exploration as seperate variable noise to actor policy maybe auto correlated noise algo
THE TARGET ACTOR IS JUST THE EVALUATION ACTOR PLUS SOME NOISE PROCESS
ornstein uklenbeck noise look that up

class for a replay network
class for actor
class for critic
+ class for noise 
