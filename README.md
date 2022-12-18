# No U-Turn Sampler

## This repo can be broken down in three main parts 
### 1-Implementation of the different hamiltonian monte carlo samplers
In this section you will find the files hmc.py and nuts.py that both contain sampler implemented in the same standard format. 

In hmc.py there is the standard hmc sampler and a more elaborate varionation AutoHMC.

In nuts.py there is the two algorithms proposed in the paper NaiveNUTS and efficientNUTS


### 2-Rosenbrock Distribution 
A notebook that breaks down the process of finding the minimum of the rosenbrock distribution using the different samplers available.

### 3-Bayesian Logistic Regression 
A notebook with an implementation of Bayesian logistic regression on the German credit dataset. Here we use the different samplers to sample the coefficient from the posterior distribution of the logistic regression model.



