#Symbolic Regression model
Steps taken:
1. Create model by translating the given model to Pyro.
   1. A big issue for making this model work is the fact that we branch on the Bernoulli samples. 
2. Tried to make NUTS work, but this did not work because of the discrete variables.
3. Tried to make HMC work before figuring out that NUTS did not work because of the discrete variables. Therefore failed because of the same reasons.
4. Currently trying to make SVI work, because online there are sources that can make it work with the discrete variables and it also works for the other models. 