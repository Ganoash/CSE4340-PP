#Symbolic Regression model
Steps taken:
1. Create model by translating the given model to Pyro.
   1. A big issue for making this model work is the fact that we branch on the Bernoulli samples. 
2. Tried to make NUTS work, but this did not work because of the discrete variables.
3. Tried to make HMC work before figuring out that NUTS did not work because of the discrete variables. Therefore failed because of the same reasons.
   1. Online sources state that NUTS/HMC should be able to work using enumeration, but I run into problems because of the branching on Bernoulli samples. These should be taken out, but I do not know how that would be possible. 
4. Tried to make SVI work, because online there are sources that can make it work with the discrete variables and it also works for the other models. 
   1. Cannot make it work because I do not know how to hide the variables that have different name_counts each trace.
5. Currently trying to make importance sampling work, as that should work with this model.