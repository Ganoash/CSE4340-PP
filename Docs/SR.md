#Symbolic Regression model
Steps taken:
1. Create model by translating the given model to Pyro.
   1. A big issue for making this model work is the fact that we branch on the Bernoulli samples. Therefore, the support is not equal for each trace.  
2. Tried to make NUTS work, but this did not work because of the discrete variables.
3. Tried to make HMC work before figuring out that NUTS did not work because of the discrete variables. Therefore failed because of the same reasons.
   1. Online sources state that NUTS/HMC should be able to work using enumeration, but I run into problems because of the branching on Bernoulli samples. These should be taken out, but I do not know how that would be possible. 
4. Tried to make SVI work, because online there are sources that can make it work with the discrete variables and it also works for the other models. 
   1. Cannot make it work because I do not know how to hide the variables that have different name_counts each trace.
5. Currently trying to make importance sampling work, as that should work with this model.
   1. Importance sampling now works. Normally importance sampling uses an EmpiricalMarginal, but since the formulas are discrete and not numbers this is not possible. Therefore, we have to sort the samples on weight (highest weight first) and then take the 5 samples with the highest weights as output. 
   2. A few different numbers of samples were tried. 
      1. To check if it worked 10-100. Everything under 100 is not able to find the exact formula, but is usually very close (think (* x 2) or (+ x x) instead of (* x 3)).
      2. To see if it finds a correct formula 500. With 500 samples it was able to find the exact formula for easy formulas (like (* x 3), and a different one which results in the same answers. 
      3. Tried 5000 and 10000, neither finds more difficult (nested) formulas exactly. While they both find the correct operators most of the time, they do not always find the correct numbers to use with these operators (think (* (/ x 9) 5) instead of (* (/ x 5) 3)). 10000 takes too long, so settled on 5000.
