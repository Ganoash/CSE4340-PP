# Submissions Log

## Symbolic Regression model

### 1st Submission

Steps taken:
1. Create model by translating the given model to Pyro.
   1. A big issue for making this model work is the fact that we branch on the Bernoulli samples. Therefore, the support is not equal for each trace.  
   2. Because theoretically the recursive depth for the model could be infinite, we had to add a maximum depth for the recursion. 
   3. Because a generated function can be impossible (e.g. division by 0), the model is allowed to regenerate a function up to 10 times. After that it just returns the identity funciton. 
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
      3. Tried 5000 and 10000, neither finds more difficult (nested) formulas exactly. While they both find the correct operators most of the time, they do not always find the correct numbers to use with these operators (think (* (/ x 9) 5) instead of (* (/ x 5) 3)). 10000 takes too long.

### 2nd Submission (09/11/22)

No changes made in the SR model. Our 2nd submission introduces a fixup for a Python-related error in our "Geological Structures" model.

### 3rd Submission (14/11/22)

We didn't implement a solution for the "SIR" problem in our 3rd submission.

## SIR model

### 1st Submission

Various attempts were made to get this model to work within pyro. A first attempt was made using the pyro built in [compartment model](https://docs.pyro.ai/en/stable/contrib.epidemiology.html), however this compartment model did not play well with the discrete variables in the global model (Tau mainly). The compartment module rewrites the SIR model to a form of HMM which simplifies inference over the large amount of discrete variables.

A second attempt of getting inference working was done using a custom model. First of all we used MCMC methods for generating estimators, however this led to 2 problems: Discrete Variables are enumerated away, so we cannot get an estimator for these using MCMC. Secondly MCMC samples led to Very High variance solutions.

To fix the issues with discrete latents and high variance we decided to switch to Variational Inference. By fitting a guide to the posterior we can estimate the latents using gradients ascent, and then estimate the most likely value for tau by drawing from the guide.

### 2nd Submission (09/11/22)

No changes made in the SIR model. Our 2nd submission introduces a fixup for a Python-related error in our "Geological Structures" model.

### 3rd Submission (14/11/22)

We didn't implement a solution for the "SIR" problem in our 3rd submission.

## Geological Structures

### 1st Submission

We used **_Variational Inference_** for the "Geological Structures" model.

#### Why Variational Inference?

Pyro offers a quite neat and simple API for Variational Inference (through the `SVI` class), therefore we decided that Variational Inference is the best inference procedure to test first for this problem.

We encountered a number of problems with MCMC and discrete variables in the SR and SIR models, therefore since the "Geological Structures" model also contains the `nclay` discrete variable, we figured out that Variational Inference would be working properly if we managed to hide the discrete variable (which we did).

On top of that, Pyro also offers an API (`AutoGuide`) for automatically generating guides for Variational Inference models. This made it quite easy for us to implement the guide for the inference procedure.

### 2nd Submission (09/11/22)

We noticed that a last minute change in the `geo.sh` script (right before our first submission) introduced a Python module error when executing the code for "Geological Structures". We added a small fix-up commit that kills the error. Below, you can see the ID of the commit that fixes the error:
- `0edb8477e05f4a3f87a31b307761852a22a4917d`

### 3rd Submission (14/11/22)

For our next iteration for the "Geological Structures" model we used **_Importance Sampling_**.

#### Why Importance Sampling?

Pyro offers a simple API for Importance Sampling (through the `Importance` class), therefore we decided that Importance Sampling is a great next step in order to compare our results with the previous submission.

On top of that, we came up with an idea on how to combine the results of Variational Inference from our previous submission with the Importance Sampling algorithm. Specifically, we know that Importance Sampling requires a "guide" distribution from which it samples. Currently, we just use the prior distribution as a guide because of its ease of use, but of course this is not optimal. However, we were trying to find a better way to create a good guide distribution for Importance Sampling and we thought about the following procedure:
1. train a guide using our Variational Inference implementation
2. use it as the guide in our Importance Sampling implementation in order to (possibly) enhance its performance

This current implementation, uses the prior as a guide in Importance Sampling. But, we hope that we can extend it in the near future (if we have sufficient time), so that it leverages the trained guide from our Variational Inference solution.

**_NOTE_**: Later on, we found out that our idea to enhance our Importance Sampling implementation with Variational Inference has already been explored in the past. Here is a relevant link: `https://ieeexplore.ieee.org/document/8553147`


## Building

### 1st Submission

We didn't implement a solution for the "Building" problem in our 1st submission.

### 2nd Submission (09/11/22)

We didn't implement a solution for the "Building" problem in our 2nd submission. Our 2nd submission introduces a fixup for a Python-related error in our "Geological Structures" model.

### 3rd Submission (14/11/22)

The given inference was too slow, so we instead implemented Variational Inference because that is much faster. 

Because of the way the priors are created from the posterior of the previous step, we have to sample from the result given by the variational inference.
This is shown by the time the inference took:

|                               | Given | Ours 5000 samples | Ours 1000 samples | Ours 500 samples  | Ours 50 samples |
|-------------------------------|-------|-------------------|-------------------|-------------------|-----------------|
| Average time taken in seconds | 183.4 | 134.2             | 114.8             | 114.1             | 121.6           |

To create this table each method was run with 5 steps, before replacing the line with the given observations. The average is taken over 3 runs. 

Because both 500 and 100 performed relatively closely we chose to take 1000 samples, as to hopefully have a more accurate result. 
The table shows that the variational inference is significantly faster.

To calculate the estimate of muA and muB, we take the trace that is the output of the final step. From this trace we take 1000 samples. Then we calculate the mean for muA and muB from these samples.
