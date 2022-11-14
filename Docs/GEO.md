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
