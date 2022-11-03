## Geological Structures

We used **_Variational Inference_** for the "Geological Structures" model.

#### Why Variational Inference?

Pyro offers a quite neat and simple API for Variational Inference (through the `SVI` class), therefore we decided that Variational Inference is the best inference procedure to test first for this problem.

We encountered a number of problems with MCMC and discrete variables in the SR and SIR models, therefore since the "Geological Structures" model also contains the `nclay` discrete variable, we figured out that Variational Inference would be working properly if we managed to hide the discrete variable (which we did).

On top of that, Pyro also offers an API (`AutoGuide`) for automatically generating guides for Variational Inference models. This made it quite easy for us to implement the guide for the inference procedure.