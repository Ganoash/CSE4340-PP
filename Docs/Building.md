# Building
The given inference was too slow, so we instead implemented Variational Inference because that is much faster. 

Because of the way the priors are created from the posterior of the previous step, we have to sample from the result given by the variational inference. 
This is shown by the time the inference took:

|                               | Given | Ours 5000 samples | Ours 1000 samples | Ours 500 samples  | Ours 50 samples |
|-------------------------------|-------|-------------------|-------------------|-------------------|-----------------|
| Average time taken in seconds | 183.4 | 134.2             | 114.8             | 114.1             | 121.6           |
To create this table each method was ran with 5 steps, before replacing the line with the given observations. The average is taken over 3 runs. 

Because both 500 and 100 performed relatively closely we chose to take 1000 samples, as to hopefully have a more accurate result. 
The table shows that the variational inference is significantly faster.