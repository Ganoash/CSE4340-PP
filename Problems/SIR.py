import numpy as np
import pyro
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc import MCMC
import pyro.distributions as dist
import pyro.primitives as prim
from pyro import condition
import torch


class Sir_Model:

    def __init__(self, observations, time_steps=60, population_size=600, s_0=torch.tensor(599), i_0=torch.tensor(1), beta_0=torch.tensor(1)):
        self.time_steps = time_steps
        self.current_time_step = 0
        self.population_size = population_size
        self.s_0 = s_0
        self.i_0 = i_0
        self.observations = [i_0] + observations
        self.beta_0 = beta_0
        self.s_t = [s_0]
        self.i_t = [i_0]
        self.beta_t = [beta_0]
        self.tau = None
        self.R0 = None
        self.rho0 = None
        self.rho1 = None
        self.rho2 = None
        self.switch_to_rho_1 = None
        self.switch_to_rho_2 = None
        self.draw()

    def draw(self):
        self.tau = prim.sample("tau", dist.Categorical(torch.from_numpy(np.ones(8))), infer={"enumerate":"parallel"})
        self.tau = self.tau.squeeze() + 2
        self.R0 = prim.sample("R0", dist.LogNormal(0.0, 1.0))

        self.rho0 = prim.sample("rho0", dist.Beta(2, 4))
        self.rho1 = prim.sample("rho1", dist.Beta(4, 4))
        self.rho2 = prim.sample("rho2", dist.Beta(8, 4))

        self.switch_to_rho_1 = prim.sample("st_rho1", dist.Uniform(15, 40))
        self.switch_to_rho_2 = prim.sample("st_rho2", dist.Uniform(30, 60))

    def run_timestep(self):
        beta_t = prim.sample(f"beta_{self.current_time_step}", dist.LogNormal(self.beta_t[self.current_time_step].log(), 0.1))
        self.beta_t.append(beta_t)
        RT = beta_t * self.R0

        individual_rate = RT / self.tau
        #print(f"RT: {RT}")
        #print(f"tau: {self.tau}")
        #print(f"individual_rate: {individual_rate}")
        p = individual_rate / self.population_size
        #print(f"p: {p}")
        combined_p = 1 - ((1 - p) ** self.i_t[self.current_time_step])
        #print(f"i_t: {self.i_t[self.current_time_step]}")
        #print(f"combined_p: {combined_p}")


        s2i = prim.sample(f"s2i_{self.current_time_step}", dist.ExtendedBinomial(torch.tensor(self.s_t[self.current_time_step], dtype=torch.float), combined_p))
        i2r = prim.sample(f"i2r_{self.current_time_step}", dist.ExtendedBinomial(torch.tensor(self.i_t[self.current_time_step], dtype=torch.float), (1/self.tau)))
        #print(f"s2i: {s2i}")
        #print(f"i2r: {i2r}")
        self.s_t.append(self.s_t[self.current_time_step] - s2i)
        self.i_t.append(self.i_t[self.current_time_step] - i2r + s2i)

        rho_to_use = self.rho2 if self.current_time_step >= self.switch_to_rho_2 else (self.rho1 if self.current_time_step > self.switch_to_rho_1 else self.rho0)
        self.current_time_step += 1
        o_t = prim.sample(f"observed sick_{self.current_time_step}", dist.Binomial(torch.tensor(s2i, dtype=torch.float), rho_to_use), obs=torch.tensor(self.observations[self.current_time_step], dtype=torch.float))
        return o_t

    def run(self):
        self.draw()
        self.current_time_step = 0
        for _ in pyro.markov(range(self.time_steps)):
            self.run_timestep()

    @classmethod
    def create_from_file(cls, file_location):
        observations = []
        with open(file_location) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        for line in lines:
            obs = int(line.split(" ")[1])
            observations.append(obs)
        return Sir_Model(observations, time_steps=len(observations))

sir = Sir_Model.create_from_file("../data/sir.data")
torch.set_default_dtype(torch.float64)
# nuts_kernel = NUTS(sir.run)
# mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=50)
# mcmc.run()

samples = mcmc.summary()
print(samples)
