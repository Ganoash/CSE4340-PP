import numpy as np
import pyro.distributions as dist
import pyro.primitives as prim
from pyro import condition
import torch


class Sir_Model:

    def __init__(self, observations, time_steps=60, population_size=600, s_0=599, i_0=1, beta_0=1):
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
        self.tau = prim.sample("tau", dist.Multinomial(1, torch.from_numpy(np.ones(8))))
        self.tau = (self.tau == 1).nonzero().squeeze() + 2
        self.R0 = prim.sample("R0", dist.LogNormal(0.0, 1.0))

        self.rho0 = prim.sample("rho0", dist.Beta(2, 4))
        self.rho1 = prim.sample("rho1", dist.Beta(4, 4))
        self.rho2 = prim.sample("rho2", dist.Beta(8, 4))

        self.switch_to_rho_1 = prim.sample("st_rho1", dist.Uniform(15, 40))
        self.switch_to_rho_2 = prim.sample("st_rho2", dist.Uniform(30, 60))

    def run_timestep(self):
        beta_t = prim.sample("beta_t", dist.LogNormal(0.1, self.beta_t[self.current_time_step]))
        self.beta_t.append(beta_t)
        RT = beta_t * self.R0
        print(f"beta_t-1: {self.beta_t[self.current_time_step]}")
        print(f"beta_t: {beta_t}")

        individual_rate = RT / self.tau
        print(f"RT: {RT}")
        print(f"tau: {self.tau}")
        print(f"individual_rate: {individual_rate}")
        p = individual_rate / self.population_size
        print(f"p: {p}")
        combined_p = 1 - ((1 - p) ** self.i_t[self.current_time_step])
        print(f"i_t: {self.i_t[self.current_time_step]}")
        print(f"combined_p: {combined_p}")
        s2i = prim.sample("s2i", dist.Binomial(self.s_t[self.current_time_step], probs=torch.from_numpy(np.array([1-combined_p, combined_p]))))[1]
        i2r = prim.sample("i2r", dist.Binomial(self.i_t[self.current_time_step], probs=torch.from_numpy(np.array([1-(1/self.tau), (1/self.tau)]))))[1]

        print(f"s2i: {s2i}")
        print(f"i2r: {i2r}")
        self.s_t.append(self.s_t[self.current_time_step] - s2i)
        self.i_t.append(self.i_t[self.current_time_step] - i2r + s2i)

        rho_to_use = self.rho2 if self.current_time_step >= self.switch_to_rho_2 else (self.rho1 if self.current_time_step > self.switch_to_rho_1 else self.rho0)
        self.current_time_step += 1
        o_t = prim.sample("observed sick", dist.Binomial(s2i, rho_to_use))
        return o_t

    def run(self):
        while self.current_time_step != self.time_steps:
            conditioned = condition(self.run_timestep, data={"observed sick": self.observations[self.current_time_step]})
            print(prim.sample("cond", conditioned))

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
sir.run()