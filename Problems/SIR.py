import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.contrib.epidemiology import CompartmentalModel, binomial_dist, infection_dist

class SIRModel(CompartmentalModel):

    def __init__(self, T, population, S_0, I_0, beta_0, data):
        compartments = ("S", "I", "beta")
        super().__init__(compartments, T, population)
        self.S_0 = S_0
        self.I_0 = I_0
        self.beta_0 = beta_0
        self.data = data

    def global_model(self):
        tau = pyro.sample("tau", dist.Categorical(torch.ones(8)))
        tau = pyro.deterministic("tau_value", tau + 2)

        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

        rho0 = pyro.sample("rho0", dist.Beta(2, 4))
        rho1 = pyro.sample("rho1", dist.Beta(4, 4))
        rho2 = pyro.sample("rho2", dist.Beta(8, 4))

        switch_to_rho_1 = pyro.sample("switch_to_rho_1", dist.Uniform(15, 40))
        switch_to_rho_2 = pyro.sample("switch_to_rho_2", dist.Uniform(30, 60))

        st0 = lambda t: switch_to_rho_1 < t
        st2 = lambda t: switch_to_rho_2 >= t
        st1 = lambda t: ~st0(t)&~st2(t)

        rho_t = lambda t: torch.add(torch.add(torch.where(st0(t), rho0, 0), torch.where(st1(t), rho1, 0)), torch.where(st2(t), rho2, 0))
        if len(rho0.size()) > 0:
            rho = torch.cat([rho_t(t) for t in range(60)], 0)
        else:
            rho = torch.stack([rho_t(t) for t in range(60)], 0)
        return tau, R0, rho

    def initialize(self, params):
        return {"S": self.S_0, "I": self.I_0, "beta": self.beta_0}

    def compute_flows(self, prev, curr, t):
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I
        beta = curr["beta"]

        return {
            "S2I_{}".format(t): S2I,
            "I2R_{}".format(t): I2R,
            "BETA_{}".format(t): beta,
        }

    def transition(self, params, state, t):
        tau, R0, rho = params
        beta = pyro.sample("BETA_{}".format(t),
                           dist.LogNormal(state["beta"].log(), 0.1))
        Rt = pyro.deterministic("Rt_{}".format(t), R0 * beta)
        # Rt = Rt.expand(tau.size())
        p = pyro.deterministic("ir_{}".format(t), (Rt/tau)/self.population)

        combined_p = pyro.deterministic("combined_p_{}".format(t), 1 - (1-p)**state["I"])

        print(state["S"].type(torch.long))
        print(combined_p.size())
        S2I = pyro.sample("S2I_{}".format(t),
                          dist.Binomial(state["S"].type(torch.long), combined_p))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"].type(torch.long), 1/tau))

        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R
        state["beta"] = beta

        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho[t]),
                    obs=self.data[t] if t_is_observed else None)

def model(params):
    S, I, beta, data = params

    tau = pyro.sample("tau", dist.Categorical(torch.ones(8)), infer={"enumerate":"parallel"})
    tau = tau.squeeze() + 2
    R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

    rho0 = pyro.sample("rho0", dist.Beta(2, 4))
    rho1 = pyro.sample("rho1", dist.Beta(4, 4))
    rho2 = pyro.sample("rho2", dist.Beta(8, 4))

    switch_to_rho_1 = pyro.sample("switch_to_rho_1", dist.Uniform(15, 40))
    switch_to_rho_2 = pyro.sample("switch_to_rho_2", dist.Uniform(30, 60))

    t = 0

    for t, y in pyro.markov(enumerate(data)):
        beta = pyro.sample("beta_t_{}".format(t),
                           dist.LogNormal(beta.log(), 0.1))
        Rt = pyro.deterministic("Rt_{}".format(t), R0 * beta)
        p = pyro.deterministic("ir_{}".format(t), (Rt / tau) / self.population)

        combined_p = pyro.deterministic("combined_p_{}".format(t), 1 - (1 - p) ** state["I"])

        S2I = pyro.sample("s2i_{}".format(t),
                          dist.Binomial(S, combined_p))
        I2R = pyro.sample("i2r_{}".format(t),
                          dist.Binomial(I, 1 / tau))

        S = S - S2I
        I = I + S2I - I2R

        rho = rho2 if t > switch_to_rho_2 else rho1 if t > switch_to_rho_1 else rho0
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho),
                    obs=y)

""""                    obs=y)
@easy_guide(model)
def guide(self, params):
    self.map_estimate("tau")

"""

if __name__ == '__main__':
    data = []
    with open("../data/run.data") as f:
        for line in f.readlines():
            measured_infected = float(line.strip().split(" ")[1])
            data.append(measured_infected)
    data = torch.tensor(data)
    model = SIRModel(60, 600, torch.tensor([599], dtype=torch.long), torch.tensor([1], dtype=torch.long), torch.tensor(1.), data)
    model.fit_mcmc()
    
    





