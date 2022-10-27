import torch
import numpy as np
import pyro
from pyro.infer import config_enumerate, infer_discrete
import pyro.distributions as dist
from pyro.contrib.epidemiology import CompartmentalModel, binomial_dist, infection_dist



class Model:

    def __init__(self, params):
        self.params = params

    @config_enumerate
    def global_model(self):
        with pyro.plate("latents"):
            rho0 = pyro.sample("rho0", dist.Beta(2, 4))
            rho1 = pyro.sample("rho1", dist.Beta(4, 4))
            rho2 = pyro.sample("rho2", dist.Beta(8, 4))

            switch_to_rho_1 = pyro.sample("switch_to_rho_1", dist.Uniform(15, 40))
            switch_to_rho_2 = pyro.sample("switch_to_rho_2", dist.Uniform(30, 60))

            rho = pyro.deterministic("rho", torch.cat(
                [
                    rho0.unsqueeze(-1).expand(rho0.shape + (round(switch_to_rho_1.item()),)),
                    rho1.unsqueeze(-1).expand(
                        rho1.shape + (round(switch_to_rho_2.item()) - round(switch_to_rho_1.item()),)),
                    rho2.unsqueeze(-1).expand(rho2.shape + (60 - round(switch_to_rho_2.item()),)),
                ],
            ))
            tau = pyro.sample("tau", dist.Categorical(torch.ones(8)), infer={"enumerate": "parallel"})
            tau = tau.squeeze() + 2
            R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

        return rho, tau, R0

    def quantize(self, name, x_real, min, max):
        """
        Randomly quantize in a way that preserves probability mass.
        We use a piecewise polynomial spline of order 3.
        """
        assert min < max
        lb = x_real.detach().floor()

        # This cubic spline interpolates over the nearest four integers, ensuring
        # piecewise quadratic gradients.
        s = x_real - lb
        ss = s * s
        t = 1 - s
        tt = t * t
        probs = torch.stack(
            [
                t * tt,
                4 + ss * (3 * s - 6),
                4 + tt * (3 * t - 6),
                s * ss,
            ],
            dim=-1,
        ) * (1 / 6)
        q = pyro.sample("Q_" + name, dist.Categorical(probs)).type_as(x_real)

        x = lb + q - 1
        x = torch.max(x, 2 * min - 1 - x)
        x = torch.min(x, 2 * max + 1 - x)

        return pyro.deterministic(name, x)

    @config_enumerate
    def model(self):
        population, data = self.params

        rho, tau, R0 = self.global_model()

        # Sample reparameterizing variables.
        S_aux = pyro.sample(
            "S_aux",
            dist.Uniform(-0.5, population + 0.5)
                .mask(False)
                .expand(data.shape)
                .to_event(1),
        )
        I_aux = pyro.sample(
            "I_aux",
            dist.Uniform(-0.5, population + 0.5)
                .mask(False)
                .expand(data.shape)
                .to_event(1),
        )

        S_curr = torch.tensor(population-1.0)
        I_curr = torch.tensor(1.0)
        beta_curr = torch.tensor(1.0)
        with pyro.plate("time_sequence"):
            for t, y in pyro.markov(enumerate(data), keep=True):
                S_prev, I_prev, beta_prev = S_curr, I_curr, beta_curr
                # sample beta, Susceptible and Infected
                beta_curr = pyro.sample("beta_t_{}".format(t),
                                   dist.LogNormal(beta_prev.log(), 0.1))
                S_curr = self.quantize("S_{}".format(t), S_aux[..., t], min=0, max=population)
                I_curr = self.quantize("I_{}".format(t), I_aux[..., t], min=0, max=population)

                # Reverse computation
                S2I = S_prev - S_curr
                I2R = I_prev - I_curr + S2I

                # compute Rt (Rate of infection) and combined_p
                Rt = R0 * beta_curr
                combined_p = 1- (1 - (Rt / tau) / population) ** I_prev
                with pyro.plate("data_plate"):
                    pyro.sample("s2i_{}".format(t),
                                      dist.ExtendedBinomial(S_curr, combined_p), obs=S2I)
                    pyro.sample("i2r_{}".format(t),
                                      dist.ExtendedBinomial(I_curr, 1 / tau), obs=I2R)

                    # rho = pyro.deterministic("rho_{}".format(t), rho2 if t > switch_to_rho_2 else rho1 if t > switch_to_rho_1 else rho0)
                    pyro.sample("obs_{}".format(t),
                                dist.ExtendedBinomial(S2I, rho[t]),
                                obs=y)



if __name__ == '__main__':
    data = []
    with open("../data/run.data") as f:
        for line in f.readlines():
            measured_infected = float(line.strip().split(" ")[1])
            data.append(measured_infected)
    data = torch.tensor(data)

    pyro.clear_param_store()
    print("Sampling:")
    model = Model((600, data))
    auto_guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model.model))

    adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
    elbo = pyro.infer.TraceEnum_ELBO()
    elbo.loss(model.model, config_enumerate(auto_guide, "parallel"))
    svi = pyro.infer.SVI(model.model, auto_guide, adam, elbo)

    losses = []
    for step in range(10):
        loss = svi.step()
        losses.append(loss)
        if step % 100 == 0:
            print("Elbo loss: {}".format(loss))
    print(losses)
    
    





