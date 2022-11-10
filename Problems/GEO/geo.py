from .deformation import calc_deformation
from .utils import sample_from_discrete_uniform
import numpy
import torch

import pyro
import pyro.distributions as dist
import pyro.primitives as prim


class GeoModel:
    def __init__(self, heads, reference_times, observed_deformations):
        # initializations
        self.heads = heads
        self.reference_times = reference_times
        self.observed_deformations = torch.FloatTensor(observed_deformations)

    @pyro.infer.config_enumerate
    def model(self):
        kv = prim.sample(name="kv", fn=dist.Cauchy(loc=-5, scale=3)).item()
        sskv = prim.sample(name="sskv",
                           fn=dist.Cauchy(loc=-3.5, scale=3)).item()
        sske = prim.sample(name="sske", fn=dist.Cauchy(loc=-5, scale=3)).item()
        nclay = sample_from_discrete_uniform(name="nclay",
                                             values=list(range(5, 11)))
        claythick = 5
        interp_times, defm, _, _ = calc_deformation(
            time=self.reference_times, head=self.heads, Kv=10**kv,
            Sskv=10**sskv, Sske=10**sske, claythick=claythick,
            nclay=nclay)

        aligned_deformation = torch.FloatTensor(
            numpy.interp(self.reference_times, interp_times, defm))

        for i in pyro.plate("data_loop", len(self.observed_deformations)):
            prim.sample(name="data_{}".format(i),
                        fn=dist.Normal(self.observed_deformations[i], 2),
                        obs=aligned_deformation[i])

        return kv, sskv, sske, nclay

    def model_importance_sampling(self, observed_deformations):
        kv = prim.sample(name="kv", fn=dist.Cauchy(loc=-5, scale=3)).item()
        sskv = prim.sample(name="sskv",
                           fn=dist.Cauchy(loc=-3.5, scale=3)).item()
        sske = prim.sample(name="sske", fn=dist.Cauchy(loc=-5, scale=3)).item()
        # print(kv, sskv, sske)
        nclay = sample_from_discrete_uniform(name="nclay",
                                             values=list(range(5, 11)))
        interp_times, defm, _, _ = calc_deformation(
            time=self.reference_times, head=self.heads, Kv=10**kv,
            Sskv=10**sskv, Sske=10**sske, claythick=5, nclay=nclay)

        aligned_deformation = torch.FloatTensor(
            numpy.interp(self.reference_times, interp_times, defm))

        for i in pyro.plate("data_loop", len(observed_deformations)):
            prim.sample(name="data_{}".format(i),
                        fn=dist.Normal(observed_deformations[i], 2),
                        obs=aligned_deformation[i])

        return kv, sskv, sske, nclay


def read_from_file(file_location: str):
    """Read the input data from the given file

    Args:
        file_location (str): The location of the given input file.

    Returns:
        (List[int], List[int], List[int]): Three lists of the same length,
            corresponding to: heads, reference time and observed deformation
            measurments.
    """
    with open(file_location) as f:
        heads = [float(h) for h in f.readline().split()]
        reference_times = [float(h) for h in f.readline().split()]
        observed_deformations = [float(h) for h in f.readline().split()]
    if len(heads) != len(reference_times) != len(observed_deformations):
        raise RuntimeError("Heads, reference times and observed deformations"
                           " were expected to have the same length but they"
                           " didn't.")
    return heads, reference_times, observed_deformations
