import torch

import pyro.distributions as dist
import pyro.primitives as prim


def sample_from_discrete_uniform(name: str, values: list) -> int:
    num = len(values)
    discrete_uniform = dist.Categorical(torch.tensor([1/num for _ in values]))
    sample_tensor = prim.sample(name=name, fn=discrete_uniform)
    if (not isinstance(sample_tensor, torch.Tensor) or
        sample_tensor.size() != torch.Size([]) or
            type(sample_tensor.item()) != int):
        raise ValueError("utils: Unnexpected return value of sample statement"
                         " for Discrete Uniform distribution.")
    index = sample_tensor.item()
    return values[index]
