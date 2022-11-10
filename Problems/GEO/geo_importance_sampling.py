import sys
import torch

from .geo import GeoModel, read_from_file
from pyro.infer import Importance
from pyro.distributions import Empirical


def run():
    # read (from stdin) the location of the input file
    _, file_location, num_of_steps, *_ = sys.argv
    num_of_steps = int(num_of_steps)

    # load the observations from the input file
    heads, reference_times, observed_deformations = read_from_file(
        file_location)

    # initialize the model
    model = GeoModel(heads, reference_times, observed_deformations)

    # perform inference
    importance = Importance(model.model_importance_sampling,
                            guide=None,
                            num_samples=num_of_steps)
    traces = importance.run(observed_deformations)

    samples = []
    for i in traces.exec_traces:
        samples.append(i.nodes["_RETURN"]["value"])
    emp = Empirical(
        torch.tensor(samples), torch.tensor(traces.log_weights))
    inf_kv, inf_sskv, inf_sske, inf_nclay = emp.mean
    print("Inferred latent variables:")
    print("    kv:", inf_kv.item())
    print("    sskv:", inf_sskv.item())
    print("    sske", inf_sske.item())
    print("    nclay", inf_nclay.item())
