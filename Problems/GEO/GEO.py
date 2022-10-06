import numpy as np
import pyro.distributions as dist
import pyro.primitives as prim
from pyro import condition
import torch


class GeoModel:
    def __init__(self):
        # initializations
        self.draw()

    def draw(self):
        pass
        # sampling

    def run(self):
        pass
        # run inference


def read_from_file(file_location: str):
    with open(file_location) as f:
        pass
        # read inputs
    return None  # inputs


if __name__ == "__main__":
    # read (from stdin) the location of the input file
    file_location = input()

    # load the inputs from the input file
    inputs = read_from_file(file_location)

    # initialize the model
    model = GeoModel(inputs)

    # perform inference
    model.run()
