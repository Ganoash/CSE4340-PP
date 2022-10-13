import numpy as np
import pyro.distributions as dist
from pyro import condition
import torch
import math
import pyro


class SR_Model:

    def __init__(self):
        self.plus = lambda a, b: a + b, '+'
        self.multiply = lambda a, b: round(a*b,0), '*'
        self.divide = lambda a, b: round(a/b, 0), '/'
        self.minus = lambda a, b: a - b, '-'
        self.power = lambda a, b: math.pow(a,b), '**'
        self.binaryOps = [self.plus, self.multiply, self.divide, self.minus, self.power]
        self.identity = lambda x: x, 'x'

    def randomConstantFunction(self):
        tensor = pyro.sample("c", dist.Multinomial(1, torch.from_numpy(np.arange(10))))
        c = 0
        for i in range(10):
            if tensor[i] == 1:
                c = i
        return lambda x: c, str(c)

    def randomCombination(self, f, g):
        tensor = pyro.sample("binaryOps", dist.Multinomial(1, torch.from_numpy(np.arange(5))))
        index = 0
        for i in range(5):
            if tensor[i] == 1:
                index = i
        op = self.binaryOps[index]
        opfn = op[0]
        ffn = f[0]
        gfn = g[0]

        return lambda x: opfn(ffn(x), gfn(x)), f[1] + op[1] + g[1]

    def randomArithmeticExpression(self):
        if pyro.sample("d1", dist.Bernoulli(0.5)):
            return self.randomCombination(self.randomArithmeticExpression(), self.randomArithmeticExpression())
        else:
            if pyro.sample("d2", dist.Bernoulli(0.5)):
                return self.identity
            else:
                return self.randomConstantFunction()

    # Not sure how the last 4 lines of the model should be translated, this is how I do it (for now)
    def run(self, data_y, data_x=None ):
        e = self.randomArithmeticExpression()
        f = e[0]

        for i in range(len(data_x)):
            func = f(data_x[i])
            pyro.sample('f_{}'.format(i), dist.Normal(data_y[i], 5), obs=func)

        print(e)
        print(e[1])
        return e[1]

    @classmethod
    def create_from_file(cls, filename):
        data_x = []
        data_y = []
        file = open(filename, 'r')
        lines = file.readlines()

        for line in lines:
            pair = line.split(" ")
            x = int(pair[0])
            y = int(pair[1])

            data_x.append(x)
            data_y.append(y)

        return data_x, data_y


data_x, data_y = SR_Model.create_from_file("../data/sir.data")
sr = SR_Model()
sr.run(data_y, data_x)