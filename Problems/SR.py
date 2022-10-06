import numpy as np
import pyro.distributions as dist
from pyro import condition
import torch
import math
import pyro


class SR_Model:

    def __init__(self, pairs):
        self.pairs = pairs
        self.plus = lambda a, b: a + b, '+'
        self.multiply = lambda a, b: round(a*b,0), '*'
        self.divide = lambda a, b: round(a/b, 0), '/'
        self.minus = lambda a, b: a - b, '-'
        self.power = lambda a, b: math.pow(a,b), '**'
        self.binaryOps = [self.plus, self.multiply, self.divide, self.minus, self.power]
        self.identity = lambda x: x, 'x'

    def randomConstantFunction(self):
        c = pyro.sample("c", dist.Multinomial(1, torch.from_numpy(np.arange(10))))
        return lambda x: c, c

    def randomCombination(self, f, g):
        index = pyro.sample("binaryOps", dist.Multinomial(1, torch.from_numpy(np.arange(5))))
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
    def run(self):
        e = self.randomArithmeticExpression()
        f = e[0]

        for (x_i, y_i) in self.pairs:
            f_x_i = pyro.sample("f_x_i", dist.Normal(y_i, 5))
            condition(f_x_i, data={"f_x_i": f(x_i)})

        return e[1]

    @classmethod
    def create_from_file(cls, filename):
        pairs = []
        file = open(filename, 'r')
        lines = file.readlines()

        for line in lines:
            pair = line.split(" ")
            x = pair[0]
            y = pair[1]

            pairs.append((int(x), int(y)))

        return SR_Model(pairs)
