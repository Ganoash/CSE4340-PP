import numpy as np
import pyro.distributions as dist
import torch
import math
import pyro
from pyro.contrib.autoname import name_count
import logging
import pyro.primitives as prim


class SR_Model:

    def __init__(self, data):
        self.plus = lambda a, b: a + b, '+'
        self.multiply = lambda a, b: round(a*b,0), '*'
        self.divide = lambda a, b: round(a/b, 0), '/'
        self.minus = lambda a, b: a - b, '-'
        self.power = lambda a, b: math.pow(a,b), '**'
        self.binaryOps = [self.plus, self.multiply, self.divide, self.minus, self.power]
        self.identity = lambda x: x, 'x'
        self.data = data

    @name_count
    def randomConstantFunction(self):
        c = pyro.sample("c", dist.Categorical(torch.from_numpy(np.ones(10))))
        c = c.item()
        return lambda x: c, str(c)

    def randomCombination(self, f, g):
        index = pyro.sample("binaryOps", dist.Categorical(torch.from_numpy(np.ones(5))))
        index = index.item()
        op = self.binaryOps[index]
        opfn = op[0]
        ffn = f[0]
        gfn = g[0]

        return lambda x: opfn(ffn(x), gfn(x)), "(" + op[1] + " " + f[1] + " " + g[1] + ")"

    def randomArithmeticExpression(self):
        with name_count():
            d1 = pyro.sample('d1', dist.Bernoulli(probs=torch.tensor([0.5])))
            if d1 == 1:
                return self.randomCombination(self.randomArithmeticExpression(), self.randomArithmeticExpression())
            else:
                d2 = pyro.sample('d2', dist.Bernoulli(probs=torch.tensor([0.5])))
                if d2 == 1:
                    return self.identity
                else:
                    return self.randomConstantFunction()

    def run(self):
        e = self.randomArithmeticExpression()
        f = e[0]

        for i in range(len(self.data)):
            x_i, y_i = self.data[i]
            func = f(x_i)
            pyro.sample('f_{}'.format(i), dist.Normal(y_i, 5), obs=func)

        print(e)
        print(e[1])
        return e[1]

    @classmethod
    def create_from_file(cls, filename):
        data= []
        file = open(filename, 'r')
        lines = file.readlines()

        for line in lines:
            pair = line.split(" ")
            x = int(pair[0])
            y = int(pair[1])

            data.append((x,y))

        return SR_Model(data)


sr = SR_Model.create_from_file("../data/sir.data")
sr.run()

