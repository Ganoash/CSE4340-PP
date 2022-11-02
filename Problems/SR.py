import numpy as np
import pyro.distributions as dist
import torch
import math
import pyro
from pyro.contrib.autoname import name_count
import sys
from pyro.infer import config_enumerate


class SR_Model:

    def __init__(self):
        self.plus = lambda a, b: a + b, '+'
        self.multiply = lambda a, b: round(a*b,0), '*'
        self.divide = lambda a, b: round(a/b, 0), '/'
        self.minus = lambda a, b: a - b, '-'
        self.power = lambda a, b: math.pow(a,b), '**'
        self.binaryOps = [self.plus, self.multiply, self.divide, self.minus, self.power]
        self.identity = lambda x: x, 'x'
        self.rec_depth = 0

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
        d1 = pyro.sample('d1', dist.Bernoulli(probs=torch.tensor([0.5])))
        # We have to limit the recursive depth, otherwise it can get stuck
        if d1 == 1 and self.rec_depth < 100:
            self.rec_depth += 1
            return self.randomCombination(self.randomArithmeticExpression(), self.randomArithmeticExpression())
        else:
            d2 = pyro.sample('d2', dist.Bernoulli(probs=torch.tensor([0.5])))
            if d2 == 1:
                return self.identity
            else:
                return self.randomConstantFunction()

    @name_count
    def run(self, data):
        self.rec_depth = 0
        self.data = data
        e = self.randomArithmeticExpression()
        f = e[0]

        # We need to limit how often it can try again, otherwise it will get stuck in a loop.
        count = 0
        i = 0
        while i < len(self.data):
            if count >= 10:
                e = self.identity
            x_i, y_i = self.data[i]
            i += 1
            try:
                val_func = f(x_i)
                pyro.sample('f_{}'.format(i), dist.Normal(torch.tensor(y_i), 5), obs=torch.tensor(val_func))
            except:
                self.rec_depth = 0
                e = self.randomArithmeticExpression()
                f = e[0]
                i = 0
        count += 1

        return e[1]

    @classmethod
    def create_from_file(cls, filename):
        data= []
        file = open(filename, 'r')
        lines = file.readlines()

        for line in lines:
            pair = line.split(" ")
            x = float(pair[0])
            y = float(pair[1])

            data.append((x,y))

        return data


if __name__ == "__main__":
    args = sys.argv[1:]
    file = args[0]
    samples = args[1]
    data = SR_Model.create_from_file(str(file))
    sr = SR_Model()

    importance = pyro.infer.Importance(sr.run, guide=None, num_samples=int(samples))

    out = importance.run(data)

    normalized = out.get_normalized_weights()
    weights = []
    for i in normalized:
        weights.append(i.item())
    values = []
    for i in out.exec_traces:
        values.append(i.nodes["_RETURN"]["value"])

    combined = zip(weights, values)

    sort = sorted(combined, key=lambda x: x[0], reverse=True)

    count = 0
    result = []
    flag1 = False
    for i in sort:
        if count >= 5:
            break
        if i[1] not in result and count < 5:
            result.append(i[1])
            count += 1

    for i in result:
        print(i)
