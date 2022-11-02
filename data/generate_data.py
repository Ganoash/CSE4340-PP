import random


def generate(fn, n):
    res = []
    for i in range(n):
        x = random.randint(1, 100)
        y = fn(x)
        res.append([x, y])
    return res

def write_to_file(list):
    f = open("sr.data", 'w')
    for i in list:
        f.write(str(i[0]) + " " + str(i[1])+"\n")
    f.close()


fn = lambda x: x * 3
generated = generate(fn, 10)
write_to_file(generated)
