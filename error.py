import random
dataset = "datasets/testdata.train"


b = 0.93

N = 10000

nx = 10

A = [random.uniform(-1, 1) for i in range(nx)]

ea = sum(A) / len(A)

# E[y] = 5 * 0.5 + 3 = 5.5

out = ""
out += str(N) + " " + str(nx) + ' 1\n'
for i in range(N):
    X = [random.uniform(0.0, 1.0) for i in range(nx)]
    y1 = sum(A[i] * X[i] for i in range(nx)) + b

    err = random.gauss(0.0, 0.01)
    y1 += err
    y1 -= ea * 0.5 - b
    out += " ".join([str(x) for x in X]) + " " + str(y1) + "\n"


with open(dataset, 'w') as f:
    f.write(out)
  