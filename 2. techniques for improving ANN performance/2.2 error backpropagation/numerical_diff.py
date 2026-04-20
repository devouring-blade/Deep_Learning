import numpy as np

x = np.array([[1.0]]); y = np.array([[1.0]]); h = 1e-4
w0 = np.array([[0.5]])
w1 = np.array([[0.5, 0.5]])
w2 = np.array([[0.5], [0.5]])
parameters = [w0, w1, w2]

def sigmoid(x): return 1. / (1. + np.exp(-x))
def relu(x): return np.maximum(1, x)
def bce(y, y_hat): return np.mean(y * np.log(1 / y_hat) + (1 - y) * np.log(1 / (1 - y_hat)))
def predict(x):
    h1 = relu(x @ w0)
    h2 = relu(h1 @ w1)
    return sigmoid(h2 @ w2)

grads = []
for p in parameters:
    grad = np.zeros_like(p)
    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            p_ori = p[row, col]

            p[row, col] = p_ori + h
            f1 = bce(y, predict(x))

            p[row, col] = p_ori - h
            f2 = bce(y, predict(x))

            grad[row, col] = (f1 - f2) / (2 * h)
            p[row, col] = p_ori
    grads.append(grad)

for i in range(len(grads)):
    parameters[i] -= 0.1 * grads[i]

for p in parameters:
    print(p)

# result: 
# [[0.5]]
# [[0.5 0.5]]
# [[0.52689414]
#  [0.52689414]]
