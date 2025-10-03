import numpy as np
from matplotlib import pyplot as plt


# activation function
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def softplus(x): return np.log(1 + np.exp(x))

# numerical differentiation
def num_differentiation(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)

# data
x = np.linspace(-5, 5, 100)
h = 1e-8

f = sigmoid
# f = relu
# f = tanh
# f = softplus

# calculate
fx = f(x)
gx = num_differentiation(f, x, h)

# visualization
plt.figure(figsize= (7, 6))
plt.plot(x, fx, label= "function f(x)")
plt.plot(x, gx, label= "derivatives g(x)")
plt.axvline(x = 0, ls= "--", lw= 0.5, c= "gray")
plt.legend()
plt.show()

