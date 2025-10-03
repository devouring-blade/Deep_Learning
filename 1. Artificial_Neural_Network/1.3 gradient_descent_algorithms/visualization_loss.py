from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import alpha


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

# data
x = np.random.rand(50, 3)
y = np.float64(np.random.rand(50, 1) > 0.5)

# weights, no biases
wh = np.random.rand(3, 20)
wo = np.random.rand(20, 1)

# mse loss function
def loss(w1, w2):
    wh[0][0] = w1
    wo[0][0] = w2
    h = np.tanh(np.dot(x, wh))
    y_hat = sigmoid(np.dot(h, wo))
    return mse(y, y_hat)

w1, w2 = np.meshgrid(np.arange(-15, 15, 0.1),
                              (-15, 15, 0.1))
zs = np.array([loss(a, b) for a, b in zip(np.ravel(w1), np.ravel(w2))])
z = zs.reshape(w1.shape)

fig = plt.figure(figsize= (7, 7))
ax = fig.add_subplot(111, projection= "3d")
ax.plot_surface(w1, w2, z, alpha = 0.7)

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("loss")
ax.azim = 40
ax.elev = 60
plt.show()


