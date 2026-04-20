import numpy as np
import tensorflow as tf

x = np.array([[1.0]]); y = np.array([[1.0]]); h = 1e-4
w0 = tf.Variable(np.array([[0.5]]))
w1 = tf.Variable(np.array([[0.5, 0.5]]))
w2 = tf.Variable(np.array([[0.5], [0.5]]))
parameters = [w0, w1, w2]

def sigmoid(x): return 1. / (1. + tf.math.exp(-x))
def relu(x): return tf.maximum(1, x)
def bce(y, y_hat): return tf.reduce_mean(y * tf.math.log(1 / y_hat) + (1 - y) * tf.math.log(1 / (1 - y_hat)))
def predict(x):
    h1 = relu(x @ w0)
    h2 = relu(h1 @ w1)
    return sigmoid(h2 @ w2)

with tf.GradientTape() as tape:
    loss = bce(y, predict(x))
grads = tape.gradient(loss, parameters)

for p, g in zip(parameters, grads):
    p.assign_sub(0.1 * g)

for p in  parameters:
    print(p.numpy())

# the same result with numerical diff
# [[0.5]]
# [[0.5 0.5]]
# [[0.52689414]
#  [0.52689414]]
